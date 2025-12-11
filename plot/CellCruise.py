# This section of code is developed by modifying CellOracle

import logging
import warnings
from copy import deepcopy
import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Union, Tuple
import pandas as pd
import scipy.stats
from numba import jit
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm as normal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

import scanpy as sc

import seaborn as sns
import anndata as ad
from sklearn.decomposition import PCA

from velocyto.estimation import colDeltaCorpartial, colDeltaCor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

from adjustText import adjust_text
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from scipy.cluster.hierarchy import linkage, cut_tree


@jit(
    signature_or_function="Tuple((float64[:,:], int64[:,:], int64[:]))(int64[:,:], float64[:, :], int64[:], int64, "
                          "int64, boolean)",
    nopython=True)
def balance_knn_loop(dsi: np.ndarray, dist: np.ndarray, lsi: np.ndarray, maxl: int, k: int,
                     return_distance: bool) -> Tuple:
    assert dsi.shape[1] >= k, "sight needs to be bigger than k"
    # numba signature "Tuple((int64[:,:], float32[:, :], int64[:]))(int64[:,:], int64[:], int64, int64, bool)"
    dsi_new = -1 * np.ones((dsi.shape[0], k + 1), np.int64)  # maybe d.shape[0]
    l = np.zeros(dsi.shape[0], np.int64)
    if return_distance:
        dist_new = np.zeros(dsi_new.shape, np.float64)
    for i in range(dsi.shape[0]):  # For every node
        el = lsi[i]
        p = 0
        j = 0
        for j in range(dsi.shape[1]):  # For every other node it is connected (sight)
            if p >= k:
                break
            m = dsi[el, j]
            if el == m:
                dsi_new[el, 0] = el
                continue
            if l[m] >= maxl:
                continue
            dsi_new[el, p + 1] = m
            l[m] = l[m] + 1
            if return_distance:
                dist_new[el, p + 1] = dist[el, j]
            p += 1
        if (j == dsi.shape[1] - 1) and (p < k):
            while p < k:
                dsi_new[el, p + 1] = el
                dist_new[el, p + 1] = dist[el, 0]
                p += 1
    if not return_distance:
        dist_new = np.ones_like(dsi_new, np.float64)
    return dist_new, dsi_new, l


def knn_balance(dsi: np.ndarray, dist: np.ndarray = None, maxl: int = 200, k: int = 60) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    dsi = dsi.astype("int64").copy(order="C")
    l = np.bincount(dsi.flat[:], minlength=dsi.shape[0])
    lsi = np.argsort(l, kind="mergesort")[::-1]

    if dist is None:
        dist = np.ones(dsi.shape, dtype="float64").copy(order="C")
        dist[:, 0] = 0

        # Change data type and contingency for numba calculation. Added by Kenji 2021/1/21
        dsi = dsi.astype("int64").copy(order="C")
        dist = dist.astype("float64").copy(order="C")
        lsi = lsi.astype("int64").copy(order="C")
        return balance_knn_loop(dsi, dist, lsi, maxl, k, return_distance=False)
    else:
        # Change data type and contingency for numba calculation. Added by Kenji 2021/1/21
        dsi = dsi.astype("int64").copy(order="C")
        dist = dist.astype("float64").copy(order="C")
        lsi = lsi.astype("int64").copy(order="C")
        return balance_knn_loop(dsi, dist, lsi, maxl, k, return_distance=True)


class BalancedKNN:
    def __init__(self, k: int = 50, sight_k: int = 100, maxl: int = 200,
                 mode: str = "distance", metric: str = "euclidean", n_jobs: int = 4) -> None:
        self.k = k
        self.sight_k = sight_k
        self.maxl = maxl
        self.mode = mode
        self.metric = metric
        self.n_jobs = n_jobs
        self.dist_new = self.dsi_new = self.l = None  # type: np.ndarray
        self.bknn = None  # type: sparse.csr_matrix

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    def fit(self, data: np.ndarray, sight_k: int = None) -> Any:
        """Fits the model

        data: np.ndarray (samples, features)
            np
        sight_k: int
            the farthest point that a node is allowed to connect to when its closest neighbours are not allowed
        """
        self.data = data
        self.fitdata = data
        if sight_k is not None:
            self.sight_k = sight_k
        logging.debug(f"First search the {self.sight_k} nearest neighbours for {self.n_samples}")
        if self.metric == "correlation":
            self.nn = NearestNeighbors(n_neighbors=self.sight_k + 1, metric=self.metric, n_jobs=self.n_jobs,
                                       algorithm="brute")
        else:
            self.nn = NearestNeighbors(n_neighbors=self.sight_k + 1, metric=self.metric, n_jobs=self.n_jobs,
                                       leaf_size=30)
        self.nn.fit(self.fitdata)
        return self

    def kneighbors(self, X: np.ndarray = None, maxl: int = None, mode: str = "distance") -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """Finds the K-neighbors of a point.

            Returns indices of and distances to the neighbors of each point.

            Parameters
            ----------
            X : array-like, shape (n_query, n_features),
                The query point or points.
                If not provided, neighbors of each indexed point are returned.
                In this case, the query point is not considered its own neighbor.

            maxl: int
                max degree of connectivity allowed

            mode : "distance" or "connectivity"
                Decides the kind of output

            Returns
            -------
            dist_new : np.ndarray (samples, k+1)
                distances to the NN
            dsi_new : np.ndarray (samples, k+1)
                indexes of the NN, first column is the sample itself
            l: np.ndarray (samples)
                l[i] is the number of connections from other samples to the sample i

            NOTE:
            First column (0) correspond to the sample itself, the nearest nenigbour is at the second column (1)

        """
        if X is not None:
            self.data = X
        if maxl is not None:
            self.maxl = maxl

        self.dist, self.dsi = self.nn.kneighbors(self.data, return_distance=True)
        logging.debug(
            f"Using the initialization network to find a {self.k}-NN graph with maximum connectivity of {self.maxl}")
        self.dist_new, self.dsi_new, self.l = knn_balance(self.dsi, self.dist, maxl=self.maxl, k=self.k)

        if mode == "connectivity":
            self.dist = np.ones_like(self.dsi)
            self.dist[:, 0] = 0
        return self.dist_new, self.dsi_new, self.l

    def kneighbors_graph(self, X: np.ndarray = None, maxl: int = None, mode: str = "distance") -> sparse.csr_matrix:
        """Retrun the K-neighbors graph as a sparse csr matrix

            Parameters
            ----------
            X : array-like, shape (n_query, n_features),
                The query point or points.
                If not provided, neighbors of each indexed point are returned.
                In this case, the query point is not considered its own neighbor.

            maxl: int
                max degree of connectivity allowed

            mode : "distance" or "connectivity"
                Decides the kind of output
            Returns
            -------
            neighbor_graph : scipy.sparse.csr_matrix
                The values are either distances or connectivity dependig of the mode parameter

            NOTE: The diagonal will be zero even though the value 0 is actually stored

        """
        dist_new, dsi_new, l = self.kneighbors(X=X, maxl=maxl, mode=mode)
        logging.debug("Returning sparse matrix")
        self.bknn = sparse.csr_matrix((np.ravel(dist_new),
                                       np.ravel(dsi_new),
                                       np.arange(0, dist_new.shape[0] * dist_new.shape[1] + 1, dist_new.shape[1])),
                                      (self.n_samples,
                                       self.n_samples))
        return self.bknn

    def smooth_data(self, data_to_smooth: np.ndarray, X: np.ndarray = None, maxl: int = None,
                    mutual: bool = False, only_increase: bool = True) -> np.ndarray:
        """Use the wights learned from knn to smooth any data matrix

        Arguments
        ---------
        data_to_smooth: (features, samples) !! NOTE !! this is different from the input (for speed issues)
            if the data is provided (samples, features), this will be detected and
            the correct operation performed at cost of some effciency
            In the case where samples == samples then the shape (features, samples) will be assumed

        """
        if self.bknn is None:
            assert (X is None) and (maxl is None), "graph was already fit with different parameters"
            self.kneighbors_graph(X=X, maxl=maxl, mode=self.mode)
        if mutual:
            connectivity = make_mutual(self.bknn > 0)
        else:
            connectivity = self.bknn.T > 0
        connectivity = connectivity.tolil()
        connectivity.setdiag(1)
        w = connectivity_to_weights(connectivity).T
        assert np.allclose(w.sum(0), 1), "weight matrix need to sum to one over the columns"
        if data_to_smooth.shape[1] == w.shape[0]:
            result = sparse.csr_matrix.dot(data_to_smooth, w)
        elif data_to_smooth.shape[0] == w.shape[0]:
            result = sparse.csr_matrix.dot(data_to_smooth.T, w).T
        else:
            raise ValueError(f"Incorrect size of matrix, none of the axis correspond to the one of graph. {w.shape}")

        if only_increase:
            return np.maximum(result, data_to_smooth)
        else:
            return result


def _adata_to_matrix(adata, layer_name, transpose=True):
    if isinstance(adata.layers[layer_name], np.ndarray):
        matrix = adata.layers[layer_name].copy()
    else:
        matrix = adata.layers[layer_name].todense().A.copy()

    if transpose:
        matrix = matrix.transpose()

    return matrix.copy(order="C")


def perform_PCA(adata, n_components: int = None, div_by_std: bool = False) -> None:
    X = _adata_to_matrix(adata, "normalized_count")

    pca = PCA(n_components=n_components)
    if div_by_std:
        pcs = pca.fit_transform(X.T / X.std(0))
    else:
        pcs = pca.fit_transform(X.T)

    adata.obsm['pcs'] = pcs
    return adata


def connectivity_to_weights(mknn: sparse.csr_matrix, axis: int = 1) -> sparse.lil_matrix:
    if type(mknn) is not sparse.csr_matrix:
        mknn = mknn.tocsr()
    return mknn.multiply(1. / sparse.csr_matrix.sum(mknn, axis=axis))


def knn_distance_matrix(data: np.ndarray, metric: str = None, k: int = 40, mode: str = 'connectivity',
                        n_jobs: int = 4) -> sparse.csr_matrix:
    if metric == "correlation":
        nn = NearestNeighbors(n_neighbors=k, metric="correlation", algorithm="brute", n_jobs=n_jobs)
        nn.fit(data)
        return nn.kneighbors_graph(X=None, mode=mode)
    else:
        nn = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs, )
        nn.fit(data)
        return nn.kneighbors_graph(X=None, mode=mode)


def convolve_by_sparse_weights(data: np.ndarray, w: sparse.csr_matrix) -> np.ndarray:
    w_ = w.T
    assert np.allclose(w_.sum(0), 1), "weight matrix need to sum to one over the columns"
    return sparse.csr_matrix.dot(data, w_)


def knn_imputation(adata, k: int = None, metric: str = "euclidean", diag: float = 1,
                   n_pca_dims: int = 50, n_jobs: int = 8,
                   balanced: bool = True, b_sight: int = None, b_maxl: int = None
                   ) -> None:
    X = _adata_to_matrix(adata, "normalized_count")

    N = adata.shape[0]  # cell number

    if k is None:
        k = int(N * 0.025)
    if b_sight is None and balanced:
        b_sight = int(k * 8)
    if b_maxl is None and balanced:
        b_maxl = int(k * 4)

    space = adata.obsm['pcs'][:, :n_pca_dims]
    # space = adata.obsm['X_pca']

    if balanced:
        bknn = BalancedKNN(k=k, sight_k=b_sight, maxl=b_maxl,
                           metric=metric, mode="distance", n_jobs=n_jobs)
        bknn.fit(space)
        knn = bknn.kneighbors_graph(mode="distance")
    else:

        knn = knn_distance_matrix(space, metric=metric, k=k,
                                  mode="distance", n_jobs=n_jobs)
    connectivity = (knn > 0).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore")  # SparseEfficiencyWarning: Changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient.
        connectivity.setdiag(diag)
    knn_smoothing_w = connectivity_to_weights(connectivity)

    ###
    Xx = convolve_by_sparse_weights(X, knn_smoothing_w)
    adata.layers["imputed_count"] = Xx.transpose().copy()

    return adata


@jit(nopython=True)
def numba_random_seed(value: int) -> None:
    """Same as np.random.seed but for numba"""
    np.random.seed(value)


def estimate_transition_prob(adata, embedding_name='X_umap',
                             n_neighbors: int = None,
                             goi=None,
                             n_jobs: int = 4, threads: int = None,
                             random_seed: int = 20230622) -> None:
    # Set the seed
    numba_random_seed(random_seed)
    X = _adata_to_matrix(adata, "imputed_count")  # [:, :ndims]
    delta_X = _adata_to_matrix(adata, "delta_X")
    embedding = adata.obsm[embedding_name]

    if n_neighbors is None:
        n_neighbors = int(adata.shape[0] / 10)

    np.random.seed(random_seed)

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    nn.fit(embedding)  # NOTE should support knn in high dimensions
    embedding_knn = nn.kneighbors_graph(mode="connectivity")

    # Based on the position where deltaX is 0, it indicates that the TG is not regulated by TF.
    # Therefore, the corresponding element at that position is also set to 0.
    zero_rows = np.all(delta_X == 0, axis=1)
    X = X[~zero_rows]
    delta_X = delta_X[~zero_rows]

    # Calculate the Pearson correlation coefficient.
    corrcoef = colDeltaCor(X, delta_X, threads=threads)

    if np.isnan(corrcoef).any():
        max_value = np.nanmax(corrcoef)
        corrcoef = np.where(np.isnan(corrcoef), max_value, corrcoef)

    # print(f"{goi}_deltaX: {delta_X.shape}")

    np.fill_diagonal(corrcoef, 0)

    adata.obsp['embedding_knn'] = embedding_knn
    adata.obsp['corrcoef'] = corrcoef


def calculate_embedding_shift(adata, embedding_name='X_umap', sigma_corr: float = 0.05) -> None:
    embedding_knn = adata.obsp['embedding_knn']
    corrcoef = adata.obsp['corrcoef']

    embedding = adata.obsm[embedding_name]

    # Replace the value of infinity with the maximum value in the matrix
    matrix = np.exp(corrcoef / sigma_corr)

    max_value = np.max(matrix[np.isfinite(matrix)])
    matrix[np.isinf(matrix)] = max_value

    transition_prob = matrix * embedding_knn.A  # naive
    # transition_prob = np.exp(corrcoef / sigma_corr) * embedding_knn.A  # naive
    transition_prob /= transition_prob.sum(1)[:, None]

    unitary_vectors = embedding.T[:, None, :] - embedding.T[:, :, None]  # shape (2,ncells,ncells)
    with np.errstate(divide='ignore', invalid='ignore'):
        unitary_vectors /= np.linalg.norm(unitary_vectors, ord=2, axis=0)  # divide by L2
        np.fill_diagonal(unitary_vectors[0, ...], 0)  # fix nans
        np.fill_diagonal(unitary_vectors[1, ...], 0)

    delta_embedding = (transition_prob * unitary_vectors).sum(2)
    delta_embedding -= (embedding_knn.A * unitary_vectors).sum(2) / embedding_knn.sum(1).A.T
    delta_embedding = delta_embedding.T

    adata.obsm['delta_embedding'] = delta_embedding


def plot_quiver(adata, colorandum, embedding_name='X_umap', ax=None, scale=30, color=None, s=5,
                show_background=False, args={"linewidths": 0.25, "width": 0.004}):
    embedding = adata.obsm[embedding_name]
    delta_embedding = adata.obsm['delta_embedding']

    # Control the relative proportion of arrow sizes for all cells.
    max_abs = np.nanmax(np.abs(delta_embedding))
    factor = 1 / max_abs
    delta_embedding = delta_embedding * factor

    if ax is None:
        ax = plt

    cell_idx_use = None
    if cell_idx_use is None:
        ix_choice = np.arange(embedding.shape[0])
    else:
        ix_choice = cell_idx_use

    # Plot whole cell with lightgray
    if show_background:
        ax.scatter(embedding[:, 0], embedding[:, 1],
                   c="lightgray", alpha=1, s=s, **args)

    # ax.scatter(embedding[ix_choice, 0], embedding[ix_choice, 1],
    #            c="lightgray", alpha=0.2, edgecolor=(0, 0, 0, 1), s=s, **args)

    if color is None:
        color = colorandum[ix_choice]

    quiver_kwargs = dict(headaxislength=7, headlength=11, headwidth=8,
                         linewidths=0.25, width=0.0045, edgecolors="k",
                         color=color, alpha=1)

    quiver = delta_embedding

    ax.quiver(embedding[ix_choice, 0], embedding[ix_choice, 1],
              quiver[ix_choice, 0],
              quiver[ix_choice, 1],
              scale=scale, **quiver_kwargs)

    ax.axis("off")
    return scale


def get_palette(adata, cname):
    c = [i.upper() for i in adata.uns[f"{cname}_colors"]]
    try:
        col = adata.obs[cname].cat.categories
        pal = pd.DataFrame({"palette": c}, index=col)
    except:
        col = adata.obs[cname].cat.categories
        c = c[:len(col)]
        pal = pd.DataFrame({"palette": c}, index=col)
    return pal


def get_clustercolor_from_anndata(adata, cluster_name, return_as):
    def float2rgb8bit(x):
        x = (x * 255).astype("int")
        x = tuple(x)

        return x

    def rgb2hex(rgb):
        return '#%02x%02x%02x' % rgb

    def float2hex(x):
        x = float2rgb8bit(x)
        x = rgb2hex(x)
        return x

    def hex2rgb(c):
        return (int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16), 255)

    pal = get_palette(adata, cluster_name)
    if return_as == "palette":
        return pal
    elif return_as == "dict":
        col_dict = {}
        for i in pal.index:
            col_dict[i] = np.array(hex2rgb(pal.loc[i, "palette"])) / 255
        return col_dict
    else:
        raise ValueErroe("return_as")
    return 0


def plot_score_diagram(inner_product_score, ntop=3, fig_save_path=None,
                       adata=None, cluster_name=None, cut_tree_number=4):
    mark = 0
    for k, v in inner_product_score.items():
        temp = inner_product_score[k].groupby('pseudotime_id')['score'].sum().reset_index(). \
            pivot_table(values='score', index='pseudotime_id').T
        temp.index = [k]
        if mark == 0:
            inner_product_all = temp
            mark = 1
        else:
            inner_product_all = pd.concat([inner_product_all, temp], axis=0, join='outer')

    max_rows = [inner_product_all[col].nlargest(ntop).index.tolist() for col in inner_product_all.columns]
    min_rows = [inner_product_all[col].nsmallest(ntop).index.tolist() for col in inner_product_all.columns]
    max_values = [inner_product_all[col].nlargest(ntop).tolist() for col in inner_product_all.columns]
    min_values = [inner_product_all[col].nsmallest(ntop).tolist() for col in inner_product_all.columns]

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[10, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    labels = []
    for i in range(inner_product_all.shape[1]):
        ax1.scatter([i + 0.5] * ntop, max_values[i], color='blue', alpha=0.8, s=2)
        ax1.scatter([i + 0.5] * ntop, min_values[i], color='red', alpha=0.8, s=2)
        texts = []

        change = -1
        for j in range(3):
            if change < 0:
                texts.append(ax1.text(i + 0.5, max_values[i][j], max_rows[i][j], ha='left', va='top',
                                      fontsize=12, color='blue', alpha=1))
                texts.append(ax1.text(i + 0.5, min_values[i][j], min_rows[i][j], ha='left', va='bottom',
                                      fontsize=12, color='red', alpha=1))
                change = change * (-1)

            else:
                texts.append(ax1.text(i + 0.5, max_values[i][j], max_rows[i][j], ha='right', va='top',
                                      fontsize=12, color='blue', alpha=1))
                texts.append(ax1.text(i + 0.5, min_values[i][j], min_rows[i][j], ha='right', va='bottom',
                                      fontsize=12, color='red', alpha=1))
                change = change * (-1)

        #     adjust_text(texts, expand_points=1)

        for j in range(ntop):
            label = f"{max_rows[i][j]}"
            labels.append(label)
            label = f"{min_rows[i][j]}"
            labels.append(label)

    max_value = inner_product_all.values.max()
    min_value = inner_product_all.values.min()
    abs_max = max(abs(max_value), abs(min_value)) + 0.5
    ax1.set_ylim([-abs_max, abs_max])
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax1.set_xticks(range(inner_product_all.shape[1]), fontsize=12)
    ax1.set_xticklabels(inner_product_all.columns)

    ax1.set_ylabel('Score', fontsize=12)

    grouped_data = adata.obs.groupby(cluster_name).agg({'Pseudotime': ['min', 'max']})

    grouped_data.columns = ['_'.join(col).strip() for col in grouped_data.columns.values]
    grouped_data = grouped_data * 10

    for i, (segment, limits) in enumerate(
            grouped_data.loc[grouped_data.sort_values(by='Pseudotime_min').index.tolist(),].iterrows()):
        ax2.barh(i + 1, height=0.9, width=limits[1] - limits[0], left=limits[0],
                 color=f'C{i}', label=f'{segment}: {limits[0]:.2f}-{limits[1]:.2f}')

    ax2.set_ylim(0.5, 0.5 + len(adata.obs[cluster_name].unique()))

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.tick_params(axis='both', which='both', length=0)
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])

    fig.tight_layout()

    handles, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels2, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=20)

    if not os.path.exists(f'{fig_save_path}/scatter_plot/'):
        os.makedirs(f'{fig_save_path}/scatter_plot/')

    plt.savefig(f'{fig_save_path}/scatter_plot/scatter_plot.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{fig_save_path}/scatter_plot/scatter_plot.pdf', dpi=600, bbox_inches='tight')

    unique_dict = {x: True for x in labels}
    unique_list = [x for x in unique_dict.keys()]

    inner_product_all_ntop = inner_product_all.loc[unique_list,]

    Z = linkage(inner_product_all_ntop, method='ward')

    clusters = cut_tree(Z, n_clusters=cut_tree_number)

    inner_product_all_ntop['cluster'] = pd.Series(clusters[:, 0], index=inner_product_all_ntop.index)

    color_bar_max = int(inner_product_all_ntop.abs().max().max())

    for j, (label, group) in enumerate(inner_product_all_ntop.groupby('cluster')):
        group = group.drop('cluster', axis=1)

        heatmap = sns.clustermap(group, cmap='YlGn', annot=True, fmt='.2f', figsize=(6, group.shape[0]),
                                 row_cluster=True, col_cluster=False, vmin=-color_bar_max, vmax=color_bar_max,
                                 cbar_pos=[1.05, 0.5, 0.05, 0.2], cbar_kws={'label': 'Score', 'drawedges': False})

        for text in heatmap.ax_heatmap.texts:
            if float(text.get_text()) > 0:
                text.set_color('blue')
            elif float(text.get_text()) < 0:
                text.set_color('red')
            else:
                text.set_color('white')

        plt.xticks([])

        if not os.path.exists(f"{fig_save_path}/heatmap_plot/"):
            os.makedirs(f"{fig_save_path}/heatmap_plot/")

        plt.savefig(f"{fig_save_path}/heatmap_plot/_group{j}.png",
                    dpi=300, bbox_inches='tight')

        plt.savefig(f"{fig_save_path}/heatmap_plot/_group{j}.pdf",
                    dpi=300, bbox_inches='tight')


def plot_score_every_TF(inner_product_score, fig_save_path=None,
                        adata=None, cluster_name=None, TF_list='all'):
    if TF_list != 'all':
        inner_product_score_ntop = inner_product_score.loc[TF_list,]
    else:
        inner_product_score_ntop = inner_product_score

    ymax = []
    for i, (key, value) in enumerate(inner_product_score_ntop.items()):
        temp_max = inner_product_score_ntop[key]['score'].abs().max() * 1.1
        ymax.append(temp_max)
    ymax = max(ymax)

    for m, (k, v) in enumerate(inner_product_score_ntop.items()):

        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[10, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        for j in range(10):
            if j not in inner_product_score_ntop[k].columns and str(j) not in inner_product_score_ntop[k].columns:
                inner_product_score_ntop[k].insert(loc=j, column=j,
                                                   value=[0 for _ in range(inner_product_score_ntop[k].shape[0])])

        ax1.axhline(y=0, color='gray', linestyle='--')

        sns.lineplot(x='pseudotime'
                     , y='score'
                     , hue='pseudotime_id'
                     , data=inner_product_score_ntop[k]
                     , ax=ax1
                     , palette='tab10', legend=False, estimator=np.mean, errorbar=('ci', 68))

        ax1.set_ylim((-ymax, ymax))
        ax1.set_xlabel('Pseudotime', fontsize=20)
        ax1.set_ylabel('Score', fontsize=20)
        ax1.set_title(f'KO: {k}', fontsize=30)

        ax1.set_xticklabels([])

        grouped_data = adata.obs.groupby(cluster_name).agg({'Pseudotime': ['min', 'max']})

        grouped_data.columns = ['_'.join(col).strip() for col in grouped_data.columns.values]
        grouped_data = grouped_data * 10

        for i, (segment, limits) in enumerate(
                grouped_data.loc[grouped_data.sort_values(by='Pseudotime_min').index.tolist(),].iterrows()):
            ax2.barh(i + 1, height=0.9, width=limits[1] - limits[0], left=limits[0],
                     color=f'C{i}', label=f'{segment}: {limits[0]:.2f}-{limits[1]:.2f}')

            ax2.set_ylim(0.5, 0.5 + len(adata.obs[cluster_name].unique()))

            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.tick_params(axis='both', which='both', length=0)
            ax2.set_yticklabels([])
            ax2.set_xticklabels([])

            fig.tight_layout()

            handles, labels = ax2.get_legend_handles_labels()
            ax2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=20)

        if not os.path.exists(f"{fig_save_path}/line_chart"):
            os.makedirs(f"{fig_save_path}/line_chart")

        fig.savefig(f"{fig_save_path}/line_chart/KO_{k}.png", dpi=600)


def count_to_fpkm(counts, effLen):
    N = np.sum(counts)
    return np.exp(np.log(counts) + np.log(1e9) - np.log(effLen) - np.log(N))


def cluster_to_PseudoBulk(adata, cluster, species=None, cluster_name='celltype', database_path=None):
    count = pd.DataFrame(adata[adata.obs[cluster_name] == cluster,].layers['raw_count'],
                         index=adata[adata.obs[cluster_name] == cluster,].obs_names,
                         columns=adata[adata.obs[cluster_name] == cluster,].var_names)
    if species == 'human':
        ref = pd.read_csv(
            os.path.join(database_path, 'Homo_sapiens.GRCh38.109_gene_length.csv'),
            index_col=0)
        ref = ref.dropna()
    else:
        ref = pd.read_csv(
            os.path.join(database_path, 'Mus_musculus.GRCm38.102_gene_length.csv'),
            index_col=0)
        ref = ref.dropna()
    ref = ref.loc[:, ['gene_name', 'gene_length']]
    ref_final = ref.groupby('gene_name')['gene_length'].sum().reset_index()
    ref_final.set_index('gene_name', inplace=True)

    fpkm = count_to_fpkm(count.sum().values, ref_final.loc[count.columns, 'gene_length'].values)
    return pd.DataFrame(fpkm, index=adata[adata.obs[cluster_name] == cluster,].var_names, columns=[cluster])


def calculate_cosine_similarity(data1: pd.Series, data2: pd.Series):
    dot_product = np.dot(data1.values.flatten().tolist(), data2.values.flatten().tolist())

    norm_x = np.linalg.norm(data1.values.flatten().tolist())
    norm_y = np.linalg.norm(data2.values.flatten().tolist())

    cosine_similarity = dot_product / (norm_x * norm_y)
    return cosine_similarity


def record_the_scores_two_clusters(adata, species, cluster1, cluster2, cluster_name='celltype'):
    """
    Calculate the cosine similarity between the expression difference of two clusters
    and the deltaX corresponding to the first cluster
    :param adata:
    :param species:
    :param cluster1:
    :param cluster2:
    :param cluster_name:
    :return:
    """
    cluster1_fpkm = cluster_to_PseudoBulk(adata, species=species, cluster=cluster1, database_path=database_path)
    cluster2_fpkm = cluster_to_PseudoBulk(adata, species=species, cluster=cluster2, database_path=database_path)

    cluster1_cluster2 = pd.DataFrame(cluster2_fpkm.values - cluster1_fpkm.values,
                                     columns=[cluster1_fpkm.columns[0] + '_' + cluster2_fpkm.columns[0]],
                                     index=cluster1_fpkm.index)

    row_index = adata.obs[adata.obs[cluster_name] == cluster1].index[0]
    row_number = adata.obs.index.get_loc(row_index)

    df = pd.DataFrame(adata.layers['delta_X'][row_number], index=adata.var_names)

    return calculate_cosine_similarity(cluster1_cluster2, df.loc[cluster1_cluster2.index,].fillna(0))


def get_deltaX(adata, deltaX_path=None, goi=None):
    """
    :param TG:
    :param adata:
    :param deltaX_path: the path where save deltaX
    :return:
    """
    import glob
    selected_files = glob.glob(deltaX_path)

    deltaX_all = []
    df = pd.DataFrame()

    for i, data_temp in enumerate(selected_files):
        deltaX = pd.read_csv(data_temp, index_col=1)
        celltype = data_temp.split('deltax_')[-1].split('.csv')[0]

        if goi not in deltaX.columns:
            # continue
            deltaX[goi] = pd.NA
            print(f'{goi}not found')

        temp = deltaX[goi]
        temp.name = celltype

        deltaX_all.append(temp)

        if df.shape == (0, 0):
            df = temp
        else:
            df = pd.concat([df, temp], axis=1, join='outer')
    df = df.T

    # Fill the missing values with zeros
    df = df.reindex(columns=adata.var_names, fill_value=0)

    df = df.loc[:, adata.var_names.to_list()]

    xx = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    xx['celltype'] = adata.obs.celltype

    repeat_list = pd.value_counts(adata.obs.celltype).loc[df.index.tolist(),].tolist()
    df_repeat_list = []
    for i, (_, row) in enumerate(df.iterrows()):
        row_repeat = np.tile(row.values, (repeat_list[i], 1))
        df_repeat = pd.DataFrame(row_repeat, columns=df.columns)
        df_repeat['celltype'] = _

        df_repeat_list.append(df_repeat)

    df_new = pd.concat(df_repeat_list, ignore_index=True)
    df_new.fillna(0, inplace=True)

    name_order = df_new.celltype.unique().tolist()
    xx.celltype = xx.celltype.astype(str)
    xx_sorted = xx.sort_values(by='celltype', key=lambda x: x.map(dict(zip(name_order, range(len(name_order))))))
    xx_temp = pd.DataFrame(df_new.drop('celltype', axis=1).values,
                           index=xx_sorted.index,
                           columns=xx_sorted.columns[:-1])

    adata.layers['delta_X'] = xx_temp.loc[adata.obs_names,].values
    return adata


def plot_arrow(adata, embedding_name='X_umap', ax=None, scale=30, color=None, s=5,
               inner_product_score_=None, goi=None, cluster_name='celltype',
               quiver_kwargs=None
               ):
    color = 'black'
    if quiver_kwargs is None:
        quiver_kwargs = dict(headaxislength=7, headlength=11, headwidth=8,
                             linewidths=0.25, width=0.0045, edgecolors="k",
                             color=color, alpha=1)
    if ax is None:
        ax = plt

    inner_product_score_['start'] = inner_product_score_.index.map(lambda x: x.split('->')[0])
    inner_product_score_['next'] = inner_product_score_.index.map(lambda x: x.split('->')[1])

    embedding = adata.obsm[embedding_name]

    all_pos = (
        pd.concat([pd.DataFrame(embedding, columns=["x", "y"], index=adata.obs_names), adata.obs[[cluster_name]]],
                  axis=1)
        .groupby(cluster_name, observed=True)
        .median()
        .sort_index()
    )

    ax = sc.pl.embedding(adata, basis=embedding_name, color=cluster_name,
                         show=False, title=f'KO: {goi}')

    for _, row in inner_product_score_.iterrows():
        start = row['start']
        next = row['next']
        score = row[goi]

        start_x, start_y = int(all_pos.loc[start, 'x']), int(all_pos.loc[start, 'y'])
        end_x, end_y = int(all_pos.loc[next, 'x']), int(all_pos.loc[next, 'y'])

        angle = np.arctan2(end_y - start_y, end_x - start_x)  # The angle of a line segment

        # Draw an arrow
        # The length of the arrow can be adjusted according to your preference.
        arrow_length = score
        # The length of the arrowhead can be adjusted according to your preference.
        arrow_head_length = arrow_length * 0.5

        dx = arrow_length * np.cos(angle)
        dy = arrow_length * np.sin(angle)

        hx = arrow_head_length * np.cos(angle + np.pi / 6)
        hy = arrow_head_length * np.sin(angle + np.pi / 6)

        ax.quiver(start_x, start_y, hx, hy,
                  scale=scale, **quiver_kwargs)
        ax.axis("off")
    return None


def identifying_important_TFs(adata, goi=None, deltaX_path=None, trajectorys=None,
                              species=None, cluster_name=None, TG=None, database_path=None):
    import glob
    selected_files = glob.glob(deltaX_path)

    adata_used = adata[:, TG]
    for i, data_temp in enumerate(selected_files):
        deltaX = pd.read_csv(data_temp, index_col=1)
        celltype = data_temp.split('deltax_')[-1].split('.csv')[0]

        if goi not in deltaX.columns:
            continue

        temp = deltaX[goi]
        temp.name = celltype

        if i == 0:
            df = temp
        else:
            df = pd.concat([df, temp], axis=1, join='outer')
    df = df.T
    df = df.loc[:, adata_used.var_names.to_list()]
    df.fillna(0, inplace=True)

    for i, cluster_ in enumerate(adata_used.obs[cluster_name].unique().to_list()):
        fpkm_ = cluster_to_PseudoBulk(adata_used, cluster_, species=species,
                                      cluster_name=cluster_name, database_path=database_path)
        if i == 0:
            fpkm_all = fpkm_
        else:
            fpkm_all = pd.concat([fpkm_all, fpkm_], axis=1, join='outer')

    # calculate the deltaX for each cluster after knocking out the TF
    df = df.T
    deltaX_fpkm = df.values.copy(order="C")

    # calculate the pseudo fpkm for each cluster
    fpkm_all = fpkm_all.loc[df.index, df.columns]
    count_fpkm = fpkm_all.values.copy(order="C")

    zero_rows = np.all(deltaX_fpkm == 0, axis=1)
    count_fpkm = count_fpkm[~zero_rows]
    deltaX_fpkm = deltaX_fpkm[~zero_rows]

    count_fpkm = pd.DataFrame(count_fpkm, index=df.index[~zero_rows], columns=df.columns)
    deltaX_fpkm = pd.DataFrame(deltaX_fpkm, index=df.index[~zero_rows], columns=df.columns)

    inner_product_score_temp = {}
    for trajectory in trajectorys:
        idx = 0
        while idx < len(trajectory) - 1:
            start = trajectory[idx]
            start_next = trajectory[idx + 1]

            data1 = count_fpkm[start_next] - count_fpkm[start]
            data2 = deltaX_fpkm[start]

            # Calculate the fpkm of deltaX for each cluster after knocking out the TF,
            # only considering the TGs connected to the TF
            nor_zero_gene = data2.loc[data2.values != 0,].index
            data1 = data1.loc[nor_zero_gene,]
            data2 = data2.loc[nor_zero_gene,]

            # Calculate the cosine similarity between two adjacent clusters:
            dot_product = np.dot(data1.values.flatten().tolist(), data2.values.flatten().tolist())
            norm_x = np.linalg.norm(data1.values.flatten().tolist())
            norm_y = np.linalg.norm(data2.values.flatten().tolist())
            cosine_similarity = dot_product / (norm_x * norm_y)

            inner_product_score_temp[f'{start}->{start_next}'] = cosine_similarity
            idx = idx + 1

    inner_product_score_TF = pd.DataFrame.from_dict(inner_product_score_temp, orient='index', columns=[goi])
    return inner_product_score_TF

