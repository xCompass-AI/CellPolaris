import pandas as pd
import numpy as np
# from data_pro import data_pro
import torch
from torch import nn, optim
from scipy.stats import norm
import torch
from scipy import optimize
import numpy as np
from torch.distributions import Normal
import torch
import torch.nn as nn
import pandas as pd
# from data_pro import data_pro
from sklearn.preprocessing import MinMaxScaler
import time
import sys
from sklearn.preprocessing import StandardScaler
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
#

def data_pro1(Network_file,RNA_file,edge_num,name):
    columns = 50
    network = pd.read_table(Network_file)
    network = network[((network["TF"].isin(name)) & network["TG"].isin(name))]

    network = network[['TF', 'TG', 'Score']]

    network_sorted = network.sort_values('Score', ascending=False)

    RNA_exp = pd.read_csv(RNA_file)
    # print(RNA_exp)
    first_column_name = RNA_exp.columns[0]
    # print(first_column_name)
    RNA_exp.set_index(first_column_name,inplace=True,drop=True, append=False)
    # print(RNA_exp)
    RNA_exp = RNA_exp.iloc[:,:columns]
    
    gene_std = np.std(RNA_exp, axis=1)
    delete_gene_set = set(gene_std[gene_std < 1].index)
    network = network[~network['TF'].isin(delete_gene_set) & ~network['TG'].isin(delete_gene_set)]
    network = network[network['TF'] != network['TG']]
    network = network.iloc[:min(edge_num, network_sorted.shape[0]), :]

    min_val = network['Score'].min()
    max_val = network['Score'].max()
    network['Score'] = ((network['Score'] - min_val) / (max_val - min_val)) * 0.6 + 0.1

    top_TF_set = set(network.iloc[:, 0]) - set(network.iloc[:, 1])
    top_TF_exp = RNA_exp.loc[RNA_exp.index.isin(top_TF_set)]
    top_TF = top_TF_exp.index.to_list()
    # print(top_TF)
    # print(top_TF_exp)

    TG_set = list(set(network.iloc[:, 1]))
    TG_exp = RNA_exp.loc[RNA_exp.index.isin(TG_set)]
    TG = TG_exp.index.to_list()

    in_degree = network['TG'].value_counts()
    father_num = pd.DataFrame(in_degree)
    father_num = father_num.reindex(TG)

    # RNA_exp = RNA_exp.drop_duplicates()
    RNA_exp = RNA_exp.groupby(RNA_exp.index).mean()

    merged_1 = pd.merge(network, RNA_exp, left_on='TF', right_index=True, how='left')
    merged_2 = pd.merge(merged_1, RNA_exp, left_on='TG', right_index=True, how='left')
    print(merged_2)

    df = merged_2
    column_range1 = df.columns[3:53]
    column_range2 = df.columns[53:103]
    covariance_vector = np.zeros(edge_num)

    for i in range(df.shape[0]):
        covariance_vector[i] = np.cov(df[column_range1].values[i], df[column_range2].values[i])[0, 1]


    # column_3 = df['Score']
    #

    # result = covariance_vector * column_3
    #

    # df['Score'] = result
    edge_dataframe = df
    df['Cov'] = covariance_vector
    TF_high_exp = top_TF_exp.values.T.tolist()
    edge_all = []
    for j in range(len(TF_high_exp)):
        edge_subset = edge_dataframe.iloc[:, :3].join(edge_dataframe.iloc[:, -1]).join(
            edge_dataframe.iloc[:, j + 3]).join(edge_dataframe.iloc[:, j + columns + 3])
        # print(edge_subset)
        edge = edge_subset.values.tolist()
        edge_all.append(edge)
    return top_TF, TG, father_num.iloc[:, 0].to_list(),edge_all,TF_high_exp, TG_exp.values.T.tolist(),edge_dataframe


def get_tf_tg_exp(rna_addr):
    RNA_exp = pd.read_csv(rna_addr)
    print(RNA_exp)

    first_column_name = RNA_exp.columns[0]
    # print(first_column_name)

    RNA_exp.set_index(first_column_name, inplace=True, drop=True, append=False)
    RNA_exp = RNA_exp.groupby(RNA_exp.index).sum()

    


class Gauss_top_TF(nn.Module):
    def __init__(self, mu_init=None, sigma_init=None):
        super(Gauss_top_TF, self).__init__()
        if mu_init is None:
            raise ValueError('No mu_init as input.')
        if sigma_init is None:
            raise ValueError('No sigma_init as input.')
        self.mu = nn.Parameter(mu_init)
        self.sigma = nn.Parameter(torch.abs(sigma_init))

    def forward(self, x):
        dist = torch.distributions.Normal(self.mu, self.sigma)
        log_prob = dist.log_prob(x)
        return log_prob


class Gauss_TG(nn.Module):
    def __init__(self, mu_init=None, sigma_init=None):
        super(Gauss_TG, self).__init__()
        if mu_init is None:
            raise ValueError('No mu_init as input.')
        if sigma_init is None:
            raise ValueError('No sigma_init as input.')
        self.mu = nn.Parameter(mu_init)
        self.sigma = nn.Parameter(torch.abs(sigma_init))

    def forward(self, x):
        dist = torch.distributions.Normal(self.mu, self.sigma)
        log_prob = dist.log_prob(x)
        return log_prob

k_list = []
alpha_list = []
class Gauss_condition(nn.Module):
    def __init__(self, edge=None, k_init=0):
        super(Gauss_condition, self).__init__()
        if edge is None:
            raise ValueError('No edge infomation.')
        self.TF = edge[0]
        self.TG = edge[1]
        self.alpha = torch.FloatTensor([edge[2]])
        self.cov = torch.FloatTensor([edge[3]])
        self.k = nn.Parameter(torch.FloatTensor([k_init]))
        # self.TFexp = edge[3]

    def forward(self, x,y):
        if self.TF in TF_high:
            gTF_high_mu = names['gTF_high_%s' % self.TF].mu
            gTF_high_sigma = names['gTF_high_%s' % self.TF].sigma
            gTG_mu = names['gTG_%s' % self.TG].mu
            gTG_sigma = names['gTG_%s' % self.TG].sigma

            # loc = (gTG_mu + self.k * self.alpha* (y - gTF_high_mu)
            #        / torch.square(gTF_high_sigma)).relu()+0.01
            loc = (gTG_mu + self.k * self.cov * (y - gTF_mu)
                   / torch.square(gTF_sigma)).relu() + 0.01
            scale = torch.sqrt((torch.square(gTG_sigma) - torch.square(self.alpha)
                                 / torch.square(gTF_high_sigma)).relu()+0.01)

            dist = torch.distributions.Normal(loc, scale)
        else:
            gTF_mu = names['gTG_%s' % self.TF].mu
            gTF_sigma = names['gTG_%s' % self.TF].sigma
            gTG_mu = names['gTG_%s' % self.TG].mu
            gTG_sigma = names['gTG_%s' % self.TG].sigma

            # loc = (gTG_mu + self.k * self.alpha * (y - gTF_mu)
            #        / torch.square(gTF_sigma)).relu()+0.01
            loc = (gTG_mu + self.k * self.cov * (y - gTF_mu)
                   / torch.square(gTF_sigma)).relu() + 0.01
            scale = torch.sqrt((torch.square(gTG_sigma) - torch.square(self.alpha)
                                 / torch.square(gTF_sigma)).relu()+0.01)

            dist = torch.distributions.Normal(loc, scale)
        log_prob = dist.log_prob(x)
        return log_prob


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(len(TF_high)):
            names['gTF_high_%s' % TF_high[i]] = Gauss_top_TF(torch.FloatTensor([TF_high_mu[i]]),
                                                             torch.FloatTensor([TF_high_std[i]]))
        for i in range(len(TG)):
            names['gTG_%s' % TG[i]] = Gauss_TG(torch.FloatTensor([TG_mu[i]]), torch.FloatTensor([TG_std[i]]))
        for i in range(len(edge)):
            names['edge_%s' % i] = Gauss_condition(edge[i])
        self.params = nn.ModuleList([x for x in names.values() if isinstance(x, nn.Module)])


    def forward(self,TF_high_exp,TG_exp,edge):
        p = q = k = 0
        for i in range(len(TF_high)):
            names['prob_TF_high_%s' % TF_high[i]] = names['gTF_high_%s' % TF_high[i]](
                torch.tensor([TF_high_exp[i]]))
            p = p + names['prob_TF_high_%s' % TF_high[i]]
        for i in range(len(TG)):
            names['prob_TG_%s' % TG[i]] = names['gTG_%s' % TG[i]](torch.FloatTensor([TG_exp[i]]))
            q = q - torch.tensor(father_num[i] - 1).to(device, dtype=torch.float) * names['prob_TG_%s' % TG[i]]
        for i in range(len(edge)):
            names['prob_edge_%s' % i] = names['edge_%s' % i](torch.tensor([edge[i][5]]),torch.tensor([edge[i][4]]))
            k = k + names['prob_edge_%s' % i]

        for i in range(len(alpha_list)):
            g_res=torch.abs(torch.log(k_list[i] / alpha_list[i] + 1e-10))
            g=g+g_res

        return -(p + q + k + g)
        # return -(p + q + k)

def get_data(dataframe, param_addr,edge_num,TF_high, TG, edge,name):

    corr_data = dataframe
    transcription_factors = list(corr_data['TF'])

    target_genes = list(corr_data['TG'])
    list_tf_tg_edge = TF_high + TG + edge
    param = param_addr
    param_list = param.values.tolist()
    len_tf_tg = len(TF_high + TG)

    regulatory_dict_tf = {}
    regulatory_dict_gene = {}
    for tf, gene in zip(transcription_factors, target_genes):
        if tf not in regulatory_dict_tf:
            regulatory_dict_tf[tf] = []
        regulatory_dict_tf[tf].append(gene)
        if gene not in regulatory_dict_gene:
            regulatory_dict_gene[gene] = []
        regulatory_dict_gene[gene].append(tf)
    return list_tf_tg_edge, param_list, regulatory_dict_gene, regulatory_dict_tf,len_tf_tg


def get_param_numpy(list_all, param_list, TG, TF_list, len_tf_tg,ko_tf):

    # print(param_list)
    # print(list_all)
    # TF_list = TF_list[:1]
    TF_list =[]

    TF_list.append(ko_tf)
    TF_list = list(set(TF_list))
    print(TF_list)
    index_tg = list_all.index(TG)
    # tg_u = param_list[0][2 * index_tg]
    # tg_sigma = param_list[0][2 * index_tg + 1]
    TF_param_list = []
    # for TF in TF_list:
    TF = ko_tf
    index_tf = list_all.index(TF)
    sublist_values = [TF, TG]
    print(sublist_values)
    index_k = [index for index, sublist in enumerate(list_all) if sublist[:2] == sublist_values][0]
    weight_value = list_all[index_k][2]
    tf_u = param_list[0][2 * index_tf]
    tf_sigma = param_list[0][2 * index_tf + 1]
    k = param_list[0][index_k + int(len_tf_tg)]
    weight_value= weight_value
    delta_x = -5*k * weight_value * tf_u / (tf_sigma ** 2)
    # delta_x = k
    print(delta_x)
    return delta_x


time_start = time.time()

if __name__ == "__main__":


    args = sys.argv

    param1 = args[1]
    param2 = args[2]
    param3 = args[3]
    param4 = args[4]
    param5 = args[5]
    param6 = args[6]

    # net_addr='/home/cwt/mouse/bys/520/test65/test/628/pseudo_bulk/generated_grn/Mo_2_generated_grn.txt'
    # rna_addr='/home/cwt/mouse/bys/520/test65/test/628/pseudo_bulk/RNA/Mo_2.txt'
    # name_csv = 'Mo_2'
    net_addr = param1
    rna_addr = param2
    rna_addr_sam = param3
    name_csv = param4
    output_folder = param5
    edge_num = int(param6)
    # scaler = MinMaxScaler()  
    # # TF_high, TG, father_num, edge, TF_high_exp, TG_exp = data_pro(Network_file=net_addr,
    # #              RNA_file=rna_addr_sam,
    # #              edge_num=edge_num)
    # data = pd.read_csv('/home/share/taipan_sample/10_Primary_P-TGC_sample.csv')
    # data = pd.read_csv('/home/share/zaoxue_sample/Ery_0_sample.csv')
    data = pd.read_csv('/home/share/jingzi_sample/RS1o2_sample.csv')
    name_gene_tf = list(data.iloc[:, 0])
    TF_high, TG, father_num, edge_all, TF_high_exp, TG_exp,dataframe_data = data_pro1(Network_file=net_addr,
                                                                       RNA_file=rna_addr_sam,
                                                                       edge_num=edge_num, name=name_gene_tf)
    print(TF_high)
    eps = 0.01
    names = {}

    list_test =[]
    TF_high_duplicated_list = [item for item in TF_high for _ in range(2)]
    TG_high_duplicated_list = [item for item in TG for _ in range(2)]
    edge_0 = edge_all[0][0]
    # print(edge_all)
    # print(edge_0)
    string = ''.join(edge_0[:2])
    # string = ''.join(str(item) for item in edge_all[0][:2])

    list_test.extend(TF_high_duplicated_list)
    list_test.extend(TG_high_duplicated_list)
    list_test.extend(string)
    pd.DataFrame({'name':list_test}).to_csv('name.csv',index=False)

    arr = np.array(TF_high_exp)

    TF_high_mu = np.mean(arr, axis=0)
 
    TF_high_std = np.std(arr, axis=0)
    print(len(TF_high_mu))


    arr_tg = np.array(TG_exp)
  
    TG_mu = np.mean(arr_tg, axis=0)

    TG_std = np.std(arr_tg, axis=0)
    father_num = torch.IntTensor(father_num).to(device)
    # edge = edge_dataframe.iloc[:, :3].join(edge_dataframe.iloc[:, 3]).join(edge_dataframe.iloc[:, columns+3])
    # edge = edge.values.tolist()
    print('--------------------------')
    edge = edge_all[0]
    # print('--------------------------')

    model = Model().to(device)
    # optim = torch.optim.SGD(model.parameters(), lr=100)
  
    mu_learning_rate = 0.01
    sigma_learning_rate = 0.01
    k_learning_rate = 0.1

    mu_params = []
    sigma_params = []
    k_params = []

    for name, param in model.named_parameters():
        if 'mu' in name:
            mu_params.append(param)
        elif 'sigma' in name:
            sigma_params.append(param)
        elif 'k' in name:
            k_params.append(param)

    optimizer1 = torch.optim.SGD([
        {'params': mu_params, 'lr': mu_learning_rate},
        {'params': sigma_params, 'lr': sigma_learning_rate},
        {'params': k_params, 'lr': k_learning_rate}
    ])

    initial_params = {name: param.clone().detach() for name, param in model.named_parameters()}
    # print(initial_params)
    # print(TF_high_mu)
    # print(TF_high)
    for i in range(1, 50):
        out = 0
        for j in range(len(TF_high_exp)):
            edge = edge_all[j]
            TF_high_exp1 = TF_high_exp[j]
            TG_exp1 = TG_exp[j]
            out += model(TF_high_exp1, TG_exp1, edge)

            # print(out)
        # optim.zero_grad()
        # out.backward()
        # optim.step()
        loss = out
        optimizer1.zero_grad()
        loss.backward()

        # max_norm = 1.0 
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer1.step()
        for name, param in model.named_parameters():
            # if i <= 5:
            if 'mu' in name:
                with torch.no_grad():
                    param.clamp_(0.9 * initial_params[name], 0.9 * initial_params[name])
            if 'sigma' in name:
                with torch.no_grad():
                    param.clamp_(0.8 * initial_params[name], 1.2 * initial_params[name])
            # else:
            #     if 'k' in name: 
            #         with torch.no_grad():
            #             if param < 0:
            #                 param.clamp_(-10, -1) 
            #             else:
            #                 param.clamp_(1, 10)
                # if 'mu' in name:
                #     with torch.no_grad():
                #         param.clamp_(1 * initial_params[name], 1 * initial_params[name])
                # if 'sigma' in name:
                #     with torch.no_grad():
                #         param.clamp_(0.9 * initial_params[name], 1.1 * initial_params[name])
        print(out)


    # for name, param in model.named_parameters():
    #     # print(f'{name} parameters:')
    #     print(name)
    #     print(param)


    # params_dict = {name: param.detach().numpy().flatten() for name, param in model.named_parameters()}
    params_dict = {name: param.detach().cpu().numpy().flatten() for name, param in model.named_parameters()}

    params_df = pd.DataFrame(params_dict)

    print(params_df)
    params_df.to_csv(f'{output_folder}/params_df_{name_csv}.csv', index=False)

    addr = rna_addr
    # ko_tf = 'ETS1'
    addr_parm = params_df
    addr_peca = net_addr
    list_tf_tg_edge, param_list, regulatory_dict_gene, regulatory_dict_tf, len_tf_tg = get_data(dataframe=dataframe_data,
                                                                                                param_addr=addr_parm,
                                                                                                edge_num=edge_num,
                                                                                                TF_high=TF_high, TG=TG, edge=edge_all[0],name=name_gene_tf)
    # tf_list = list(regulatory_dict_tf.keys())
    # tf_list = ['Foxk1','Nrf1','Hbp1','Lhx2','Tfcp2','Trim28','Nfya','Pknox1','Homez','Stat4']
    tf_list = ["Hsfy2","Rfx4","Creb3l4","Etv2","Rfx2","Fev","Hoxa4","Hoxd8","Esx1","Zfp42","Rfx1","Mesp1","Rfx5","Crem","Alx1","Dlx5","Hsf2","Rfx3","Ctcfl","Mxd4","Mxd1","Tbp","Nfya","Mybl1","Mxi1","Egr3","Hsf1","Nkx2-1","Creb3","Smc3","Atf1","Tal1","Npas2","Nkx2-6","Foxj1","Rad21","Pbx4","Nanog","Zbtb3","Nr2c1","Atf2","Pknox2"]
    # target_gene_list = regulatory_dict_tf[tf_list[0]]
    # print(target_gene_list)
    # print(tf_list)
    df_k = pd.DataFrame(columns=['Gene'])
    RNA_exp = pd.read_table(addr, header=None)
    # print(data_c)
    RNA_exp = RNA_exp.iloc[:, :2]
    RNA_exp.columns = ['Gene', 'Exp']
    RNA_exp.set_index('Gene', drop=True, append=False, inplace=True)
    # RNA_exp = np.log1p(RNA_exp)
    # RNA_exp['Exp'] = scaler.fit_transform(RNA_exp[['Exp']])

    network = pd.read_table(addr_peca)
    network = network.iloc[:edge_num, :]
    gene_list = list(set(list(network['TF']) + list(network['TG'])))
    # print(len(gene_list))
    for i in range(len(tf_list)):
        ko_tf = tf_list[i]
        if ko_tf not in regulatory_dict_tf.keys():
            continue
        else:
            target_gene_list = regulatory_dict_tf[ko_tf]
            # # target_gene_list = regulatory_dict_tf[ko_tf]
            # # for i in target_gene_list:
            # #     if i in tf_list:
            # #         target_gene_list2 = regulatory_dict_tf[i]
            # #         target_gene_list.extend(target_gene_list2)
            # #         target_gene_list = list(set(target_gene_list))
      
            # propagation_rounds = 2
 
            # target_gene_list = propagate_genes(regulatory_dict_tf, ko_tf, propagation_rounds)

            # print(len(target_gene_list))
            data_c = RNA_exp.loc[target_gene_list]
            # init_exp_list = list(data_c[ko_tf])
            tg_pre_list = []
            #
            deltax_results = []
            for tg in target_gene_list:
                # print('------')
                # print(tg)
                init_TG_exp = data_c.loc[tg, 'Exp']

                deltax = get_param_numpy(list_tf_tg_edge, param_list, tg, regulatory_dict_gene[tg],
                                                                len_tf_tg,
                                                                ko_tf)
                deltax_results.append(deltax)
           
                # deltax_results = minmax_scale(deltax_results)

            res = pd.DataFrame({ko_tf: deltax_results}, index=target_gene_list)
            # print(res)
            # res.to_csv(f'{output_folder}/res_{ko_tf}.csv', index=True)
            res = res.reset_index()
            res.columns = ['Gene', 'Exp1']
            res.set_index('Gene', drop=True, append=False, inplace=True)

            data_gene = pd.DataFrame({'Gene': gene_list})

            merged_df = pd.merge(data_gene, res, on='Gene', how='left')
            # print(merged_df)
            # merged_df['Exp'] = merged_df['Exp1'].fillna(merged_df['Exp'])
            # result_df = merged_df[['Gene', 'Exp']]
            # print(result_df)
            result_df = merged_df
            result_df.columns = ['Gene', ko_tf]

            df_k = pd.merge(result_df, df_k, how='left', on='Gene')
            print(df_k)
            df_k = df_k.fillna(0)
    df_k.to_csv(f'{output_folder}/deltax_{name_csv}.csv', index=True)

time_end = time.time()
run_time = time_end - time_start
print(f"Run times：{run_time:.2f}秒")
