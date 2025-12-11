from .base_model.ncf import NCF


def build_model(name, num_nodes: int, **kwargs):
    if name.lower() == "ncf":
        model = NCF(num_nodes)
    else:
        raise NotImplementedError
    return model
