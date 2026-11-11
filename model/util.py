import logging
import os
import torch
import random
import matplotlib
import scanpy as sc
import numpy as np
import scipy.sparse as sp

matplotlib.use('Agg')

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

def setup_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization

def get_logger():
    """Get logging."""
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    # model.main() can be called repeatedly in a seed/configuration sweep.
    # Reuse our stream handler so every call does not duplicate all later logs.
    handlers = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_crosp_stream_handler", False)
    ]
    if handlers:
        ch = handlers[0]
        ch.setFormatter(formatter)
    else:
        ch = logging.StreamHandler()
        ch._crosp_stream_handler = True
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def tranfer(A, size):
    A = A.astype('int64')
    adjacency_matrix = np.zeros((size, size))
    # 更新邻接矩阵
    for i in range(A.shape[1]):
        source_node = A[0, i]
        target_node = A[1, i]
        adjacency_matrix[source_node, target_node] = 1
    return adjacency_matrix

def build_adj(adj):
    # 将 NumPy ndarray 转换为稀疏矩阵
    sparse_matrix = sp.coo_matrix(adj)
    # 提取边的索引和权重
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col))
    edge_index = torch.from_numpy(indices).to(dtype=torch.float32, device=device)
    values = torch.tensor(sparse_matrix.data, dtype=torch.float32).to(device)
    return edge_index, values, adj.shape

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='CroSP', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    embedding = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    if embedding.ndim != 2:
        raise ValueError(
            f"mclust input must be a 2D matrix, got shape {embedding.shape}"
        )
    if not np.isfinite(embedding).all():
        raise ValueError("mclust input contains non-finite values")
    r_embedding = rpy2.robjects.numpy2ri.numpy2rpy(embedding)
    # With mclust 6.1.3, a matrix passed directly through rpy2 has no column
    # names. Mclust then deparses the full foreign-call expression and may
    # generate a dimnames vector longer than ncol(data). Explicit feature names
    # avoid that bridge-specific failure without changing the numeric matrix.
    feature_names = robjects.StrVector(
        [f"feature_{index + 1}" for index in range(embedding.shape[1])]
    )
    r_embedding = robjects.r['colnames<-'](r_embedding, feature_names)
    res = rmclust(
        r_embedding,
        num_cluster,
        modelNames,
        verbose=False,
    )
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def preprocess(adata):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)


def filter_with_overlap_gene(adata, adata_sc):
    # remove all-zero-valued genes
    # sc.pp.filter_genes(adata, min_cells=1)
    # sc.pp.filter_genes(adata_sc, min_cells=1)

    if 'highly_variable' not in adata.var.keys():
        raise ValueError("'highly_variable' are not existed in adata!")
    else:
        adata = adata[:, adata.var['highly_variable']]

    if 'highly_variable' not in adata_sc.var.keys():
        raise ValueError("'highly_variable' are not existed in adata_sc!")
    else:
        adata_sc = adata_sc[:, adata_sc.var['highly_variable']]

        # Refine `marker_genes` so that they are shared by both adatas
    genes = list(set(adata.var.index) & set(adata_sc.var.index))
    genes.sort()
    print('Number of overlap genes:', len(genes))

    adata.uns["overlap_genes"] = genes
    adata_sc.uns["overlap_genes"] = genes

    adata = adata[:, genes]
    adata_sc = adata_sc[:, genes]

    return adata, adata_sc










