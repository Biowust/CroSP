import scipy.io as scio
from scipy.sparse import bmat

from .module import *
from .utils_func import seed_everything

seed_everything(45)

def process_data_view1(adata, k_neighbors=6):
    # Convert adata.X to a CSR matrix
    adata.X = sp.csr_matrix(adata.X)

    # Extract highly variable genes if available
    if 'highly_variable' in adata.var.columns:
        adata_Vars = adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    # Check if Spatial_Net exists
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    # Transfer data to PyTorch tensor
    data = Transfer_pytorch_Data(adata_Vars)

    # Construct kNN graph with spatial_mtx
    spatial_mtx = adata.obsm["spatial"].copy()
    metric_graph = spatial_prior_graph(spatial_mtx, k_neighbors=k_neighbors)

    # Extract node features and edge indices
    X1 = data.x.numpy()
    A1 = data.edge_index.numpy()

    return X1, A1, metric_graph


def process_Visium(
    platform,
    save_path,
    data_path,
    data_name,
    dataset_name,
    task_type='Identify',
    pca_n_comps=3000,
    n_top_genes=3000,
    spatial_type="LinearRegress",
    use_morphological=True,
):
    adata = get_adata(platform=platform, save_path=save_path, data_path=data_path,
                      data_name=data_name, dataset_name=dataset_name)
    # Normalization
    adata = preprocess_adata(adata, n_top_genes=n_top_genes)
    #The rad_cutoff parameter is important and needs to be set correctly
    Cal_Spatial_Net(adata, rad_cutoff=150)
    Stats_Spatial_Net(adata, save_path, data_name, dataset_name)

    X1, A1, metric_graph = process_data_view1(adata)

    adata = get_image_crop(adata, save_path=save_path, data_name=data_name)
    adata_image = get_augment(adata, spatial_type=spatial_type, use_morphological=use_morphological)
    image_graph = construct_interaction(adata, n_neighbors=6)
    A2 = image_graph

    data_image = data_process(adata_image, pca_n_comps=pca_n_comps)
    X2 = data_image


    data_save_path = os.path.join(save_path, "Data", dataset_name, data_name)
    os.makedirs(data_save_path, exist_ok=True)
    scio.savemat(os.path.join(data_save_path, f'data_{data_name}.mat'),
                 {"X1": X1, "A1": A1, "X2": X2, "A2": A2, "metric_graph": metric_graph})

    print('process_data_and_save complete！')


def process_STARmap(
    platform,
    save_path,
    data_path,
    data_name,
    pca_n_comps=3000,
    n_top_genes=3000,
    spatial_type="BallTree",
    use_morphological=False,
):
    adata = get_adata(platform=platform, save_path=save_path, data_path=data_path,
                      data_name=data_name, dataset_name=platform)
    # Normalization
    adata = preprocess_adata(adata, n_top_genes=n_top_genes)
    Cal_Spatial_Net(adata, rad_cutoff=400)
    Stats_Spatial_Net(adata, save_path, data_name, platform)

    X1, A1, metric_graph = process_data_view1(adata)

    adata_image = get_augment(adata, spatial_type=spatial_type, use_morphological=use_morphological)
    A2 = A1
    data_image = data_process(adata_image, pca_n_comps=pca_n_comps)
    X2 = data_image

    data_save_path = os.path.join(save_path, "Data", platform, data_name)
    os.makedirs(data_save_path, exist_ok=True)
    scio.savemat(os.path.join(data_save_path, f'data_{data_name}.mat'),
                 {"X1": X1, "A1": A1, "X2": X2, "A2": A2, "metric_graph": metric_graph})

    print('process_data_and_save complete！')


def process_osmFISH(
    platform,
    save_path,
    data_path,
    data_name,
    pca_n_comps=3000,
    n_top_genes=3000,
    spatial_type="BallTree",
    use_morphological=False,
):
    adata = get_adata(platform=platform, save_path=save_path, data_path=data_path,
                      data_name=data_name, dataset_name=platform)
    Cal_Spatial_Net(adata, rad_cutoff=500)
    Stats_Spatial_Net(adata, save_path, data_name, platform)

    X1, A1, metric_graph = process_data_view1(adata)

    adata_image = get_augment(adata, spatial_type=spatial_type, use_morphological=use_morphological)
    A2 = A1
    data_image = data_process(adata_image, pca_n_comps=pca_n_comps)
    X2 = data_image

    data_save_path = os.path.join(save_path, "Data", platform, data_name)
    os.makedirs(data_save_path, exist_ok=True)
    scio.savemat(os.path.join(data_save_path, f'data_{data_name}.mat'),
                 {"X1": X1, "A1": A1, "X2": X2, "A2": A2, "metric_graph": metric_graph})

    print('process_data_and_save complete！')


def process_Slide_seqV2(
    platform,
    save_path,
    data_path,
    data_name,
    pca_n_comps=3000,
    n_top_genes=3000,
    spatial_type="BallTree",
    use_morphological=False,
):
    adata = get_adata(platform=platform, save_path=save_path, data_path=data_path,
                      data_name=data_name, dataset_name=platform)
    adata = preprocess_adata(adata, n_top_genes=n_top_genes)
    Cal_Spatial_Net(adata, rad_cutoff=50)
    Stats_Spatial_Net(adata, save_path, data_name, platform)

    X1, A1, metric_graph = process_data_view1(adata)

    adata_image = get_augment(adata, spatial_type=spatial_type, use_morphological=use_morphological)
    A2 = A1
    data_image = data_process(adata_image, pca_n_comps=pca_n_comps)
    X2 = data_image

    data_save_path = os.path.join(save_path, "Data", platform, data_name)
    os.makedirs(data_save_path, exist_ok=True)
    scio.savemat(os.path.join(data_save_path, f'data_{data_name}.mat'),
                 {"X1": X1, "A1": A1, "X2": X2, "A2": A2, "metric_graph": metric_graph})

    print('process_data_and_save complete！')


def process_Stereo_seq(
    platform,
    save_path,
    data_path,
    data_name,
    pca_n_comps=3000,
    n_top_genes=3000,
    spatial_type="BallTree",
    use_morphological=False,
):
    adata = get_adata(platform=platform, save_path=save_path, data_path=data_path,
                      data_name=data_name, dataset_name=platform)
    adata = preprocess_adata(adata, n_top_genes=n_top_genes)
    Cal_Spatial_Net(adata, k_cutoff=6, model='KNN')
    Stats_Spatial_Net(adata, save_path, data_name, platform)

    X1, A1, metric_graph = process_data_view1(adata)

    adata_image = get_augment(adata, spatial_type=spatial_type, use_morphological=use_morphological)
    A2 = A1
    data_image = data_process(adata_image, pca_n_comps=pca_n_comps)
    X2 = data_image

    data_save_path = os.path.join(save_path, "Data", platform, data_name)
    os.makedirs(data_save_path, exist_ok=True)
    scio.savemat(os.path.join(data_save_path, f'data_{data_name}.mat'),
                 {"X1": X1, "A1": A1, "X2": X2, "A2": A2, "metric_graph": metric_graph})

    print('process_data_and_save complete！')


def process_3D(
    platform,
    save_path,
    data_path,
    data_name,
    pca_n_comps=3000,
    n_top_genes=3000,
    spatial_type="BallTree",
    use_morphological=False,
):
    adata = get_adata(platform=platform, save_path=save_path, data_path=data_path,
                      data_name=data_name, dataset_name=platform)
    # Normalization
    adata = preprocess_adata(adata, n_top_genes=n_top_genes)
    section_order = ['Puck_180531_13', 'Puck_180531_16', 'Puck_180531_17',
                     'Puck_180531_18', 'Puck_180531_19', 'Puck_180531_22',
                     'Puck_180531_23']
    Cal_Spatial_Net_3D(adata, rad_cutoff_2D=50, rad_cutoff_Zaxis=50,
                       key_section='Section_id', section_order=section_order, verbose=True)
    #Running CroSP with 2D spatial networks (for comparison)
    adata_2D = adata.copy()
    adata_2D.uns['Spatial_Net'] = adata.uns['Spatial_Net_2D'].copy()
    adata = adata_2D

    X1, A1, metric_graph = process_data_view1(adata)

    adata_image = get_augment(adata, spatial_type=spatial_type, use_morphological=use_morphological)
    A2 = A1
    data_image = data_process(adata_image, pca_n_comps=pca_n_comps)
    X2 = data_image

    data_save_path = os.path.join(save_path, "Data", platform, data_name)
    os.makedirs(data_save_path, exist_ok=True)
    scio.savemat(os.path.join(data_save_path, f'data_{data_name}.mat'),
                 {"X1": X1, "A1": A1, "X2": X2, "A2": A2, "metric_graph": metric_graph})
    print('process_data_and_save complete！')





