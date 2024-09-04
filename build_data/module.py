import os.path

import ot
import sklearn
import scipy.sparse as sp
import scanpy as sc
from scipy.sparse import bmat

from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from pathlib import Path

from .utils_func import *
from .his_feat import image_feature, image_crop
from .augment import augment_adata

def get_adata(
		platform,
		save_path,
		data_path,
		data_name,
		dataset_name,
		verbose=True,
):
	assert platform in ['10X', 'Slide_seqV2', 'Stereo_seq', 'STARmap', 'osmFISH', '3D']
	if platform == '10X':
		adata = read_10X_Visium(os.path.join(data_path, dataset_name, data_name))
	elif platform == 'Slide_seqV2':
		adata = read_SlideSeqV2(os.path.join(data_path, dataset_name, data_name))
	elif platform == 'Stereo_seq':
		adata = read_stereoSeq(os.path.join(data_path, dataset_name, data_name))
	elif platform == 'STARmap':
		adata = read_STARmap(os.path.join(data_path, dataset_name, data_name))
	elif platform == 'osmFISH':
		adata = read_osmFISH(os.path.join(data_path, dataset_name, data_name))
	elif platform == '3D':
		adata = read_3D(os.path.join(data_path, dataset_name, data_name))
	else:
		raise ValueError(
			f"""\
						 {platform!r} does not support.
								""")
	if verbose:
		save_data_path = Path(os.path.join(save_path, "Data", dataset_name, data_name))
		save_data_path.mkdir(parents=True, exist_ok=True)
		adata.write(os.path.join(save_data_path, f'{data_name}_raw.h5ad'), compression="gzip")
	return adata

def get_image_crop(
		adata,
		save_path,
		data_name,
		cnnType='ResNet50',
		pca_n_comps=50,
):
	save_path_image_crop = Path(os.path.join(save_path, 'Image_crop', data_name))
	save_path_image_crop.mkdir(parents=True, exist_ok=True)
	adata = image_crop(adata, save_path=save_path_image_crop)
	adata = image_feature(adata, pca_components=pca_n_comps, cnnType=cnnType).extract_image_feat()
	return adata

def preprocess_adata(adata, n_top_genes=3000, target_sum=1e4):
    """
    Preprocess an AnnData object with highly variable genes selection, total normalization, and log transformation.

    Parameters:
        adata (AnnData): The AnnData object to be preprocessed.
        n_top_genes (int): Number of top highly variable genes to select.
        target_sum (float): Target total sum after normalization.

    Returns:
        adata (AnnData): Preprocessed AnnData object.
    """
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_top_genes)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return adata

def spatial_prior_graph(feature_matrix, k_neighbors, v=1):
	"""
    Construct spatial prior graph for metric learning

    Parameters
    ------
    feature_matrix
        spatial corrdinate matrix
    k_neighbors
        number of neighbors to construct graph
    v
        scale factor in student's t kernel

    Returns
    ------
    scipy.sparse.csr_matrx
        spatial graph of sparse matrix format
    """
	dist = kneighbors_graph(feature_matrix, n_neighbors=k_neighbors, mode="distance").tocoo()
	dist.data = (1 + 1e-6 + dist.data ** 2 / v) ** (-(1 + v) / 2)
	dist.data = dist.data / (np.array(dist.sum(axis=1)).reshape(-1, 1).repeat(k_neighbors, axis=1).reshape(-1))
	spatial_graph = dist.tocsr()

	return spatial_graph

def construct_interaction(adata, n_neighbors=6):
	"""Constructing spot-to-spot interactive graph"""
	position = adata.obsm["image_feat_pca"]

	# calculate distance matrix
	distance_matrix = ot.dist(position, position, metric='euclidean')
	n_spot = distance_matrix.shape[0]

	adata.obsm['distance_matrix'] = distance_matrix

	# find k-nearest neighbors
	interaction = np.zeros([n_spot, n_spot])
	for i in range(n_spot):
		vec = distance_matrix[i, :]
		distance = vec.argsort()
		for t in range(1, n_neighbors + 1):
			y = distance[t]
			interaction[i, y] = 1

	adata.obsm['graph_neigh'] = interaction

	# transform adj to symmetrical adj
	adj = interaction
	adj = adj + adj.T
	adj = np.where(adj > 1, 1, adj)
	adj = adj + np.eye(adj.shape[0])

	# 获取非零元素的坐标
	nonzero_indices = np.argwhere(adj == 1)
	num_nonzero_elements = nonzero_indices.shape[0]
	result_A = np.zeros((2, num_nonzero_elements))

	# 将非零元素的坐标填充到 result_array 中
	result_A[0, :] = nonzero_indices[:, 0]
	result_A[1, :] = nonzero_indices[:, 1]

	return result_A

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
	"""\
	Construct the spatial neighbor networks.
	-------
	The spatial networks are saved in adata.uns['Spatial_Net']
	"""

	assert (model in ['Radius', 'KNN'])
	if verbose:
		print('------Calculating spatial graph...')
	coor = pd.DataFrame(adata.obsm['spatial'])
	coor.index = adata.obs.index
	coor.columns = ['imagerow', 'imagecol']

	if model == 'Radius':
		nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
		distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
		KNN_list = []
		for it in range(indices.shape[0]):
			KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

	if model == 'KNN':
		nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff + 1).fit(coor)
		distances, indices = nbrs.kneighbors(coor)
		KNN_list = []
		for it in range(indices.shape[0]):
			KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))

	KNN_df = pd.concat(KNN_list)
	KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

	Spatial_Net = KNN_df.copy()
	Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
	id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
	Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
	Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
	if verbose:
		print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
		print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

	adata.uns['Spatial_Net'] = Spatial_Net


def Cal_Spatial_Net_3D(adata, rad_cutoff_2D, rad_cutoff_Zaxis,
					   key_section='Section_id', section_order=None, verbose=True):
	"""\
    Construct the spatial neighbor networks.
    The 3D spatial networks are saved in adata.uns['Spatial_Net'].
    """
	adata.uns['Spatial_Net_2D'] = pd.DataFrame()
	adata.uns['Spatial_Net_Zaxis'] = pd.DataFrame()
	num_section = np.unique(adata.obs[key_section]).shape[0]
	if verbose:
		print('Radius used for 2D Spatial_Net:', rad_cutoff_2D)
		print('Radius used for Spatial_Net between sections:', rad_cutoff_Zaxis)
	for temp_section in np.unique(adata.obs[key_section]):
		if verbose:
			print('------Calculating 2D Spatial_Net of section ', temp_section)
		temp_adata = adata[adata.obs[key_section] == temp_section,]
		Cal_Spatial_Net(
			temp_adata, rad_cutoff=rad_cutoff_2D, verbose=False)
		temp_adata.uns['Spatial_Net']['SNN'] = temp_section
		if verbose:
			print('This graph contains %d edges, %d cells.' %
				  (temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
			print('%.4f neighbors per cell on average.' %
				  (temp_adata.uns['Spatial_Net'].shape[0] / temp_adata.n_obs))
		adata.uns['Spatial_Net_2D'] = pd.concat(
			[adata.uns['Spatial_Net_2D'], temp_adata.uns['Spatial_Net']])
	for it in range(num_section - 1):
		section_1 = section_order[it]
		section_2 = section_order[it + 1]
		if verbose:
			print('------Calculating Spatial_Net between adjacent section %s and %s.' %
				  (section_1, section_2))
		Z_Net_ID = section_1 + '-' + section_2
		temp_adata = adata[adata.obs[key_section].isin(
			[section_1, section_2]),]
		Cal_Spatial_Net(
			temp_adata, rad_cutoff=rad_cutoff_Zaxis, verbose=False)
		spot_section_trans = dict(
			zip(temp_adata.obs.index, temp_adata.obs[key_section]))
		temp_adata.uns['Spatial_Net']['Section_id_1'] = temp_adata.uns['Spatial_Net']['Cell1'].map(
			spot_section_trans)
		temp_adata.uns['Spatial_Net']['Section_id_2'] = temp_adata.uns['Spatial_Net']['Cell2'].map(
			spot_section_trans)
		used_edge = temp_adata.uns['Spatial_Net'].apply(
			lambda x: x['Section_id_1'] != x['Section_id_2'], axis=1)
		temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[used_edge,]
		temp_adata.uns['Spatial_Net'] = temp_adata.uns['Spatial_Net'].loc[:, ['Cell1', 'Cell2', 'Distance']]
		temp_adata.uns['Spatial_Net']['SNN'] = Z_Net_ID
		if verbose:
			print('This graph contains %d edges, %d cells.' %
				  (temp_adata.uns['Spatial_Net'].shape[0], temp_adata.n_obs))
			print('%.4f neighbors per cell on average.' %
				  (temp_adata.uns['Spatial_Net'].shape[0] / temp_adata.n_obs))
		adata.uns['Spatial_Net_Zaxis'] = pd.concat(
			[adata.uns['Spatial_Net_Zaxis'], temp_adata.uns['Spatial_Net']])
	adata.uns['Spatial_Net'] = pd.concat(
		[adata.uns['Spatial_Net_2D'], adata.uns['Spatial_Net_Zaxis']])
	if verbose:
		print('3D SNN contains %d edges, %d cells.' %
			  (adata.uns['Spatial_Net'].shape[0], adata.n_obs))
		print('%.4f neighbors per cell on average.' %
			  (adata.uns['Spatial_Net'].shape[0] / adata.n_obs))


def Stats_Spatial_Net(adata, data_path, data_name, dataset_name=None):
	import matplotlib.pyplot as plt
	Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
	Mean_edge = Num_edge / adata.shape[0]
	plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
	plot_df = plot_df / adata.shape[0]
	fig, ax = plt.subplots(figsize=[3, 2])
	plt.ylabel('Percentage')
	plt.xlabel('')
	plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)
	ax.bar(plot_df.index, plot_df)
	output_dir = Path(os.path.join(data_path, "Figure", dataset_name, data_name))
	output_dir.mkdir(parents=True, exist_ok=True)
	plt.savefig(os.path.join(output_dir, f'Neighbors.svg'), bbox_inches='tight', dpi=300)


def Transfer_pytorch_Data(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data


def get_augment(
		adata,
		adjacent_weight=0.3,
		neighbour_k=4,
		spatial_k=30,
		n_components=100,
		md_dist_type="cosine",
		gb_dist_type="correlation",
		use_morphological=True,
		use_data="raw",
		spatial_type="KDTree"
):
	adata = augment_adata(adata,
						  md_dist_type=md_dist_type,
						  gb_dist_type=gb_dist_type,
						  n_components=n_components,
						  use_morphological=use_morphological,
						  use_data=use_data,
						  neighbour_k=neighbour_k,
						  adjacent_weight=adjacent_weight,
						  spatial_k=spatial_k,
						  spatial_type=spatial_type
						  )
	print("Augment molecule expression is Done!")
	return adata

def data_process(adata, pca_n_comps=3000):
	adata.raw = adata
	adata.X = adata.obsm["augment_gene_data"].astype(np.float64)
	data = sc.pp.normalize_total(adata, target_sum=1, inplace=False)['X']
	data = sc.pp.log1p(data)
	data = sc.pp.scale(data)
	print("Please wait for a while, PCA operation is in progress")
	# data = sc.pp.pca(data, n_comps=pca_n_comps)
	from sklearn.decomposition import PCA
	pca = PCA(n_components=pca_n_comps)
	data = pca.fit_transform(data)
	return data




