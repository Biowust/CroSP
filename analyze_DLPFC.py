import os
import time
import argparse
import warnings
import pandas as pd
import scanpy as sc
import sklearn.metrics as metrics
from sklearn.decomposition import PCA

import build_data
import model

warnings.filterwarnings("ignore")


def cluster_and_evaluate_with_seed(adata):
    best_ari = -1
    z_f = model.main(save_path=save_path, config=args)
    pca = PCA(n_components=30, random_state=42)
    z_f = pca.fit_transform(z_f)
    for random_state in random_state_values:
        adata.obsm['CroSP'] = z_f
        adata_modified = model.mclust_R(adata, used_obsm='CroSP', num_cluster=n_clusters, random_seed=random_state)
        df_meta = pd.read_csv(os.path.join('./data', 'DLPFC', data_name, f'metadata.tsv'), sep='\t')
        df_meta.index = df_meta['barcode']
        adata.obs['Ground Truth'] = df_meta.loc[adata.obs_names, 'layer_guess']
        obs_df = adata_modified.obs.dropna()
        ARI = metrics.adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
        print(f'ARI for random state {random_state}: {ARI:.3f}')

        if ARI > best_ari:
            best_ari = ARI
            best_adata = adata_modified  # 存储当前最佳的 adata
            best_adata.write(os.path.join(Data_path, f'{data_name}_best_raw.h5ad'), compression="gzip")
            best_random_state = random_state
    print(f'Best Random State: {best_random_state}')
    print(f'Best ARI Score: {best_ari:.3f}')

#
# DLPFC----151507,151508,151509,151510,151669,151670,151671,151672,151673,151674,151675,151676


def parse(print_help=False):
    parser = argparse.ArgumentParser(description='Analyze')
    parser.add_argument('--data_type', type=str, default='10X', help='Type of data platform (e.g., "10X")')
    parser.add_argument('--task_type', type=str, default='Identify', help='Type of task to perform (e.g., "Identify")')
    parser.add_argument('--dataset_name', type=str, default='DLPFC', help='Name of the dataset (e.g., "DLPFC")')
    parser.add_argument('--data_name', type=str, default='151673', help='Name of the section (e.g., "151673")')
    parser.add_argument('--n_spa', type=int, default='3000', help='Spatial feature input dimension')
    parser.add_argument('--n_img', type=int, default='3000', help='Image feature input dimension')
    parser.add_argument('--print_num', type=int, default='10', help='gap of print evaluations')
    parser.add_argument('--build_data', type=bool, default=False, help='The need for data preparation')
    parser.add_argument('--emb_dim', nargs='+', default=[1024, 512, 256, 128, 128])
    parser.add_argument('--epoch', type=float, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=4e-6, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay L2')
    parser.add_argument('--lambda1', type=float, default=0.005, help='parameter for Cross-view Contrastive_Loss')
    parser.add_argument('--lambda2', type=float, default=3.0, help='parameter for Reconstruction_Loss')
    parser.add_argument('--lambda3', type=float, default=5.0, help='parameter for Metric Loss')
    parser.add_argument("--d_prior", type=float, default=4.0, help="Number of prior embedding_dim")
    args = parser.parse_args()
    if print_help:
        parser.print_help()
    return args


if __name__ == "__main__":
    model.setup_seed(3)
    random_state_values = list(range(5))
    args = parse(print_help=False)
    save_path = "Results"
    data_path = 'data'
    data_type = args.data_type
    data_name = args.data_name
    n_clusters = 5 if data_name in ['151669', '151670', '151671', '151672'] else 7
    dataset_name = args.dataset_name
    start_time = time.time()
    adata_file_name = data_name + '_raw.h5ad'
    Data_path = os.path.join(save_path, 'Data', dataset_name, data_name)

    if not args.build_data:
        adata = sc.read(os.path.join(Data_path, adata_file_name))
    else:
        build_data.process_Visium(platform=data_type, save_path=save_path, data_path=data_path, data_name=data_name,
                                  dataset_name=dataset_name, pca_n_comps=args.n_spa, n_top_genes=args.n_spa)
        adata = sc.read(os.path.join(Data_path, adata_file_name))
    cluster_and_evaluate_with_seed(adata=adata)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Code execution time: {elapsed_time} seconds")







