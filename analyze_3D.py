import os
import time
import argparse
import warnings
import scanpy as sc

import build_data
import model

warnings.filterwarnings("ignore")


def parse(print_help=False):
    parser = argparse.ArgumentParser(description='Analyze')
    parser.add_argument('--data_type', type=str, default='Slide_seq', help='Type of data platform (e.g., "10X")')
    parser.add_argument('--task_type', type=str, default='3D', help='Type of task to perform (e.g., "Identify")')
    parser.add_argument('--data_name', type=str, default='3D_Hippo', help='Name of the section')
    parser.add_argument('--n_spa', type=int, default='3000', help='Spatial feature input dimension')
    parser.add_argument('--n_img', type=int, default='3000', help='Image feature input dimension')
    parser.add_argument('--print_num', type=int, default='10', help='gap of print evaluations')
    parser.add_argument('--build_data', type=bool, default=False, help='The need for data preparation')
    parser.add_argument('--emb_dim', nargs='+', default=[1024, 512, 256, 128, 128])
    parser.add_argument('--epoch', type=float, default=300, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=4e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay L2')
    parser.add_argument('--lambda1', type=float, default=0.005, help='parameter for Cross-view Contrastive_Loss')
    parser.add_argument('--lambda2', type=float, default=5.0, help='parameter for Reconstruction_Loss')
    parser.add_argument('--lambda3', type=float, default=5.0, help='parameter for Metric Loss')
    parser.add_argument("--d_prior", type=float, default=4.0, help="Number of prior embedding_dim")
    args = parser.parse_args()
    if print_help:
        parser.print_help()
    return args


if __name__ == "__main__":
    model.setup_seed(3)
    args = parse(print_help=False)
    save_path = "Results"
    data_path = 'data'
    task_type = args.task_type
    data_name = args.data_name
    start_time = time.time()
    adata_file_name = data_name + '_raw.h5ad'
    Data_path = os.path.join(save_path, 'Data', task_type, data_name)

    if not args.build_data:
        adata = sc.read(os.path.join(Data_path, adata_file_name))
    else:
        build_data.process_3D(platform=task_type, save_path=save_path, data_path=data_path,
                                      data_name=data_name, pca_n_comps=args.n_spa, n_top_genes=args.n_spa)
        adata = sc.read(os.path.join(Data_path, adata_file_name))
    z_f = model.main(save_path=save_path, config=args)
    adata.obsm['CroSP'] = z_f
    adata.write(os.path.join(Data_path, f'{data_name}_result_raw.h5ad'), compression="gzip")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Code execution time: {elapsed_time} seconds")







