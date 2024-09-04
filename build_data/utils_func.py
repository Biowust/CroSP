import os
import scanpy as sc
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from anndata import AnnData
from ._compat import Literal

_QUALITY = Literal["fulres", "hires", "lowres"]
_background = ["black", "white"]
def seed_everything(seed):
    torch.backends.cudnn.benchmark = False  # Close optimization
    torch.backends.cudnn.deterministic = True  # Close optimization
    torch.backends.cudnn.enabled = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.use_deterministic_algorithms(True)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def read_10X_Visium(path, 
                    genome=None,
                    count_file='filtered_feature_bc_matrix.h5',
                    library_id=None,
                    load_images=True,
                    quality='hires',
                    image_path=None):
    adata = sc.read_visium(path, 
                        genome=genome,
                        count_file=count_file,
                        library_id=library_id,
                        load_images=load_images,
                        )
    adata.var_names_make_unique()
    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]
    if quality == "fulres":
        image_coor = adata.obsm["spatial"]
        img = plt.imread(image_path, 0)
        adata.uns["spatial"][library_id]["images"]["fulres"] = img
    else:
        scale = adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_" + quality + "_scalef"]
        image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    adata.uns["spatial"][library_id]["use_quality"] = quality
    return adata


def read_SlideSeqV2(path,
                 library_id = None,
                 scale = None,
                 quality = "hires",
                 spot_diameter_fullres= 50,
                 background_color = "white",):
    parts = path.split('/')
    data_name = parts[-1]
    meta = pd.read_csv(os.path.join(path, "bead_locations.csv"), index_col=0)
    count = pd.read_csv(os.path.join(path, "digital_expression.txt"), sep='\t', index_col=0)

    adata = AnnData(count.T)

    if scale is None:
        max_coor = np.max(meta[["xcoord", "ycoord"]].values)
        scale = 2000 / max_coor

    adata.obs["imagecol"] = meta["xcoord"].values * scale
    adata.obs["imagerow"] = meta["ycoord"].values * scale

    # Create image
    max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
    max_size = int(max_size + 0.1 * max_size)

    if background_color == "black":
        image = Image.new("RGBA", (max_size, max_size), (0, 0, 0, 0))
    else:
        image = Image.new("RGBA", (max_size, max_size), (255, 255, 255, 255))
    imgarr = np.array(image)

    if library_id is None:
        library_id = "Slide-seqV2"

    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"][quality] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = quality
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"][
        "tissue_" + quality + "_scalef"] = scale

    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"
    ] = spot_diameter_fullres
    adata.obsm["spatial"] = meta[["xcoord", "ycoord"]].values

    if data_name == 'Puck_200127_15':
        used_barcode = pd.read_csv(os.path.join(path, 'used_barcodes.txt'), sep='\t', header=None)
        used_barcode = used_barcode[0]
        adata = adata[used_barcode,]
    else:
        pass
    sc.pp.filter_genes(adata, min_cells=50)

    return adata


def read_stereoSeq(path,
                library_id=None,
                scale=None,
                quality="hires",
                spot_diameter_fullres=1,
                background_color="white",
                ):
    parts = path.split('/')
    data_name = parts[-1]
    if data_name != 'Mouse_embryo':
        counts_file = os.path.join(path, 'counts.tsv')
        coor_file = os.path.join(path, 'position.tsv')
        counts = pd.read_csv(counts_file, sep='\t', index_col=0)
        coor_df = pd.read_csv(coor_file, sep='\t')
        counts.columns = ['Spot_' + str(x) for x in counts.columns]
        coor_df.index = coor_df['label'].map(lambda x: 'Spot_' + str(x))
        coor_df = coor_df.loc[:, ['x', 'y']]
        adata = sc.AnnData(counts.T)
        adata.var_names_make_unique()
        coor_df = coor_df.loc[adata.obs_names, ['y', 'x']]
        adata.obsm["spatial"] = coor_df.to_numpy()
        used_barcode = pd.read_csv(os.path.join(path, 'used_barcodes.txt'), sep='\t', header=None)
        used_barcode = used_barcode[0]
        adata = adata[used_barcode,]
        sc.pp.filter_genes(adata, min_cells=50)

        if scale == None:
            max_coor = np.max(adata.obsm["spatial"])
            scale = 20 / max_coor

        adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
        adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale

        # Create image
        max_size = np.max([adata.obs["imagecol"].max(), adata.obs["imagerow"].max()])
        max_size = int(max_size + 0.1 * max_size)
        if background_color == "black":
            image = Image.new("RGB", (max_size, max_size), (0, 0, 0, 0))
        else:
            image = Image.new("RGB", (max_size, max_size), (255, 255, 255, 255))
        imgarr = np.array(image)

        if library_id is None:
            library_id = "StereoSeq"

        adata.uns["spatial"] = {}
        adata.uns["spatial"][library_id] = {}
        adata.uns["spatial"][library_id]["images"] = {}
        adata.uns["spatial"][library_id]["images"][quality] = imgarr
        adata.uns["spatial"][library_id]["use_quality"] = quality
        adata.uns["spatial"][library_id]["scalefactors"] = {}
        adata.uns["spatial"][library_id]["scalefactors"]["tissue_" + quality + "_scalef"] = scale
        adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] = spot_diameter_fullres
    else:
        adata = sc.read(os.path.join(path, 'E9.5_E1S1.MOSTA.h5ad'))
    return adata


def read_STARmap(path):
    adata = sc.read(os.path.join(path, 'STARmap_20180505_BY3_1k.h5ad'))
    return adata


def read_osmFISH(path):
    adata = sc.read(os.path.join(path, 'osmFISH_MSC.h5ad'))
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=None)
    sc.pp.log1p(adata)
    return adata


def read_3D(path):
    data = pd.read_csv(os.path.join(path, '3D_Hippo_expression.txt'), sep='\t', index_col=0)
    Aligned_coor = pd.read_csv(os.path.join(path, 'ICP_Align_Coor.txt'), sep='\t', index_col=0)
    adata = sc.AnnData(data)
    adata.obs['X'] = Aligned_coor.loc[adata.obs_names, 'X']
    adata.obs['Y'] = Aligned_coor.loc[adata.obs_names, 'Y']
    adata.obs['Z'] = Aligned_coor.loc[adata.obs_names, 'Z']
    adata.obs['Section_id'] = Aligned_coor.loc[adata.obs_names, 'Section']
    adata.obsm['spatial'] = adata.obs.loc[:, ['X', 'Y']].values
    return adata





