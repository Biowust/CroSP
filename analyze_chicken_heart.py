#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache" / "xdg"))
os.environ.setdefault("TORCH_HOME", "/data/lxj/.cache/torch")
os.environ.setdefault(
    "R_LIBS_USER", "/home/lxj/R/x86_64-pc-linux-gnu-library/4.6"
)

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image
import scanpy as sc
import scipy.io as scio
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import NearestNeighbors

import model
from build_data.demo_t import process_data_view1
from build_data.his_feat import image_feature
from build_data.module import (
    Cal_Spatial_Net,
    Stats_Spatial_Net,
    construct_interaction,
    data_process,
    get_augment,
    preprocess_adata,
)
from plot_chicken_heart_ground_truth import (
    HORIZONTALLY_FLIPPED_STAGES,
    REGION_COLORS,
    REGION_ORDER,
    VERTICALLY_FLIPPED_STAGES,
)
from chicken_heart_plot_style import apply_touching_marker_area


SAMPLES = ("D7", "D10", "D14")
SAMPLE_PREFIX = {"D7": "D7-B1", "D10": "D10-C1", "D14": "D14-D1"}
# Prespecified benchmark K.  It is never used to train the CroSP embedding.
N_CLUSTERS = {"D7": 7, "D10": 6, "D14": 6}
# Predicted cluster identifiers are arbitrary and are not matched to anatomical
# classes.  For visual consistency, each sample nevertheless uses exactly the
# same non-black palette as Ground Truth (assigned to numeric cluster ids in
# sorted order, without implying a semantic match).
PREDICTION_COLORS = tuple(REGION_COLORS[region] for region in REGION_ORDER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("preflight", "prepare", "train", "evaluate", "aggregate", "all"),
        default="preflight",
    )
    parser.add_argument("--samples", nargs="+", choices=SAMPLES, default=list(SAMPLES))
    parser.add_argument("--model-seeds", nargs="+", type=int, default=[3])
    parser.add_argument("--mclust-seed", type=int, default=0)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "new_data" / "chicken_heart"
    )
    parser.add_argument("--save-path", type=Path, default=ROOT / "Results")
    parser.add_argument("--dataset-name", default="ChickenHeart")
    parser.add_argument(
        "--run-name",
        default="canonical",
        help="Short identifier used to keep configurations in separate folders.",
    )
    parser.add_argument(
        "--development-sample",
        choices=SAMPLES,
        default=None,
        help="Record the only sample whose labels were used to choose this configuration.",
    )
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")

    # View construction.  D7 has 535 spots, so 512 components are the largest
    # convenient common feature dimension across all three stages.
    parser.add_argument("--n-features", type=int, default=512)
    parser.add_argument("--image-pcs", type=int, default=50)
    parser.add_argument("--patch-size", type=int, default=50)
    parser.add_argument("--metric-neighbors", type=int, default=6)
    parser.add_argument("--image-neighbors", type=int, default=6)
    parser.add_argument(
        "--rad-cutoff",
        type=float,
        default=0.0,
        help="Full-resolution spatial radius; 0 infers the first Visium shell.",
    )

    # Historical CroSP Visium defaults; apply unchanged to every stage.
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=4e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lambda1", type=float, default=0.005)
    parser.add_argument("--lambda2", type=float, default=3.0)
    parser.add_argument("--lambda3", type=float, default=5.0)
    parser.add_argument("--d-prior", type=float, default=4.0)
    parser.add_argument("--gamma-init", type=float, default=0.6)
    parser.add_argument(
        "--emb-dim", nargs="+", type=int, default=[1024, 512, 256, 128, 128]
    )
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--print-num", type=int, default=10)
    args = parser.parse_args()

    if len(set(args.samples)) != len(args.samples):
        parser.error("duplicate samples are not allowed")
    if len(set(args.model_seeds)) != len(args.model_seeds):
        parser.error("duplicate model seeds are not allowed")
    if args.n_features >= 535:
        parser.error("n-features must be < 535 so D7 PCA is well-defined")
    if args.patch_size < 1 or args.patch_size > 80:
        parser.error("patch-size must lie in 1..80")
    if re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name) is None:
        parser.error("run-name may contain only letters, numbers, dot, dash, underscore")
    return args


def sample_paths(args: argparse.Namespace, sample: str) -> dict[str, Path]:
    raw_sample = args.data_root / "raw_data" / "chick_heart" / sample
    result_dir = args.save_path / "Data" / args.dataset_name / sample
    return {
        "raw_sample": raw_sample,
        "matrix_h5": raw_sample / "filtered_feature_bc_matrix.h5",
        "positions": raw_sample / "spatial" / "tissue_positions_list.csv",
        "lowres": raw_sample / "spatial" / "tissue_lowres_image.png",
        "scalefactors": raw_sample / "spatial" / "scalefactors_json.json",
        "benchmark_h5ad": args.data_root
        / "SpatialModal"
        / "chick_heart"
        / f"{sample}.h5ad",
        "metadata": args.data_root
        / "raw_data"
        / "chick_heart"
        / "spatialRNAseq_metadata.csv",
        "result_dir": result_dir,
        "raw_h5ad": result_dir / f"{sample}_raw_label_free.h5ad",
        "prepared_h5ad": result_dir / f"{sample}_prepared_label_free.h5ad",
        "matrix": result_dir / f"data_{sample}.mat",
        "prepare_summary": result_dir / "preparation_summary.json",
        "image_features": result_dir / "image_features.npz",
        "crop_dir": args.save_path / "Image_crop" / f"{args.dataset_name}_{sample}",
    }


def run_paths(args: argparse.Namespace, sample: str, seed: int) -> dict[str, Path]:
    base = sample_paths(args, sample)
    run_dir = base["result_dir"] / "runs" / args.run_name / f"seed_{seed}"
    figure_dir = (
        args.save_path
        / "Figure"
        / args.dataset_name
        / sample
        / args.run_name
        / f"seed_{seed}"
    )
    return {
        **base,
        "run_dir": run_dir,
        "figure_dir": figure_dir,
        "model_output": run_dir / "model_output.mat",
        "training_summary": run_dir / "training_summary.json",
        "metrics": run_dir / "metrics.csv",
        "assignments": run_dir / "assignments.csv",
        "final_h5ad": run_dir / f"{sample}_CroSP_seed{seed}.h5ad",
        "config": run_dir / "run_config.json",
        "figure_png": figure_dir / f"{sample}_CroSP_seed{seed}.png",
        "figure_svg": figure_dir / f"{sample}_CroSP_seed{seed}.svg",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_region(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.replace("\\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def infer_radius(coordinates: np.ndarray) -> tuple[float, float]:
    distances, _ = NearestNeighbors(n_neighbors=2).fit(coordinates).kneighbors()
    nearest = float(np.median(distances[:, 1]))
    return float(np.ceil(nearest * 1.02)), nearest


def read_chicken_visium(sample_dir: Path) -> ad.AnnData:
    """Read the uploaded Visium layout without requiring a missing hires PNG."""

    matrix = sample_dir / "filtered_feature_bc_matrix.h5"
    spatial = sample_dir / "spatial"
    positions_path = spatial / "tissue_positions_list.csv"
    scalefactors_path = spatial / "scalefactors_json.json"
    image_path = spatial / "tissue_lowres_image.png"
    adata = sc.read_10x_h5(matrix)
    adata.var_names_make_unique()
    positions = pd.read_csv(
        positions_path,
        header=None,
        names=(
            "barcode",
            "in_tissue",
            "array_row",
            "array_col",
            "pxl_row_in_fullres",
            "pxl_col_in_fullres",
        ),
        index_col="barcode",
    )
    missing = adata.obs_names.difference(positions.index)
    if len(missing):
        raise ValueError(f"{sample_dir.name}: {len(missing)} matrix barcodes lack coordinates")
    aligned = positions.loc[adata.obs_names]
    for column in ("in_tissue", "array_row", "array_col"):
        adata.obs[column] = aligned[column].to_numpy()
    adata.obsm["spatial"] = aligned[
        ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    ].to_numpy(float)
    with scalefactors_path.open() as handle:
        scalefactors = json.load(handle)
    scale = float(scalefactors["tissue_lowres_scalef"])
    adata.obs["imagecol"] = adata.obsm["spatial"][:, 0] * scale
    adata.obs["imagerow"] = adata.obsm["spatial"][:, 1] * scale
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    library_id = SAMPLE_PREFIX[sample_dir.name]
    adata.uns["spatial"] = {
        library_id: {
            "images": {"lowres": image},
            "scalefactors": scalefactors,
            "use_quality": "lowres",
            "metadata": {"source_image": str(image_path)},
        }
    }
    return adata


def load_benchmark_layout(path: Path) -> tuple[pd.Index, np.ndarray]:
    benchmark = ad.read_h5ad(path, backed="r")
    try:
        if "patches" not in benchmark.obsm:
            raise KeyError(f"{path}: missing obsm['patches']")
        barcodes = benchmark.obs_names.copy()
        patches = np.asarray(benchmark.obsm["patches"])
    finally:
        benchmark.file.close()
    if patches.shape != (len(barcodes), 80, 80, 3):
        raise ValueError(f"Unexpected patch shape in {path}: {patches.shape}")
    if not np.isfinite(patches).all():
        raise ValueError(f"Non-finite patch values in {path}")
    return barcodes, patches


def aligned_truth(metadata_path: Path, sample: str, barcodes: pd.Index) -> pd.Series:
    metadata = pd.read_csv(metadata_path, index_col=0)
    ids = pd.Index([f"{SAMPLE_PREFIX[sample]}_{barcode}" for barcode in barcodes])
    missing = ids.difference(metadata.index)
    if len(missing):
        raise ValueError(f"{sample}: {len(missing)} spots lack metadata")
    truth = clean_region(metadata.loc[ids, "region"])
    truth.index = barcodes
    if truth.isna().any():
        raise ValueError(f"{sample}: missing region labels")
    return truth


def validate_mclust() -> str:
    import rpy2.robjects as robjects

    robjects.r.library("mclust")
    return str(robjects.r("as.character(packageVersion('mclust'))")[0])


def preflight_sample(args: argparse.Namespace, sample: str) -> dict[str, object]:
    paths = sample_paths(args, sample)
    required = (
        "matrix_h5",
        "positions",
        "lowres",
        "scalefactors",
        "benchmark_h5ad",
        "metadata",
    )
    missing = [str(paths[key]) for key in required if not paths[key].exists()]
    if missing:
        raise FileNotFoundError("Missing chicken-heart inputs: " + ", ".join(missing))

    benchmark = ad.read_h5ad(paths["benchmark_h5ad"], backed="r")
    try:
        barcodes = benchmark.obs_names.copy()
        patch_shape = tuple(benchmark.obsm["patches"].shape)
        processed_genes = int(benchmark.n_vars)
    finally:
        benchmark.file.close()

    raw = read_chicken_visium(paths["raw_sample"])
    absent = barcodes.difference(raw.obs_names)
    if len(absent):
        raise ValueError(f"{sample}: {len(absent)} benchmark spots lack raw counts")
    selected = raw[barcodes].copy()
    truth = aligned_truth(paths["metadata"], sample, barcodes)
    if truth.nunique() != N_CLUSTERS[sample]:
        raise ValueError(
            f"{sample}: expected {N_CLUSTERS[sample]} regions, found {truth.nunique()}"
        )
    radius, nearest = infer_radius(selected.obsm["spatial"])
    record = {
        "sample": sample,
        "raw_spots": int(raw.n_obs),
        "benchmark_spots": int(selected.n_obs),
        "raw_genes": int(selected.n_vars),
        "processed_h5ad_genes": processed_genes,
        "regions": int(truth.nunique()),
        "patch_shape": list(patch_shape),
        "median_nearest_distance": nearest,
        "inferred_radius": radius,
        "all_barcodes_aligned": len(absent) == 0,
        "ground_truth_complete": bool(truth.notna().all()),
    }
    print("[PREFLIGHT] " + json.dumps(record))
    return record


def preflight(args: argparse.Namespace) -> None:
    records = [preflight_sample(args, sample) for sample in args.samples]
    checkpoint = (
        Path(os.environ["TORCH_HOME"])
        / "hub"
        / "checkpoints"
        / "resnet50-0676ba61.pth"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Cached ResNet50 weights not found: {checkpoint}")
    software = {
        "torch": torch.__version__,
        "scanpy": sc.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mclust": validate_mclust(),
        "resnet50_weights": str(checkpoint),
        "resnet50_sha256": sha256(checkpoint),
    }
    print("[SOFTWARE] " + json.dumps(software))
    output = args.save_path / "Data" / args.dataset_name / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"samples": records, "software": software}, indent=2))


def write_center_crops(
    barcodes: pd.Index,
    patches: np.ndarray,
    crop_dir: Path,
    patch_size: int,
) -> list[str]:
    crop_dir.mkdir(parents=True, exist_ok=True)
    source_size = patches.shape[1]
    start = (source_size - patch_size) // 2
    stop = start + patch_size
    outputs: list[str] = []
    for barcode, patch in zip(barcodes, patches):
        array = np.clip(patch, 0, 255).astype(np.uint8)
        array = array[start:stop, start:stop]
        output = crop_dir / f"{barcode}.png"
        Image.fromarray(array, mode="RGB").save(output)
        outputs.append(str(output))
    return outputs


def prepare_sample(args: argparse.Namespace, sample: str) -> None:
    paths = sample_paths(args, sample)
    if (
        paths["matrix"].exists()
        and paths["raw_h5ad"].exists()
        and paths["prepared_h5ad"].exists()
        and not args.force_prepare
    ):
        print(f"[SKIP] {sample} prepared input exists: {paths['matrix']}")
        return

    started = time.perf_counter()
    paths["result_dir"].mkdir(parents=True, exist_ok=True)
    barcodes, patches = load_benchmark_layout(paths["benchmark_h5ad"])
    adata = read_chicken_visium(paths["raw_sample"])
    absent = barcodes.difference(adata.obs_names)
    if len(absent):
        raise ValueError(f"{sample}: {len(absent)} benchmark spots lack raw counts")
    adata = adata[barcodes].copy()
    adata.var_names_make_unique()
    raw_totals = np.asarray(adata.X.sum(axis=1)).ravel()
    if np.any(raw_totals <= 0):
        raise ValueError(f"{sample}: benchmark subset contains zero-count spots")
    adata.obs["raw_total_counts"] = raw_totals
    adata.obs["raw_n_genes_by_counts"] = np.asarray(
        (adata.X > 0).sum(axis=1)
    ).ravel()

    # This checkpoint intentionally contains no anatomical labels.
    adata.write_h5ad(paths["raw_h5ad"], compression="gzip")
    adata = preprocess_adata(adata, n_top_genes=args.n_features, target_sum=1e4)
    if int(adata.var["highly_variable"].sum()) != args.n_features:
        raise RuntimeError(f"{sample}: failed to select {args.n_features} HVGs")

    radius, nearest = infer_radius(adata.obsm["spatial"])
    if args.rad_cutoff > 0:
        radius = float(args.rad_cutoff)
    Cal_Spatial_Net(adata, rad_cutoff=radius, model="Radius")
    mean_neighbors = len(adata.uns["Spatial_Net"]) / adata.n_obs
    if mean_neighbors < 2:
        raise RuntimeError(f"{sample}: spatial graph too sparse ({mean_neighbors:.3f})")
    Stats_Spatial_Net(
        adata, str(args.save_path), sample, dataset_name=args.dataset_name
    )
    x1, a1, metric_graph = process_data_view1(
        adata, k_neighbors=args.metric_neighbors
    )

    adata.obs["slices_path"] = write_center_crops(
        barcodes, patches, paths["crop_dir"], args.patch_size
    )
    adata = image_feature(
        adata,
        pca_components=args.image_pcs,
        cnnType="ResNet50",
        seeds=88,
    ).extract_image_feat()
    np.savez_compressed(
        paths["image_features"],
        barcodes=adata.obs_names.to_numpy(),
        image_feat_pca=adata.obsm["image_feat_pca"],
    )
    adata_image = get_augment(
        adata, spatial_type="LinearRegress", use_morphological=True
    )
    a2 = construct_interaction(adata, n_neighbors=args.image_neighbors)
    x2 = data_process(adata_image, pca_n_comps=args.n_features)
    if x1.shape != x2.shape:
        raise RuntimeError(f"{sample}: view shapes differ: X1={x1.shape}, X2={x2.shape}")
    if not all(np.isfinite(matrix).all() for matrix in (x1, x2, a1, a2)):
        raise RuntimeError(f"{sample}: non-finite prepared values")

    scio.savemat(
        paths["matrix"],
        {"X1": x1, "A1": a1, "X2": x2, "A2": a2, "metric_graph": metric_graph},
    )
    # Save a label-free preparation checkpoint for audit/replotting.
    adata.write_h5ad(paths["prepared_h5ad"], compression="gzip")
    summary = {
        "sample": sample,
        "spots": int(adata.n_obs),
        "raw_genes": int(adata.n_vars),
        "n_features": args.n_features,
        "image_pcs": args.image_pcs,
        "patch_source": "processed H5AD obsm['patches']",
        "source_patch_shape": list(patches.shape),
        "crosp_center_crop_px": args.patch_size,
        "median_nearest_distance": nearest,
        "rad_cutoff": radius,
        "mean_spatial_neighbors": mean_neighbors,
        "x1_shape": list(x1.shape),
        "x2_shape": list(x2.shape),
        "seconds": time.perf_counter() - started,
        "contains_ground_truth": False,
    }
    paths["prepare_summary"].write_text(json.dumps(summary, indent=2))
    print("[PREPARED] " + json.dumps(summary))
    del adata, adata_image, patches, x1, x2, a1, a2, metric_graph
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_config(args: argparse.Namespace, sample: str, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        data_name=sample,
        data_type="10X",
        task_type="Identify",
        dataset_name=args.dataset_name,
        n_spa=args.n_features,
        n_img=args.n_features,
        print_num=args.print_num,
        emb_dim=args.emb_dim,
        epoch=args.epoch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        d_prior=args.d_prior,
        gamma_init=args.gamma_init,
        img_scaling="minmax",
        fusion_type="scalar",
        save_view_embeddings=True,
        save_fusion_weights=False,
        model_seed=seed,
    )


def selection_rule(args: argparse.Namespace) -> str:
    if args.development_sample is None:
        return "prespecified configuration; labels used only after training"
    return (
        f"configuration selected on {args.development_sample}; frozen before "
        "evaluation of all other samples"
    )


def train_sample(args: argparse.Namespace, sample: str, seed: int) -> None:
    paths = run_paths(args, sample, seed)
    if not paths["matrix"].exists():
        raise FileNotFoundError(f"Prepare {sample} first: {paths['matrix']}")
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA unavailable; use the GPU environment or pass --allow-cpu")
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    config = model_config(args, sample, seed)
    model.setup_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    embedding = model.main(save_path=str(args.save_path), config=config)
    seconds = time.perf_counter() - started
    generic = paths["result_dir"] / f"{sample}_model_output.mat"
    payload = scio.loadmat(generic) if generic.exists() else {"data": embedding}
    payload = {key: value for key, value in payload.items() if not key.startswith("__")}
    payload["data"] = np.asarray(embedding)
    scio.savemat(paths["model_output"], payload)
    summary = {
        "sample": sample,
        "model_seed": seed,
        "seconds": seconds,
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated() / 1024**2)
            if torch.cuda.is_available()
            else 0.0
        ),
        "embedding_shape": list(embedding.shape),
        "ground_truth_loaded": False,
    }
    paths["training_summary"].write_text(json.dumps(summary, indent=2))
    config_payload = vars(config).copy()
    config_payload.update(
        {
            "run_name": args.run_name,
            "mclust_seed": args.mclust_seed,
            "n_clusters": N_CLUSTERS[sample],
            "development_sample": args.development_sample,
            "selection_rule": selection_rule(args),
            "model_py_sha256": sha256(ROOT / "model" / "CroSP.py"),
            "loss_py_sha256": sha256(ROOT / "model" / "loss.py"),
            "prepared_mat_sha256": sha256(paths["matrix"]),
        }
    )
    paths["config"].write_text(json.dumps(config_payload, indent=2))
    print("[TRAINED] " + json.dumps(summary))


def run_mclust(embedding: np.ndarray, clusters: int, seed: int) -> np.ndarray:
    matrix = np.ascontiguousarray(embedding, dtype=np.float64)
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri

    robjects.r.library("mclust")
    robjects.globalenv["crosp_input"] = numpy2ri.numpy2rpy(matrix)
    robjects.globalenv["crosp_k"] = robjects.IntVector([clusters])
    robjects.globalenv["crosp_seed"] = robjects.IntVector([seed])
    robjects.r(
        "set.seed(crosp_seed); "
        "crosp_fit <- Mclust(crosp_input, G=crosp_k, modelNames='EEE', verbose=FALSE)"
    )
    labels = np.asarray(
        robjects.globalenv["crosp_fit"].rx2("classification"), dtype=int
    )
    if labels.shape[0] != matrix.shape[0]:
        raise RuntimeError("mclust output length mismatch")
    return labels.astype(str)


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def plot_spatial(
    ax: plt.Axes,
    adata: ad.AnnData,
    column: str,
    colors: dict[str, str],
    title: str,
    sample: str,
) -> list:
    library = next(iter(adata.uns["spatial"]))
    spatial = adata.uns["spatial"][library]
    quality = spatial.get("use_quality", "lowres")
    image = spatial["images"][quality]
    ax.set_facecolor("white")
    labels = [str(value) for value in pd.unique(adata.obs[column].astype(str))]
    labels = [label for label in colors if label in labels]
    collections = []
    for label in labels:
        selected = adata.obs[column].astype(str).eq(label).to_numpy()
        collections.append(ax.scatter(
            adata.obs.loc[selected, "imagecol"],
            adata.obs.loc[selected, "imagerow"],
            s=1,
            c=colors[label],
            edgecolors="none",
            linewidths=0,
            alpha=1.0,
        ))
    x = adata.obs["imagecol"].to_numpy(float)
    y = adata.obs["imagerow"].to_numpy(float)
    margin = max(12.0, 0.055 * max(np.ptp(x), np.ptp(y)))
    x_min, x_max = max(0.0, x.min() - margin), min(image.shape[1], x.max() + margin)
    y_min, y_max = max(0.0, y.min() - margin), min(image.shape[0], y.max() + margin)
    ax.set_xlim((x_max, x_min) if sample in HORIZONTALLY_FLIPPED_STAGES else (x_min, x_max))
    ax.set_ylim((y_min, y_max) if sample in VERTICALLY_FLIPPED_STAGES else (y_max, y_min))
    ax.set_aspect("equal")
    ax.set_title(title, pad=5)
    ax.axis("off")
    return collections


def legend_handles(colors: dict[str, str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markeredgewidth=0,
            markersize=5,
            label=label,
        )
        for label, color in colors.items()
    ]


def save_evaluation_figure(
    adata: ad.AnnData,
    sample: str,
    seed: int,
    metrics: dict[str, float],
    png: Path,
    svg: Path,
) -> None:
    configure_plot_style()
    truth_colors = {
        region: REGION_COLORS[region]
        for region in REGION_ORDER
        if adata.obs["region"].eq(region).any()
    }
    cluster_labels = sorted(adata.obs["CroSP_cluster"].astype(str).unique(), key=int)
    sample_palette = list(truth_colors.values())
    if len(cluster_labels) > len(sample_palette):
        raise ValueError("More CroSP clusters than Ground Truth palette colors")
    predicted_colors = {
        label: sample_palette[index]
        for index, label in enumerate(cluster_labels)
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.25))
    truth_collections = plot_spatial(
        axes[0], adata, "region", truth_colors, f"{sample}: ground truth", sample
    )
    predicted_collections = plot_spatial(
        axes[1],
        adata,
        "CroSP_cluster",
        predicted_colors,
        f"CroSP (ARI = {metrics['ARI']:.3f})",
        sample,
    )
    axes[0].legend(
        handles=legend_handles(truth_colors),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=6.5,
        ncol=2,
    )
    axes[1].legend(
        handles=legend_handles(predicted_colors),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=6.5,
        ncol=3,
        title="CroSP cluster",
        title_fontsize=7,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.22, wspace=0.08)
    marker_area = apply_touching_marker_area(
        axes[0],
        adata.obs["imagecol"],
        adata.obs["imagerow"],
        [*truth_collections, *predicted_collections],
    )
    fig.canvas.draw()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg, format="svg")
    fig.savefig(png, format="png", dpi=300)
    plt.close(fig)


def evaluate_sample(args: argparse.Namespace, sample: str, seed: int) -> None:
    paths = run_paths(args, sample, seed)
    if not paths["raw_h5ad"].exists() or not paths["model_output"].exists():
        raise FileNotFoundError(f"Missing prepared/trained files for {sample} seed {seed}")
    adata = sc.read_h5ad(paths["raw_h5ad"])
    payload = scio.loadmat(paths["model_output"])
    latent = np.asarray(payload["data"])
    if latent.shape[0] != adata.n_obs:
        raise RuntimeError(f"{sample}: embedding rows do not match spots")
    components = min(args.pca_components, latent.shape[0] - 1, latent.shape[1])
    embedding = PCA(n_components=components, random_state=42).fit_transform(latent)
    prediction = run_mclust(embedding, N_CLUSTERS[sample], args.mclust_seed)
    truth = aligned_truth(paths["metadata"], sample, adata.obs_names)
    adata.obsm["CroSP_full"] = latent.astype(np.float32)
    adata.obsm["CroSP"] = embedding.astype(np.float32)
    adata.obs["CroSP_cluster"] = pd.Categorical(prediction)
    adata.obs["region"] = pd.Categorical(truth.loc[adata.obs_names].to_numpy())
    metrics = {
        "run_name": args.run_name,
        "sample": sample,
        "model_seed": seed,
        "mclust_seed": args.mclust_seed,
        "n_clusters": N_CLUSTERS[sample],
        "n_spots": int(adata.n_obs),
        "ARI": float(adjusted_rand_score(truth, prediction)),
        "NMI": float(normalized_mutual_info_score(truth, prediction)),
        "AMI": float(adjusted_mutual_info_score(truth, prediction)),
        "pca_components": components,
        "development_sample": args.development_sample or "none",
        "selection_rule": selection_rule(args),
    }
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    if paths["config"].exists():
        config_payload = json.loads(paths["config"].read_text())
        config_payload.update(
            {
                "pca_components": components,
                "development_sample": args.development_sample,
                "selection_rule": selection_rule(args),
            }
        )
        paths["config"].write_text(json.dumps(config_payload, indent=2))
    pd.DataFrame([metrics]).to_csv(paths["metrics"], index=False)
    adata.obs[["region", "CroSP_cluster"]].to_csv(
        paths["assignments"], index_label="barcode"
    )
    adata.uns["CroSP_run"] = metrics
    adata.write_h5ad(paths["final_h5ad"], compression="gzip")
    save_evaluation_figure(
        adata,
        sample,
        seed,
        metrics,
        paths["figure_png"],
        paths["figure_svg"],
    )
    print("[EVALUATED] " + json.dumps(metrics))


def save_combined_figure(args: argparse.Namespace, seed: int) -> None:
    available = [
        sample
        for sample in args.samples
        if run_paths(args, sample, seed)["final_h5ad"].exists()
    ]
    if not available:
        return
    configure_plot_style()
    fig, axes = plt.subplots(2, len(available), figsize=(3.8 * len(available), 7.0))
    axes = np.asarray(axes).reshape(2, len(available))
    marker_groups = []
    used_regions: set[str] = set()
    used_clusters: set[str] = set()
    for column, sample in enumerate(available):
        adata = sc.read_h5ad(run_paths(args, sample, seed)["final_h5ad"])
        truth_colors = {
            region: REGION_COLORS[region]
            for region in REGION_ORDER
            if adata.obs["region"].astype(str).eq(region).any()
        }
        cluster_labels = sorted(
            adata.obs["CroSP_cluster"].astype(str).unique(), key=int
        )
        sample_palette = list(truth_colors.values())
        if len(cluster_labels) > len(sample_palette):
            raise ValueError("More CroSP clusters than Ground Truth palette colors")
        predicted_colors = {
            label: sample_palette[index]
            for index, label in enumerate(cluster_labels)
        }
        used_regions.update(truth_colors)
        used_clusters.update(predicted_colors)
        ari = float(adata.uns["CroSP_run"]["ARI"])
        truth_collections = plot_spatial(
            axes[0, column], adata, "region", truth_colors, f"{sample}", sample
        )
        predicted_collections = plot_spatial(
            axes[1, column],
            adata,
            "CroSP_cluster",
            predicted_colors,
            f"{sample}  (ARI = {ari:.3f})",
            sample,
        )
        marker_groups.append(
            (
                axes[0, column],
                adata.obs["imagecol"].to_numpy(float),
                adata.obs["imagerow"].to_numpy(float),
                [*truth_collections, *predicted_collections],
            )
        )
    axes[0, 0].text(
        -0.08, 0.5, "Ground truth", transform=axes[0, 0].transAxes,
        rotation=90, va="center", ha="right", fontsize=10, fontweight="bold"
    )
    axes[1, 0].text(
        -0.08, 0.5, "CroSP", transform=axes[1, 0].transAxes,
        rotation=90, va="center", ha="right", fontsize=10, fontweight="bold"
    )
    truth_legend = {
        region: REGION_COLORS[region] for region in REGION_ORDER if region in used_regions
    }
    cluster_legend = {
        label: PREDICTION_COLORS[index]
        for index, label in enumerate(sorted(used_clusters, key=int))
    }
    first_legend = fig.legend(
        handles=legend_handles(truth_legend),
        loc="lower center", bbox_to_anchor=(0.35, 0.015), frameon=False,
        fontsize=6.5, ncol=3, title="Anatomical region", title_fontsize=7,
    )
    fig.add_artist(first_legend)
    fig.legend(
        handles=legend_handles(cluster_legend),
        loc="lower center", bbox_to_anchor=(0.79, 0.035), frameon=False,
        fontsize=6.5, ncol=4, title="CroSP cluster", title_fontsize=7,
    )
    fig.subplots_adjust(left=0.045, right=0.99, top=0.96, bottom=0.17, wspace=0.05, hspace=0.08)
    for ax, x, y, collections in marker_groups:
        apply_touching_marker_area(ax, x, y, collections)
    fig.canvas.draw()
    figure_dir = args.save_path / "Figure" / args.dataset_name / args.run_name
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = figure_dir / f"{args.dataset_name}_CroSP_D7_D10_D14_seed{seed}"
    fig.savefig(stem.with_suffix(".svg"), format="svg")
    fig.savefig(stem.with_suffix(".png"), format="png", dpi=300)
    plt.close(fig)


def aggregate(args: argparse.Namespace) -> None:
    rows: list[pd.DataFrame] = []
    for sample in args.samples:
        for seed in args.model_seeds:
            path = run_paths(args, sample, seed)["metrics"]
            if path.exists():
                rows.append(pd.read_csv(path))
    if not rows:
        raise FileNotFoundError("No chicken-heart metrics found for aggregation")
    metrics = pd.concat(rows, ignore_index=True)
    root = args.save_path / "Data" / args.dataset_name
    seed_tag = "seeds_" + "-".join(str(seed) for seed in args.model_seeds)
    metrics.to_csv(
        root / f"seed_metrics_{args.run_name}_{seed_tag}.csv", index=False
    )
    summary = (
        metrics.groupby("sample", sort=False)
        .agg(
            seeds=("model_seed", "nunique"),
            ARI_mean=("ARI", "mean"),
            ARI_sd=("ARI", "std"),
            ARI_min=("ARI", "min"),
            ARI_max=("ARI", "max"),
            NMI_mean=("NMI", "mean"),
            NMI_sd=("NMI", "std"),
            AMI_mean=("AMI", "mean"),
            AMI_sd=("AMI", "std"),
        )
        .reset_index()
    )
    summary.to_csv(
        root / f"seed_summary_{args.run_name}_{seed_tag}.csv", index=False
    )
    reporting_seed = 3 if 3 in args.model_seeds else args.model_seeds[0]
    save_combined_figure(args, reporting_seed)
    print("[AGGREGATE]\n" + summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.save_path = args.save_path.resolve()
    if args.stage in ("preflight", "all"):
        preflight(args)
    if args.stage in ("prepare", "all"):
        for sample in args.samples:
            model.setup_seed(args.model_seeds[0])
            prepare_sample(args, sample)
    if args.stage in ("train", "all"):
        for sample in args.samples:
            for seed in args.model_seeds:
                train_sample(args, sample, seed)
    if args.stage in ("evaluate", "all"):
        for sample in args.samples:
            for seed in args.model_seeds:
                evaluate_sample(args, sample, seed)
    if args.stage in ("aggregate", "all"):
        aggregate(args)


if __name__ == "__main__":
    main()
