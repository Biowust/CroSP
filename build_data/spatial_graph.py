"""Optional spatial-graph interface that preserves historical CroSP code.

The inherited preprocessing functions and their platform-specific radius
defaults remain untouched.  New callers can use this module to choose Radius,
KNN, or an exact distance-free 10x Visium lattice graph.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as scio

from .module import (
    Cal_Spatial_Net,
    Stats_Spatial_Net,
    construct_interaction,
    data_process,
    get_adata,
    get_augment,
    get_image_crop,
    preprocess_adata,
)
from .demo_t import process_data_view1


VISIUM_LATTICE_OFFSETS = (
    (0, 2), (0, -2), (1, 1), (1, -1), (-1, 1), (-1, -1),
)
PLATFORM_GRAPH_DEFAULTS = {
    "10X": {"graph_type": "Radius", "rad_cutoff": 150, "k_cutoff": 6},
    "STARmap": {"graph_type": "Radius", "rad_cutoff": 400, "k_cutoff": 6},
    "osmFISH": {"graph_type": "Radius", "rad_cutoff": 500, "k_cutoff": 6},
    "Slide_seqV2": {"graph_type": "Radius", "rad_cutoff": 50, "k_cutoff": 6},
    "Stereo_seq": {"graph_type": "KNN", "rad_cutoff": None, "k_cutoff": 6},
}


def canonical_graph_type(graph_type: str) -> str:
    aliases = {
        "radius": "Radius",
        "knn": "KNN",
        "lattice": "VisiumLattice",
        "topology": "VisiumLattice",
        "visiumlattice": "VisiumLattice",
        "visium_lattice": "VisiumLattice",
    }
    key = str(graph_type).replace("-", "_").lower()
    if key not in aliases:
        raise ValueError(
            "graph_type must be Radius, KNN, or VisiumLattice; "
            f"got {graph_type!r}"
        )
    return aliases[key]


def visium_lattice_network(
    adata: Any,
    array_row_key: str = "array_row",
    array_col_key: str = "array_col",
) -> pd.DataFrame:
    missing = [key for key in (array_row_key, array_col_key) if key not in adata.obs]
    if missing:
        raise ValueError(
            "VisiumLattice requires Space Ranger array coordinates; "
            f"missing {missing}"
        )
    rows = pd.to_numeric(adata.obs[array_row_key], errors="raise").to_numpy()
    cols = pd.to_numeric(adata.obs[array_col_key], errors="raise").to_numpy()
    if not np.isfinite(rows).all() or not np.isfinite(cols).all():
        raise ValueError("Visium lattice coordinates must be finite")
    if not np.equal(rows, np.rint(rows)).all() or not np.equal(cols, np.rint(cols)).all():
        raise ValueError("Visium lattice coordinates must be integer-valued")
    coordinates = list(zip(rows.astype(int), cols.astype(int)))
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("Visium lattice coordinates must be unique per spot")
    coordinate_to_index = {
        coordinate: index for index, coordinate in enumerate(coordinates)
    }
    records = []
    for source, (array_row, array_col) in enumerate(coordinates):
        for row_offset, col_offset in VISIUM_LATTICE_OFFSETS:
            target = coordinate_to_index.get(
                (array_row + row_offset, array_col + col_offset)
            )
            if target is not None:
                records.append((
                    str(adata.obs_names[source]),
                    str(adata.obs_names[target]),
                    1.0,
                ))
    return pd.DataFrame(records, columns=["Cell1", "Cell2", "Distance"])


def construct_spatial_network(
    adata: Any,
    graph_type: str = "Radius",
    rad_cutoff: float | None = None,
    k_cutoff: int | None = None,
    array_row_key: str = "array_row",
    array_col_key: str = "array_col",
    verbose: bool = True,
) -> Any:
    """Construct a selectable spatial graph and return the modified AnnData."""
    graph_type = canonical_graph_type(graph_type)
    if graph_type == "Radius":
        if rad_cutoff is None or float(rad_cutoff) <= 0:
            raise ValueError("Radius graph requires a positive rad_cutoff")
        Cal_Spatial_Net(
            adata, rad_cutoff=float(rad_cutoff), model="Radius", verbose=verbose
        )
        config = {"graph_type": graph_type, "rad_cutoff": float(rad_cutoff)}
    elif graph_type == "KNN":
        if k_cutoff is None or int(k_cutoff) < 1 or int(k_cutoff) >= adata.n_obs:
            raise ValueError(
                "KNN graph requires 1 <= k_cutoff < number of spots"
            )
        Cal_Spatial_Net(
            adata, k_cutoff=int(k_cutoff), model="KNN", verbose=verbose
        )
        config = {"graph_type": graph_type, "k_cutoff": int(k_cutoff)}
    else:
        if verbose:
            print("------Calculating exact Visium lattice graph...")
        adata.uns["Spatial_Net"] = visium_lattice_network(
            adata,
            array_row_key=array_row_key,
            array_col_key=array_col_key,
        )
        config = {
            "graph_type": graph_type,
            "array_row_key": array_row_key,
            "array_col_key": array_col_key,
        }
        if verbose:
            edges = adata.uns["Spatial_Net"].shape[0]
            print(f"The graph contains {edges} edges, {adata.n_obs} cells.")
            print(f"{edges / adata.n_obs:.4f} neighbors per cell on average.")
    adata.uns["Spatial_Net_Config"] = config
    return adata


def process_Visium_with_graph(
    platform: str,
    save_path: str,
    data_path: str,
    data_name: str,
    dataset_name: str,
    task_type: str = "Identify",
    pca_n_comps: int = 3000,
    n_top_genes: int = 3000,
    spatial_type: str = "LinearRegress",
    use_morphological: bool = True,
    spatial_graph_type: str = "Radius",
    rad_cutoff: float = 150,
    k_cutoff: int = 6,
) -> None:
    """Visium preprocessing with a selectable spatial graph.

    Defaults reproduce the inherited ``process_Visium`` Radius-150 route.
    """
    adata = get_adata(
        platform=platform,
        save_path=save_path,
        data_path=data_path,
        data_name=data_name,
        dataset_name=dataset_name,
    )
    adata = preprocess_adata(adata, n_top_genes=n_top_genes)
    construct_spatial_network(
        adata,
        graph_type=spatial_graph_type,
        rad_cutoff=rad_cutoff,
        k_cutoff=k_cutoff,
    )
    Stats_Spatial_Net(adata, save_path, data_name, dataset_name)
    x1, a1, metric_graph = process_data_view1(adata)
    adata = get_image_crop(adata, save_path=save_path, data_name=data_name)
    adata_image = get_augment(
        adata,
        spatial_type=spatial_type,
        use_morphological=use_morphological,
    )
    a2 = construct_interaction(adata, n_neighbors=6)
    x2 = data_process(adata_image, pca_n_comps=pca_n_comps)
    data_save_path = os.path.join(save_path, "Data", dataset_name, data_name)
    os.makedirs(data_save_path, exist_ok=True)
    scio.savemat(
        os.path.join(data_save_path, f"data_{data_name}.mat"),
        {"X1": x1, "A1": a1, "X2": x2, "A2": a2, "metric_graph": metric_graph},
    )
    print("process_data_and_save complete!")
