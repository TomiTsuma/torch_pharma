"""Dataset metadata and normalization constants for NExT-Mol."""

QM9_DF_CONFIG = {
    "name": "QM9-df",
    "pos_std": 1.7226,
    "max_atoms": 29,
    "max_sf_tokens": 256,
    "in_node_features": 44,
    "in_edge_features": 4,
    "atom_encoder": {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4},
}

GEOM_DRUGS_CONFIG = {
    "name": "Geom-drugs-df",
    "pos_std": 2.4777,
    "max_atoms": 181,
    "max_sf_tokens": 512,
    "in_node_features": 74,
    "in_edge_features": 4,
}

PROPERTY_NORMALIZATIONS = {
    "mu": {"mean": 2.6726, "mad": 1.0339},
    "alpha": {"mean": 75.1301, "mad": 8.7751},
    "homo": {"mean": -0.2399, "mad": 0.2141},
    "lumo": {"mean": 0.0116, "mad": 0.2141},
    "gap": {"mean": 0.2514, "mad": 0.2499},
    "Cv": {"mean": 31.5860, "mad": 4.4934},
}


def get_dataset_info(name: str) -> dict:
    key = name.lower()
    if "qm9" in key:
        return QM9_DF_CONFIG
    if "drug" in key or "geom" in key:
        return GEOM_DRUGS_CONFIG
    raise ValueError(f"Unknown dataset: {name}")
