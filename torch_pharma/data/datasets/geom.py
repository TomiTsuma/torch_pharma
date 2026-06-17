import os
from torch_pharma.paths import TORCH_PHARMA_HOME

# Lazy load/imports inside the factory to avoid circular imports or missing dependency errors during boot
class GEOMDatasetFactory:
    @staticmethod
    def get_dataset(dataset_type="drugs", **kwargs):
        """
        Factory method to resolve and instantiate the requested GEOM dataset variant.
        
        Parameters:
            dataset_type (str): "drugs" (GeomDrugsDataset from adaptive components),
                                "qm" (GeomQMDataset from adaptive qm9 components),
                                or "tordf" (GeomDrugsTorDFDataset from nextmol components).
        """
        if dataset_type == "drugs":
            from torch_pharma.data.components.geom.geom_dataset_adaptive import GeomDrugsDataset
            return GeomDrugsDataset(**kwargs)
        elif dataset_type == "qm":
            from torch_pharma.data.components.geom.geom_dataset_adaptive_qm9 import GeomQMDataset
            return GeomQMDataset(**kwargs)
        elif dataset_type == "tordf":
            from torch_pharma.data.components.nextmol.datasets.geom_drugs import GeomDrugsTorDFDataset
            return GeomDrugsTorDFDataset(**kwargs)
        else:
            raise ValueError(f"Unknown GEOM dataset type: {dataset_type}")
