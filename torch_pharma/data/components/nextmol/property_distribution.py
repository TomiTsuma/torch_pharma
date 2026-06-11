"""Property normalization for conditional NExT-Mol generation."""

from torch_pharma.data.components.nextmol.dataset_config import PROPERTY_NORMALIZATIONS


class PropertyDistribution:
    def __init__(self, property_name: str):
        if property_name not in PROPERTY_NORMALIZATIONS:
            raise ValueError(f"Unknown property: {property_name}")
        stats = PROPERTY_NORMALIZATIONS[property_name]
        self.mean = stats["mean"]
        self.mad = stats["mad"]

    def normalize(self, value: float) -> float:
        return (value - self.mean) / self.mad

    def denormalize(self, value: float) -> float:
        return value * self.mad + self.mean
