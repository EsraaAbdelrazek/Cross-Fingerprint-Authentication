from .base_dataset import BaseDataset
from .polyu_dataset import PolyUDataset
from .pair_generator import generate_pairs, split_pairs

DATASET_REGISTRY = {
    "polyu": PolyUDataset,
}
