"""
Abstract base class for all fingerprint datasets.
Each new dataset subclasses BaseDataset and implements the interface below.
"""

from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """
    Interface every dataset implementation must satisfy.

    Workflow:
        1. reorganize()         — copy raw files into subject subfolders
        2. preprocess()         — resize images and normalize format
        3. load_images()        — load all images into memory as numpy arrays
        4. create_generators()  — create Keras train/val generators (Stage 1)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @abstractmethod
    def detect_format(self, source_dir: str) -> str:
        """
        Detect the layout of raw source files.
        Returns a string tag (e.g. 'flat', 'nested') used by reorganize().
        """

    @abstractmethod
    def reorganize(self, source_dir: str, out_dir: str, num_subjects: int) -> None:
        """
        Copy raw files into standardized subject subfolders:
            out_dir/subject_001/  img1.jpg ...
            out_dir/subject_002/  ...
        Must be idempotent (skip if already done).
        """

    @abstractmethod
    def preprocess(self, src_dir: str, dst_dir: str, img_size: tuple) -> None:
        """
        Resize and normalize all images for model input.
        Must be idempotent (skip if already done).
        """

    @abstractmethod
    def load_images(self, processed_dir: str, img_size: tuple) -> dict:
        """
        Load all preprocessed images into memory.
        Returns: { subject_name: [img_array, ...] }
        """

    @abstractmethod
    def create_generators(self, processed_dir: str, img_size: tuple,
                          batch_size: int, val_split: float):
        """
        Create Keras ImageDataGenerator train/val generators for Stage 1 classification.
        Returns: (train_gen, val_gen)
        """
