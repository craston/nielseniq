import logging
import shutil
from enum import Enum
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pytesseract
import torch
import wget
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Define an Enum for fixed dataset columns
class DatasetColumns(Enum):
    IMAGE_URL = "image_url"
    ALLERGENS = "allergens"
    NUTRISCORE_GRADE = "nutriscore_grade"
    NUTRIENT_LIST = "nutrient_list"


class OFFDataset(Dataset):
    """
    Base class for all Open Food Facts (OFF) datasets.
    """

    def __init__(
        self,
        image_folder: Path,
        df: pd.DataFrame,
        image_transforms: Optional[List[Transform]] = None,
        label_column: Optional[str] = None,
        force_download: bool = False,
    ):
        """
        Args:
            image_folder (Path): Path to the folder containing images.
            df (pd.DataFrame): DataFrame containing dataset columns.
            image_transforms (list[Transform] | None): Image transformations.
            label_column (str | None): Column name containing labels.
            force_download (bool): If True, force image download.
        """
        self.image_folder = image_folder
        self.df = df
        self.image_transforms = image_transforms
        self.label_column = label_column

        # Prepare image directory and download missing images
        self._check_images_exist(force_download)

        # Create label mapping if a label column is provided
        if label_column:
            self.label_map = self._create_label_mapping(label_column)

    def _check_images_exist(self, force_download: bool):
        """
        Ensures images exist in `image_folder`, downloads missing images if necessary.
        """
        if force_download:
            shutil.rmtree(self.image_folder, ignore_errors=True)

        self.image_folder.mkdir(parents=True, exist_ok=True)

        logging.info(f"Checking for missing images in {self.image_folder}...")
        for url in tqdm(self.df[DatasetColumns.IMAGE_URL.value]):
            image_name = self._get_image_name(url)
            if not (self.image_folder / image_name).exists():
                self._download_image(url, self.image_folder / image_name)

    @staticmethod
    def _get_image_name(url: str) -> str:
        """
        Extracts a formatted image name from the OFF image URL.
        """
        try:
            name = "_".join(url.split("/")[-5:])
        except Exception as e:
            logger.error(f"Failed to extract image name from {url}. Error: {e}")
        return name

    @staticmethod
    def _download_image(url: str, save_path: Path):
        """
        Downloads an image from a URL and saves it to the given path.
        """
        try:
            wget.download(url, str(save_path))
        except Exception as e:
            logger.warning(f"Failed to download {url}. Skipping. Error: {e}")

    def _create_label_mapping(self, column_name: str) -> dict:
        """
        Creates a mapping from labels to integer indices.
        """
        unique_labels = set()
        for labels in self.df[column_name]:
            unique_labels.update(labels.split(","))  # Handles multi-class labels
        return {label: idx for idx, label in enumerate(sorted(unique_labels))}

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.df)

    def _load_image(self, idx: int) -> Optional[Image.Image]:
        """
        Loads and returns an image given the index.
        Args:
            idx (int): Index of the image to load.
        Returns:
            Image.Image: Loaded image.
        """
        image_name = self._get_image_name(
            self.df.iloc[idx][DatasetColumns.IMAGE_URL.value]
        )
        image_path = self.image_folder / image_name

        try:
            image = Image.open(image_path).convert("RGB")
            if self.image_transforms:
                image = self.image_transforms(image)
            return image
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}. Error: {e}")
            return None

    def __getitem__(self, idx: int):
        raise NotImplementedError("Subclasses must implement __getitem__.")


class OFFSingleClass(OFFDataset):
    """
    Dataset for Single-Class Classification (NutriScore classification).
    """

    def __init__(
        self,
        image_folder: Path,
        df: pd.DataFrame,
        image_transforms: Optional[List[Transform]] = None,
        force_download=False,
    ):
        super().__init__(
            image_folder,
            df,
            image_transforms,
            DatasetColumns.NUTRISCORE_GRADE.value,
            force_download,
        )

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a single sample from the dataset.
        Args:
            idx (int): Index of the sample to return.
        Returns:
            dict: A dictionary containing the image and label.
        """
        image = self._load_image(idx)
        if image is None:
            return None

        label = self.df.iloc[idx][DatasetColumns.NUTRISCORE_GRADE.value]
        return {
            "image": image,
            "label": torch.tensor(self.label_map[label], dtype=torch.long),
        }


class OFFMultiClass(OFFDataset):
    """
    Dataset for Multi-Class Classification (Allergen classification).
    """

    def __init__(
        self,
        image_folder: Path,
        df: pd.DataFrame,
        image_transforms: Optional[List[Transform]] = None,
        force_download=False,
    ):
        super().__init__(
            image_folder,
            df,
            image_transforms,
            DatasetColumns.ALLERGENS.value,
            force_download,
        )

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a single sample from the dataset.
        Args:
            idx (int): Index of the sample to return.
        Returns:
            dict: A dictionary containing the image and label.
        """
        image = self._load_image(idx)
        if image is None:
            return None

        labels = self.df.iloc[idx][DatasetColumns.ALLERGENS.value].split(",")
        label_vector = torch.zeros(len(self.label_map), dtype=torch.float32)
        for label in labels:
            label_vector[self.label_map[label]] = 1

        return {"image": image, "label": label_vector}


class OFFEntityExtraction(OFFDataset):
    """
    Dataset for OCR-Based Entity Extraction (Nutrient detection).
    """

    def __init__(
        self,
        image_folder: Path,
        df: pd.DataFrame,
        image_transforms: Optional[List[Transform]] = None,
        force_download=False,
    ):
        super().__init__(
            image_folder,
            df,
            image_transforms,
            DatasetColumns.NUTRIENT_LIST.value,
            force_download,
        )

    def embed_text(self, text: str) -> torch.Tensor:
        """
        Converts text to embeddings using a pre-trained model.
        """
        raise NotImplementedError("Subclasses must implement embed_text.")

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a single sample from the dataset.
        Args:
            idx (int): Index of the sample to return.
        Returns:
            dict: A dictionary containing the image, text, and label.
        """
        image = self._load_image(idx)
        if image is None:
            return None

        # Extract text using OCR
        extracted_text = pytesseract.image_to_string(image)

        # convert text to embeddings
        text_embedding = self.embed_text(extracted_text)

        labels = self.df.iloc[idx][DatasetColumns.NUTRIENT_LIST.value].split(",")
        label_vector = torch.zeros(len(self.label_map), dtype=torch.float32)
        for label in labels:
            label_vector[self.label_map[label]] = 1

        return {"image": image, "text": text_embedding, "label": label_vector}
