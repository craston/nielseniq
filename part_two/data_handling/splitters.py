import pandas as pd
import torch
from part_two.data_handling.off import DatasetColumns
from abc import ABC, abstractmethod


class Splitter(ABC):
    """
    Abstract Base Class for different types of DataFrame splitters.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
        df (pd.DataFrame): DataFrame to be split.
        """
        self.df = df

    @abstractmethod
    def split(self):
        """Abstract method for splitting data."""
        pass


class RowSplitter(Splitter):
    """
    Splits a DataFrame row-wise into multiple sets based on specified fractions.
    """

    def __init__(self, df: pd.DataFrame, fractions: list[float]):
        """
        Args:
        df (pd.DataFrame): DataFrame to be split row-wise into sets.
        fractions (list[float]): List of fractions to split the DataFrame into.
                                 sum(fractions) should be equal to 1.
        """
        super().__init__(df)
        assert abs(sum(fractions) - 1.0) < 1e-6, (
            "Sum of fractions should be equal to 1."
        )
        self.fractions = fractions

    def split(self) -> list[pd.DataFrame]:
        """
        Splits the DataFrame into sets based on the fractions provided.

        Returns:
        list[pd.DataFrame]: List of DataFrame subsets.

        Ensure that torch is globally seeded before calling this function.
        """
        num_rows = len(self.df)
        num_splits = [int(frac * num_rows) for frac in self.fractions]
        num_splits[-1] = num_rows - sum(num_splits[:-1])  # Ensure all rows are included

        # Generate shuffled indices using PyTorch (globally seeded)
        indices = torch.randperm(num_rows)

        # Split indices based on computed row counts
        start_idx = 0
        split_dfs = []
        for count in num_splits:
            split_indices = indices[start_idx : start_idx + count].numpy()
            split_dfs.append(self.df.iloc[split_indices])
            start_idx += count

        return split_dfs


class AttributeSplitter(Splitter):
    """
    Splits a DataFrame column-wise by selecting specific columns.
    """

    def __init__(self, df: pd.DataFrame, col_names: list[DatasetColumns]):
        """
        Args:
        df (pd.DataFrame): DataFrame to be split column-wise.
        col_names (list[DatasetColumns]): List of column names to be selected.
        """
        super().__init__(df)
        self.col_names = col_names

    def split(self) -> pd.DataFrame:
        """
        Selects the columns from the DataFrame based on the column names provided.

        Returns:
        pd.DataFrame: Subset of the original DataFrame containing the selected columns.
        """
        cols = [col.value for col in self.col_names]

        # check if the columns are present in the DataFrame
        assert all(col in self.df.columns for col in cols), (
            "Column not found in DataFrame."
        )

        return self.df[cols]
