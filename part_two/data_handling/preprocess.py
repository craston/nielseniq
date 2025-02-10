import pandas as pd
from part_two.data_handling.off import DatasetColumns
from part_two.data_handling.splitters import AttributeSplitter, RowSplitter


def split_dataset(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Split the dataset into train and test sets based on the configuration.
    """
    # Step 1: Apply AttributeSplitter (if enabled)
    if config["splitter"]["attribute_splitter"]["enabled"]:
        col_names = config["splitter"]["attribute_splitter"]["columns"]

        # Ensure all columns exist in the DatasetColumns Enum
        try:
            dataset_cols = [DatasetColumns[col] for col in col_names]
        except KeyError as e:
            raise ValueError(f"Invalid column name in YAML: {e}")
        attribute_splitter = AttributeSplitter(df, dataset_cols)
        df = attribute_splitter.split()  # Update df with selected columns

    # Step 2: Apply RowSplitter (if enabled)
    if config["splitter"]["row_splitter"]["enabled"]:
        fractions = config["splitter"]["row_splitter"]["fractions"]

        assert len(fractions) == 2
        "RowSplitter requires exactly two fractions for train-test split."

        row_splitter = RowSplitter(df, fractions)
        split_data = row_splitter.split()
    else:
        # If row splitting is disabled, return then split by the deafult 80-20 split
        row_splitter = RowSplitter(df, [0.8, 0.2])
        split_data = row_splitter.split()

    return split_data
