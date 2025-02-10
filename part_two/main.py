from enum import Enum
import logging
from pathlib import Path

import pandas as pd
import torch
import typer
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from train import TrainerClassifier, TrainerNER

from data_handling.off import (
    OFFEntityExtraction,
    OFFMultiClass,
    OFFSingleClass,
)
from data_handling.preprocess import split_dataset
from utils import set_global_seed

app = typer.Typer()

logging.basicConfig(level=logging.INFO)


class TaskType(Enum):  # noqa: F821
    CLASSIFICATION = "classification"
    MULTICLASS = "multiclass"
    NER = "ner"


DATASET_MAPPING = {
    TaskType.CLASSIFICATION: OFFSingleClass,
    TaskType.MULTICLASS: OFFMultiClass,
    TaskType.NER: OFFEntityExtraction,
}

TRAINER_MAPPING = {
    TaskType.CLASSIFICATION: TrainerClassifier,
    TaskType.MULTICLASS: TrainerClassifier,
    TaskType.NER: TrainerNER,
}


def load_yaml_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


@app.command()
def run(
    csv_file: Path = typer.Option(..., help="Path to the CSV file."),
    config_file: Path = typer.Option(..., help="Path to the YAML config file."),
    image_folder: Path = typer.Option(..., help="Path to store images."),
    checkpoint_path: Path = typer.Option(
        Path("./checkpoints"), help="Path to save checkpoints."
    ),
    resume: bool = typer.Option(False, help="Resume training from a checkpoint."),
):
    # Load configuration
    config = load_yaml_config(config_file)
    set_global_seed(config["seed"])

    # Load dataset
    df = pd.read_csv(csv_file)
    df_train, df_test = split_dataset(df, config)

    # Define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Create dataset
    dataset_class = DATASET_MAPPING[TaskType(config["task"])]

    train_dataset = dataset_class(image_folder, df_train, image_transforms=transform)
    test_dataset = dataset_class(image_folder, df_test, image_transforms=transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Initialize model, criterion, optimizer
    model = config["model"]()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Initialize Trainer
    trainer_class = TRAINER_MAPPING[TaskType(config["task"])]
    trainer = trainer_class(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=config["device"],
        checkpoint_path=checkpoint_path,
        resume=resume,
    )

    # Train the model
    trainer.train(train_loader, config["epochs"])

    # Evaluate the model
    accuracy = trainer.test(test_loader)
    logging.info(f"Final Test Accuracy: {accuracy:.2f}%")
