"""
prepare_inference_dataset.py

Prepare the transformer inference dataset in the same way as training, but for inference data directory.
"""
from pathlib import Path
from src.modeling.prepare_dataset import prepare_transformer_dataset

def prepare_inference_transformer_dataset(
    source_feature_table_path: Path,
    output_csv_path: Path,
    split_report_path: Path,
    split_seed: int = 42,
    split_train: float = 0.7,
    split_val: float = 0.15,
    split_test: float = 0.15,
):
    """
    Prepare the transformer inference dataset (same logic as training, but for inference directory).
    """
    return prepare_transformer_dataset(
        source_feature_table_path=source_feature_table_path,
        output_csv_path=output_csv_path,
        split_report_path=split_report_path,
        split_seed=split_seed,
        split_train=split_train,
        split_val=split_val,
        split_test=split_test,
    )
