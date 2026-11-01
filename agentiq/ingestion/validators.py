from pathlib import Path

import pandas as pd


class DataValidationError(ValueError):
    """Raised when an input file or dataframe cannot be analyzed."""


def validate_uploaded_file(path: str | Path) -> None:
    file_path = Path(path)
    if not file_path.exists():
        raise DataValidationError(f"File not found: {file_path}")
    if file_path.stat().st_size == 0:
        raise DataValidationError("Uploaded file is empty.")
    if file_path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
        raise DataValidationError("Supported formats are CSV and Excel.")


def validate_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        raise DataValidationError("Dataset is empty after loading.")
    if dataframe.columns.duplicated().any():
        raise DataValidationError("Duplicate column names detected in the dataset.")
    return dataframe
