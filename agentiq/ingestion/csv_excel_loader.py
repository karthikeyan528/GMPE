from pathlib import Path

import pandas as pd

from .validators import validate_dataframe, validate_uploaded_file


class DataLoader:
    """Load supported tabular files and apply common validation."""

    def load(self, path: str | Path) -> pd.DataFrame:
        file_path = Path(path)
        validate_uploaded_file(file_path)
        if file_path.suffix.lower() == ".csv":
            dataframe = pd.read_csv(file_path)
        else:
            dataframe = pd.read_excel(file_path)
        return validate_dataframe(dataframe)
