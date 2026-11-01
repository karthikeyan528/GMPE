import numpy as np
import pandas as pd


class DatasetProfiler:
    def profile(self, dataframe: pd.DataFrame) -> dict:
        numeric = dataframe.select_dtypes(include=[np.number])
        categorical = dataframe.select_dtypes(include=["object", "category"])
        return {
            "shape": tuple(dataframe.shape),
            "columns": list(dataframe.columns),
            "dtypes": dataframe.dtypes.astype(str).to_dict(),
            "missing_percent": (dataframe.isna().mean() * 100).round(2).to_dict(),
            "duplicate_rows": int(dataframe.duplicated().sum()),
            "numeric_summary": numeric.describe().to_dict() if not numeric.empty else {},
            "categorical_summary": categorical.describe().to_dict() if not categorical.empty else {},
        }
