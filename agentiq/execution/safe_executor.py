import pandas as pd


class SafeExecutionError(RuntimeError):
    """Raised when a requested operation is not in the allowed registry."""


class SafeExecutor:
    _OPERATIONS = {"describe", "corr"}

    def __init__(self) -> None:
        self.logs: list[dict] = []

    def execute(
        self,
        dataframe: pd.DataFrame,
        operation: str,
        columns: list[str] | None = None,
        filters: dict[str, object] | None = None,
    ) -> dict:
        self.logs.append({"operation": operation, "status": "started"})
        try:
            if operation not in self._OPERATIONS:
                raise SafeExecutionError(f"Unsupported operation: {operation}")
            selected = dataframe
            if columns:
                missing_columns = [column for column in columns if column not in selected]
                if missing_columns:
                    raise SafeExecutionError(
                        f"Unknown columns: {', '.join(missing_columns)}"
                    )
                selected = selected.loc[:, columns]
            for column, value in (filters or {}).items():
                if column not in dataframe:
                    raise SafeExecutionError(f"Unknown filter column: {column}")
                selected = selected.loc[dataframe[column].eq(value)]
            if selected.empty:
                raise SafeExecutionError("No rows matched the requested filters.")
            if operation == "describe":
                result = selected.describe(include="all").to_dict()
            else:
                if columns and not all(
                    pd.api.types.is_numeric_dtype(selected[column])
                    for column in columns
                ):
                    raise SafeExecutionError(
                        "Correlation requires numeric columns."
                    )
                if columns and len(selected) < 2:
                    raise SafeExecutionError(
                        "At least two matching rows are required for correlation."
                    )
                result = selected.corr(numeric_only=True).to_dict()
        except Exception as exc:
            self.logs.append({"operation": operation, "status": "failed", "error": str(exc)})
            if isinstance(exc, SafeExecutionError):
                raise
            raise SafeExecutionError(f"Safe execution failed: {exc}") from exc
        self.logs.append({"operation": operation, "status": "success"})
        return result

    def recommend_correlations(
        self,
        dataframe: pd.DataFrame,
        filters: dict[str, object] | None = None,
    ) -> list[dict[str, object]]:
        selected = dataframe
        for column, value in (filters or {}).items():
            if column not in dataframe:
                raise SafeExecutionError(f"Unknown filter column: {column}")
            selected = selected.loc[dataframe[column].eq(value)]

        numeric_columns = list(
            selected.select_dtypes(include="number").columns
        )
        recommendations: list[dict[str, object]] = []
        for index, first_column in enumerate(numeric_columns):
            for second_column in numeric_columns[index + 1 :]:
                pair = selected[[first_column, second_column]].dropna()
                if (
                    len(pair) < 2
                    or pair[first_column].nunique() < 2
                    or pair[second_column].nunique() < 2
                ):
                    continue
                correlation = pair[first_column].corr(pair[second_column])
                recommendations.append(
                    {
                        "column 1": first_column,
                        "column 2": second_column,
                        "complete rows": len(pair),
                        "correlation": round(float(correlation), 4),
                    }
                )

        return sorted(
            recommendations,
            key=lambda result: abs(float(result["correlation"])),
            reverse=True,
        )
