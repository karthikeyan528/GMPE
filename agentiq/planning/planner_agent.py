from dataclasses import dataclass, field
import re

import pandas as pd


@dataclass(frozen=True)
class AnalysisStep:
    name: str
    column: str | None
    method: str
    reason: str


@dataclass
class AnalysisPlan:
    question: str
    relevant_columns: list[str] = field(default_factory=list)
    steps: list[AnalysisStep] = field(default_factory=list)
    selected_columns: list[str] = field(default_factory=list)
    filters: dict[str, object] = field(default_factory=dict)


class PlannerAgent:
    def build_plan(
        self,
        question: str,
        columns: list[str],
        dataframe: pd.DataFrame | None = None,
    ) -> AnalysisPlan:
        relevant = self._infer_columns(question, columns)
        selected_columns = self._select_numeric_pair(relevant, dataframe)
        filters = self._infer_product_filter(question, dataframe)
        steps = [AnalysisStep(
            "descriptive_summary", None, "describe",
            "Establish the dataset distribution before deeper analysis.",
        )]
        if len(selected_columns) == 2 or (
            "correlation" in question.lower() and relevant
        ):
            steps.append(AnalysisStep(
                "correlation_analysis",
                selected_columns[0] if selected_columns else relevant[0],
                "corr",
                "Quantify the relationship between the selected numeric variables.",
            ))
        return AnalysisPlan(
            question, relevant, steps, selected_columns=selected_columns, filters=filters
        )

    @staticmethod
    def _infer_columns(question: str, columns: list[str]) -> list[str]:
        return [
            column
            for column in columns
            if re.search(
                rf"(?<!\w){re.escape(column)}(?!\w)", question, flags=re.IGNORECASE
            )
        ]

    @staticmethod
    def _select_numeric_pair(
        relevant: list[str], dataframe: pd.DataFrame | None
    ) -> list[str]:
        if dataframe is None:
            return []
        numeric_columns = set(
            dataframe.select_dtypes(include="number").columns
        )
        return [column for column in relevant if column in numeric_columns]

    @staticmethod
    def _infer_product_filter(
        question: str, dataframe: pd.DataFrame | None
    ) -> dict[str, object]:
        if dataframe is None:
            return {}

        product_columns = [
            column
            for column in dataframe.columns
            if str(column).casefold() in {"product", "product_name", "product name"}
        ]
        if not product_columns:
            product_columns = list(
                dataframe.select_dtypes(include=["object", "category"]).columns
            )

        matches: list[tuple[int, str, object]] = []
        for column in product_columns:
            for value in dataframe[column].dropna().unique():
                value_text = str(value).strip()
                if not value_text:
                    continue
                match = re.search(
                    rf"(?<!\w){re.escape(value_text)}(?!\w)",
                    question,
                    flags=re.IGNORECASE,
                )
                if match and re.search(r"\b(?:for|of)\b", question[: match.start()], re.I):
                    matches.append((len(value_text), str(column), value))

        if not matches:
            return {}
        _, column, value = max(matches, key=lambda match: match[0])
        return {column: value}
