from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from agentiq.execution import SafeExecutor
from agentiq.ingestion import DataValidationError, validate_dataframe
from agentiq.planning import PlannerAgent
from agentiq.profiling import DatasetProfiler


def load_uploaded_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise DataValidationError("Supported formats are CSV and Excel.")
    if not content:
        raise DataValidationError("Uploaded file is empty.")

    source = BytesIO(content)
    dataframe = pd.read_csv(source) if suffix == ".csv" else pd.read_excel(source)
    return validate_dataframe(dataframe)


def main() -> None:
    st.set_page_config(page_title="AgentIQ Data Analyst", layout="wide")
    st.title("AgentIQ | AI Data Analyst")
    st.caption(
        "Upload a dataset to explore it in your browser. The current planner is "
        "rule-based; it does not call an external AI model."
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file", type=["csv", "xlsx", "xls"]
    )
    if uploaded_file is None:
        st.info("Upload a dataset to begin.")
        return

    try:
        dataframe = load_uploaded_dataframe(
            uploaded_file.name, uploaded_file.getvalue()
        )
    except (
        DataValidationError,
        ImportError,
        OSError,
        UnicodeError,
        ValueError,
        pd.errors.ParserError,
    ) as exc:
        st.error(f"Could not load this dataset: {exc}")
        return

    st.subheader("Dataset preview")
    st.dataframe(dataframe.head(100), use_container_width=True)

    profile = DatasetProfiler().profile(dataframe)
    st.subheader("Dataset profile")
    first, second, third = st.columns(3)
    first.metric("Rows", profile["shape"][0])
    second.metric("Columns", profile["shape"][1])
    third.metric("Duplicate rows", profile["duplicate_rows"])

    profile_column, summary_column = st.columns(2)
    with profile_column:
        st.markdown("**Columns and missing values**")
        st.dataframe(
            pd.DataFrame(
                {
                    "data type": profile["dtypes"],
                    "missing (%)": profile["missing_percent"],
                }
            ),
            use_container_width=True,
        )
    with summary_column:
        summary = profile["numeric_summary"] or profile["categorical_summary"]
        if summary:
            st.markdown("**Summary statistics**")
            st.dataframe(pd.DataFrame(summary), use_container_width=True)
        else:
            st.info("No numeric or categorical summary is available.")

    st.subheader("Ask about your data")
    question = st.text_input(
        "Analysis question",
        placeholder="For example: analyze sales and profit for product Widget A",
    )
    if not question.strip():
        return

    plan = PlannerAgent().build_plan(question, list(dataframe.columns), dataframe)
    if plan.selected_columns:
        st.caption(f"Selected columns: {', '.join(plan.selected_columns)}")
    if plan.filters:
        column, value = next(iter(plan.filters.items()))
        st.caption(f"Product filter: {column} = {value}")
    elif len(plan.selected_columns) == 2:
        st.caption(
            "No product filter was recognized. Mention a product value after "
            "'for product' to limit the analysis."
        )
    st.markdown("**Analysis plan**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "step": step.name,
                    "method": step.method,
                    "column": step.column,
                    "selected columns": ", ".join(plan.selected_columns),
                    "reason": step.reason,
                }
                for step in plan.steps
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )

    if st.button("Run analysis", type="primary"):
        executor = SafeExecutor()
        for step in plan.steps:
            try:
                result = executor.execute(
                    dataframe,
                    step.method,
                    columns=plan.selected_columns or None,
                    filters=plan.filters or None,
                )
            except RuntimeError as exc:
                st.warning(f"Could not run {step.method}: {exc}")
                if step.method == "corr":
                    recommendations = executor.recommend_correlations(
                        dataframe, plan.filters or None
                    )
                    if recommendations:
                        st.markdown(
                            "**Recommended numeric column pairs** "
                            "(based on complete rows in the selected data):"
                        )
                        st.dataframe(
                            pd.DataFrame(recommendations),
                            hide_index=True,
                            use_container_width=True,
                        )
                    else:
                        st.info(
                            "No numeric column pair has at least two complete "
                            "rows with varying values for this selection. Add "
                            "more rows for this product or broaden/remove the "
                            "product filter."
                        )
                continue
            st.markdown(f"**{step.name.replace('_', ' ').title()}**")
            st.dataframe(pd.DataFrame(result), use_container_width=True)


if __name__ == "__main__":
    main()
