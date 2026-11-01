import pandas as pd
import pytest

from agentiq.execution import SafeExecutionError, SafeExecutor
from agentiq.ingestion import DataLoader, DataValidationError
from agentiq.planning import PlannerAgent
from agentiq.profiling import DatasetProfiler


def test_loader_profiles_and_plans_csv(tmp_path):
    source = tmp_path / "sales.csv"
    source.write_text("sales,customers\n10,2\n20,4\n", encoding="utf-8")

    dataframe = DataLoader().load(source)
    profile = DatasetProfiler().profile(dataframe)
    plan = PlannerAgent().build_plan("show sales correlation", list(dataframe.columns))

    assert profile["shape"] == (2, 2)
    assert plan.relevant_columns == ["sales"]
    assert [step.method for step in plan.steps] == ["describe", "corr"]


def test_loader_rejects_unsupported_files(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("not data", encoding="utf-8")

    with pytest.raises(DataValidationError, match="Supported formats"):
        DataLoader().load(source)


def test_executor_restricts_operations_and_logs_failures():
    dataframe = pd.DataFrame({"sales": [10, 20], "customers": [2, 4]})
    executor = SafeExecutor()

    result = executor.execute(dataframe, "corr")
    assert result["sales"]["sales"] == 1.0

    with pytest.raises(SafeExecutionError, match="Unsupported operation"):
        executor.execute(dataframe, "eval")
    assert executor.logs[-1]["status"] == "failed"


def test_planner_and_executor_analyze_two_columns_for_one_product():
    dataframe = pd.DataFrame(
        {
            "Product": ["Widget A", "Widget A", "Widget B"],
            "sales": [1, 2, 100],
            "profit": [2, 4, 1],
            "visits": [10, 20, 5],
        }
    )
    plan = PlannerAgent().build_plan(
        "analyze sales and profit for product Widget A",
        list(dataframe.columns),
        dataframe,
    )

    assert plan.selected_columns == ["sales", "profit"]
    assert plan.filters == {"Product": "Widget A"}

    result = SafeExecutor().execute(
        dataframe,
        "corr",
        columns=plan.selected_columns,
        filters=plan.filters,
    )
    assert set(result) == {"sales", "profit"}
    assert result["sales"]["profit"] == 1.0
    assert result["sales"].get("visits") is None


def test_planner_and_executor_analyze_three_requested_columns():
    dataframe = pd.DataFrame(
        {
            "Product": ["Widget A", "Widget A", "Widget A"],
            "sales": [1, 2, 3],
            "profit": [2, 4, 6],
            "units": [3, 6, 9],
            "unmentioned": [30, 20, 10],
        }
    )
    plan = PlannerAgent().build_plan(
        "compare sales, profit and units for product Widget A",
        list(dataframe.columns),
        dataframe,
    )

    assert plan.selected_columns == ["sales", "profit", "units"]
    result = SafeExecutor().execute(
        dataframe,
        "corr",
        columns=plan.selected_columns,
        filters=plan.filters,
    )
    assert set(result) == {"sales", "profit", "units"}
    assert result["sales"]["units"] == 1.0


def test_executor_reports_when_product_filter_matches_too_few_rows():
    dataframe = pd.DataFrame(
        {"Product": ["Widget A", "Widget B"], "sales": [1, 2], "profit": [2, 4]}
    )

    with pytest.raises(SafeExecutionError, match="At least two matching rows"):
        SafeExecutor().execute(
            dataframe,
            "corr",
            columns=["sales", "profit"],
            filters={"Product": "Widget A"},
        )


def test_executor_recommends_eligible_pairs_within_product_filter():
    dataframe = pd.DataFrame(
        {
            "Product": ["Widget A", "Widget A", "Widget A", "Widget B", "Widget B"],
            "sales": [1, 2, 3, 10, 20],
            "profit": [2, 4, 6, 20, 40],
            "inventory": [None, 5, 6, 1, 2],
        }
    )

    recommendations = SafeExecutor().recommend_correlations(
        dataframe, {"Product": "Widget A"}
    )

    assert recommendations == [
        {
            "column 1": "sales",
            "column 2": "profit",
            "complete rows": 3,
            "correlation": 1.0,
        },
        {
            "column 1": "sales",
            "column 2": "inventory",
            "complete rows": 2,
            "correlation": 1.0,
        },
        {
            "column 1": "profit",
            "column 2": "inventory",
            "complete rows": 2,
            "correlation": 1.0,
        },
    ]
