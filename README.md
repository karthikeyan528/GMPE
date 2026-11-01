# AgentIQ

Small Python foundation for evidence-based data analysis, with a local browser app for ingestion, profiling, planning, and restricted execution. There is no command-line analysis interface.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

For subsequent sessions, activate the environment again with:

```bash
source .venv/bin/activate
```

## Check That It Works

Run the test suite:

```bash
python -m pytest -q
```

A healthy run currently reports three passing tests. You can also run a direct smoke check:

```bash
python - <<'PY'
import pandas as pd

from agentiq.execution import SafeExecutor
from agentiq.planning import PlannerAgent
from agentiq.profiling import DatasetProfiler

dataframe = pd.DataFrame({"sales": [10, 20], "customers": [2, 4]})
print(DatasetProfiler().profile(dataframe)["shape"])
print(PlannerAgent().build_plan("show sales correlation", list(dataframe.columns)))
print(SafeExecutor().execute(dataframe, "corr"))
PY
```

The smoke check should print `(2, 2)`, an analysis plan, and a correlation result. The tests are the primary correctness check.

## Browser App

Install the browser app dependencies and start the local app:

```bash
python -m pip install -e '.[web]'
python -m streamlit run agentiq/app.py
```

Open the local URL printed by Streamlit and upload a CSV or Excel dataset. To analyze two numeric columns for one product, enter a question such as `analyze sales and profit for product Widget A`, replacing the column names and product value with ones from your file. The app shows which columns and product it recognized, then displays summary statistics and correlation for only that product's rows when you select **Run analysis**. Planning is currently rule-based and does not use a language model.

## Layout

- `agentiq/ingestion`: CSV and Excel loading plus validation
- `agentiq/profiling`: dataset summaries
- `agentiq/planning`: deterministic analysis plans
- `agentiq/execution`: allow-listed analysis operations
- `agentiq/app.py`: local Streamlit browser app
- `tests`: focused behavior tests
