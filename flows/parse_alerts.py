from utils.dataset_util import get_dataset_directory


PARSE_ALERTS_SCRIPT_TEMPLATE = """
import re
import pandas as pd

target_column = '{target_column}'
missing_threshold = {missing_threshold}

columns_to_drop = []
reasons = []

for idx in range(len(data)):
    col_name = str(data.iloc[idx]['Column Name'])
    alert_text = str(data.iloc[idx]['Alert type'])

    if col_name == target_column:
        continue

    missing_match = re.search(r'\\((\\d+\\.?\\d*)%\\)\\s*missing values', alert_text)
    if missing_match:
        pct = float(missing_match.group(1))
        if pct > missing_threshold:
            columns_to_drop.append(col_name)
            reasons.append(str(pct) + '% missing values')
            continue

    if 'unique' in alert_text.lower():
        columns_to_drop.append(col_name)
        reasons.append('Unique values (identifier)')
        continue

result = pd.DataFrame({{
    'column_name': columns_to_drop,
    'reason': reasons,
    'action': ['drop'] * len(columns_to_drop),
}})
"""


def parse_alerts_flow(ikigai, app, config):
    """Build a flow that uses the Python facet to parse EDA alerts.

    The Python script reads the alerts DataFrame, identifies columns to
    drop based on missing-value thresholds and uniqueness, and outputs a
    structured DataFrame listing each column, the reason, and the action.

    Flow structure:
        IMPORTED (alerts) --name="alerts"--> PYTHON --name="result"--> EXPORTED
    """
    flow_builder = ikigai.builder
    facet_types = ikigai.facet_types

    args = config["arguments"]
    alerts_dataset_name = args["alerts_dataset_name"]
    alerts_ds = app.datasets[alerts_dataset_name]

    target_column = args.get("target_column", "Survived")
    missing_threshold = args.get("missing_threshold", 19.0)

    import_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name=alerts_dataset_name
    ).arguments(
        dataset_id=alerts_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )

    script = PARSE_ALERTS_SCRIPT_TEMPLATE.format(
        target_column=target_column,
        missing_threshold=missing_threshold,
    )

    python_facet = flow_builder.facet(
        facet_type=facet_types.MID.PYTHON, name="Parse Alerts"
    ).arguments(
        script=script,
    ).add_arrow(import_facet, name="alerts")

    export_directory = get_dataset_directory(app, args.get("export_directory", []))
    dir_id = export_directory.directory_id if export_directory else ""

    export_name = args["export_name"]
    flow_builder.facet(
        facet_type=facet_types.OUTPUT.EXPORTED, name=export_name
    ).arguments(
        dataset_name=export_name,
        file_type="csv",
        header=True,
        directory=dir_id,
    ).add_arrow(python_facet, name="result")

    return flow_builder.build()
