from utils.custom_facets_utils import get_custom_facet_by_name
from utils.dataset_util import create_dummy_data, get_dataset_directory


def explainability_flow(ikigai, app, config, headers):
    flow_builder = ikigai.builder
    facet_types = ikigai.facet_types

    # Get dataset names from config
    forecast_name = config["arguments"]["forecast_dataset_name"]
    actuals_name = config["arguments"]["actual_dataset_name"]

    # Ensure datasets exist
    create_dummy_data(app, forecast_name)
    create_dummy_data(app, actuals_name)
    forecast_ds = app.datasets[forecast_name]
    actuals_ds = app.datasets[actuals_name]

    # Import forecast dataset
    forecast_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name="forecast_dataset"
    ).arguments(
        dataset_id=forecast_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )

    # Import actuals dataset
    actuals_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name="actuals_dataset"
    ).arguments(
        dataset_id=actuals_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )

    # Get explainability_cf custom facet
    explain_cf = get_custom_facet_by_name(
        ikigai, headers, "explainability_cf"
    )["custom_facet_id"]

    # Apply custom facet with two inputs: forecast and actuals
    explain_cf_facet = (
        flow_builder.facet(
            facet_type=facet_types.MID.Custom_facet, name="Explainability"
        )
        .add_arrow(forecast_facet, name="forecast")
        .add_arrow(actuals_facet, name="actuals")
        .arguments(custom_facet_id=explain_cf)
        ._update_arguments(
            arguments=[
                {
                    "name": "time_col",
                    "type": "str",
                    "value": config["arguments"]["time_col"],
                },
                {
                    "name": "value_col",
                    "type": "str",
                    "value": config["arguments"]["value_col"],
                },
                {
                    "name": "identifier_columns",
                    "type": "str",
                    "value": ",".join(config["arguments"]["identifier_cols"]),
                },
                {
                    "name": "lags",
                    "type": "str",
                    "value": ",".join(
                        str(l) for l in config["arguments"].get("lags", [7, 14])
                    ),
                },
                {
                    "name": "production",
                    "type": "bool",
                    "value": config["arguments"].get("production", True),
                },
            ]
        )
    )

    # Export period-level result
    period_dir = get_dataset_directory(
        app, config["arguments"]["export_directory"]
    )
    period_name = config["arguments"]["export_period_name"]
    flow_builder.facet(
        facet_type=facet_types.OUTPUT.EXPORTED, name=period_name
    ).add_arrow(explain_cf_facet, name="period").arguments(
        dataset_name=period_name,
        file_type="csv",
        header=True,
        directory=period_dir.directory_id if period_dir else "",
    )

    # Export quarterly result
    quarterly_name = config["arguments"]["export_quarterly_name"]
    flow_builder.facet(
        facet_type=facet_types.OUTPUT.EXPORTED, name=quarterly_name
    ).add_arrow(explain_cf_facet, name="quarterly").arguments(
        dataset_name=quarterly_name,
        file_type="csv",
        header=True,
        directory=period_dir.directory_id if period_dir else "",
    )

    return flow_builder.build()
