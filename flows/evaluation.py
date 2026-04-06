from utils.custom_facets_utils import get_custom_facet_by_name
from utils.dataset_util import create_dummy_data, get_dataset_directory


def evaluation_flow(ikigai, app, config, headers):
    flow_builder = ikigai.builder
    facet_types = ikigai.facet_types

    # Get dataset names from config
    primary_name = config["arguments"]["primary_dataset_name"]
    forecast_name = config["arguments"]["forecast_dataset_name"]

    # Ensure datasets exist
    create_dummy_data(app, primary_name)
    create_dummy_data(app, forecast_name)
    primary_ds = app.datasets[primary_name]
    forecast_ds = app.datasets[forecast_name]

    # Import primary (reals) dataset
    primary_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name="primary_dataset"
    ).arguments(
        dataset_id=primary_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )

    # Import forecast dataset
    forecast_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name="forecast_dataset"
    ).arguments(
        dataset_id=forecast_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )

    # Get evaluate_forecasts custom facet
    eval_cf = get_custom_facet_by_name(
        ikigai, headers, "evaluate_forecasts"
    )["custom_facet_id"]

    # Apply custom facet with two inputs: forecast and real
    eval_cf_facet = (
        flow_builder.facet(
            facet_type=facet_types.MID.Custom_facet, name="Evaluate Forecasts"
        )
        .add_arrow(forecast_facet, name="forecast")
        .add_arrow(primary_facet, name="real")
        .arguments(custom_facet_id=eval_cf)
        ._update_arguments(
            arguments=[
                {
                    "name": "time_column",
                    "type": "str",
                    "value": config["arguments"]["time_col"],
                },
                {
                    "name": "value_column",
                    "type": "str",
                    "value": config["arguments"]["value_col"],
                },
                {
                    "name": "identifier_columns",
                    "type": "str",
                    "value": ",".join(config["arguments"]["identifier_cols"]),
                },
                {
                    "name": "levels",
                    "type": "str",
                    "value": config["arguments"].get("levels", ""),
                },
                {
                    "name": "aggregate_errors",
                    "type": "bool",
                    "value": config["arguments"].get("aggregate_errors", True),
                },
                {
                    "name": "error_window",
                    "type": "int",
                    "value": config["arguments"].get("error_window", 1),
                },
            ]
        )
    )

    # Export evaluation result
    export_directory = get_dataset_directory(
        app, config["arguments"]["export_directory"]
    )
    export_name = config["arguments"]["export_dataset_name"]
    export_facet = (
        flow_builder.facet(facet_type=facet_types.OUTPUT.EXPORTED, name=export_name)
        .add_arrow(eval_cf_facet)
        .arguments(
            dataset_name=export_name,
            file_type="csv",
            header=True,
            directory=export_directory.directory_id if export_directory else "",
        )
    )

    return flow_builder.build()
