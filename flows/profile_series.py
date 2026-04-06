from utils.custom_facets_utils import get_custom_facet_by_name
from utils.dataset_util import create_dummy_data, get_dataset_directory


def profile_series_flow(ikigai, app, config, headers):
    flow_builder = ikigai.builder
    facet_types = ikigai.facet_types

    # Get dataset name from config
    primary_name = config["arguments"]["primary_dataset_name"]
    output_name = config["arguments"]["output_dataset_name"]

    # Ensure dataset exists
    create_dummy_data(app, primary_name)
    primary_ds = app.datasets[primary_name]

    # Import primary dataset
    primary_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name="primary_dataset"
    ).arguments(
        dataset_id=primary_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )

    # Get classify_and_profile_series custom facet
    profile_cf = get_custom_facet_by_name(
        ikigai, headers, "classify_and_profile_series"
    )["custom_facet_id"]

    # Apply custom facet
    profile_cf_facet = (
        flow_builder.facet(
            facet_type=facet_types.MID.Custom_facet, name="Classify and Profile Series"
        )
        .add_arrow(primary_facet, name="primary")
        .arguments(custom_facet_id=profile_cf)
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
                    "type": "list_str",
                    "value": config["arguments"]["identifier_cols"],
                },
                {
                    "name": "recency_history_periods",
                    "type": "int",
                    "value": config["arguments"]["recency_history_periods"],
                },
                {
                    "name": "seasonal_period",
                    "type": "int",
                    "value": config["arguments"]["seasonal_period"],
                },
            ]
        )
    )

    # Export result
    export_directory = get_dataset_directory(
        app, config["arguments"]["export_directory"]
    )
    export_facet = (
        flow_builder.facet(facet_type=facet_types.OUTPUT.EXPORTED, name=output_name)
        .add_arrow(profile_cf_facet)
        .arguments(
            dataset_name=output_name,
            file_type="csv",
            header=True,
            directory=export_directory.directory_id if export_directory else "",
        )
    )

    return flow_builder.build()
