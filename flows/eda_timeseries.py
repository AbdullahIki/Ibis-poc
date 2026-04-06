from utils.dataset_util import get_dataset_directory


def eda_timeseries_flow(ikigai, app, config):
    """Build a time-series EDA flow for ACF analysis.

    The EDA_TIME_SERIES facet requires time_column, identifier_column,
    and value_column as facet arguments. It produces an ACF output.

    Flow structure:
        IMPORTED ──► EDA_TIME_SERIES ──acf──► EXPORTED
    """
    flow_builder = ikigai.builder
    facet_types = ikigai.facet_types

    args = config["arguments"]
    dataset_name = args["dataset_name"]
    dataset = app.datasets[dataset_name]

    import_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name=dataset_name
    ).arguments(
        dataset_id=dataset.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )

    eda_ts_facet = flow_builder.facet(
        facet_type=facet_types.MID.EDA_TIME_SERIES, name="EDA Time Series"
    ).arguments(
        time_column=args["time_col"],
        identifier_column=args["identifier_col"],
        value_column=args["value_col"],
    ).add_arrow(import_facet)

    export_directory = get_dataset_directory(app, args.get("export_directory", []))
    dir_id = export_directory.directory_id if export_directory else ""

    acf_name = args["acf_name"]
    flow_builder.facet(
        facet_type=facet_types.OUTPUT.EXPORTED, name=acf_name
    ).arguments(
        dataset_name=acf_name,
        file_type="csv",
        header=True,
        directory=dir_id,
    ).add_arrow(eda_ts_facet, output_type="acf")

    return flow_builder.build()
