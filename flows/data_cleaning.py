from utils.dataset_util import get_dataset_directory


def data_cleaning_flow(ikigai, app, config):
    """Build a flow that drops specified columns from a dataset.

    Uses the platform DROP_COLUMNS system facet to remove columns
    identified by the alert-parsing step.

    Flow structure:
        IMPORTED ---> DROP_COLUMNS(target_columns) ---> EXPORTED
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

    columns_to_drop = args["columns_to_drop"]

    drop_facet = flow_builder.facet(
        facet_type=facet_types.MID.DROPCOLUMNS, name="Drop Columns"
    ).arguments(
        target_columns=columns_to_drop,
    ).add_arrow(import_facet)

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
    ).add_arrow(drop_facet)

    return flow_builder.build()
