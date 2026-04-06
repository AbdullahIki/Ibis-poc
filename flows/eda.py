from utils.dataset_util import get_dataset_directory


def eda_flow(ikigai, app, config):
    """Build a tabular EDA flow.

    The EDA facet takes a single imported dataset (no facet arguments)
    and produces 7 outputs: numeric_statistic, text_statistic, alert,
    correlation_tabular, correlation_heatmap, numeric_histogram,
    text_histogram.

    Flow structure:
        IMPORTED ──► EDA ──► 7x EXPORTED
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

    eda_facet = flow_builder.facet(
        facet_type=facet_types.MID.EDA, name="Exp Data Analysis"
    ).add_arrow(import_facet)

    export_directory = get_dataset_directory(app, args.get("export_directory", []))
    dir_id = export_directory.directory_id if export_directory else ""

    output_map = [
        ("numeric_statistic", args["numeric_statistic_name"]),
        ("text_statistic", args["text_statistic_name"]),
        ("alert", args["alert_name"]),
        ("correlation_tabular", args["correlation_tabular_name"]),
        ("correlation_heatmap", args["correlation_heatmap_name"]),
        ("numeric_histogram", args["numeric_histogram_name"]),
        ("text_histogram", args["text_histogram_name"]),
    ]

    for output_type, export_name in output_map:
        flow_builder.facet(
            facet_type=facet_types.OUTPUT.EXPORTED, name=export_name
        ).arguments(
            dataset_name=export_name,
            file_type="csv",
            header=True,
            directory=dir_id,
        ).add_arrow(eda_facet, output_type=output_type)

    return flow_builder.build()
