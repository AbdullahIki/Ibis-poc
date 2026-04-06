from utils.dataset_util import get_dataset_directory
from utils.model_util import ensure_model


def decomposition_flow(ikigai, app, config):
    flow_builder = ikigai.builder
    facet_types = ikigai.facet_types
    model_types = ikigai.model_types

    args = config["arguments"]

    model_type = model_types.DECOMPOSITION.seasonaltrendloess
    model_name = args.get("model_name", "decomposition")
    ensure_model(app, model_name, model_type)

    primary_name = args["primary_dataset_name"]
    aux_name = args["aux_dataset_name"]
    primary_ds = app.datasets[primary_name]
    aux_ds = app.datasets[aux_name]

    primary_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name="primary_dataset"
    ).arguments(
        dataset_id=primary_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )

    aux_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name="aux_dataset"
    ).arguments(
        dataset_id=aux_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )

    aux_time_col = args.get("aux_time_col", "date")
    if aux_time_col != args["time_col"]:
        aux_source = aux_facet.facet(
            facet_type=facet_types.MID.RENAME_COLUMNS,
        ).arguments(
            columns=[
                {
                    "column_name": aux_time_col,
                    "new_column_name": args["time_col"],
                }
            ],
        )
    else:
        aux_source = aux_facet

    decomp_model = (
        flow_builder.model_facet(
            facet_type=facet_types.MID.TIME_SERIES_ANALYSIS,
            model_type=model_type,
        )
        .add_arrow(primary_facet, input_type="primary")
        .add_arrow(aux_source, input_type="auxiliary")
        .arguments(
            model_name=model_name,
            version="",
            overwrite=True,
        )
        .parameters(
            time_column=args["time_col"],
            value_column=args["value_col"],
            identifier_columns=args["identifier_cols"],
            period = args.get("period", 14),
        )
        .hyperparameters(
            covariate_effect=args.get("covariate_effect", "Additive"),
            fill_missing_value=args.get("fill_missing_value", "zero"),
            output_components=args.get(
                "output_components",
                ["Seasonality", "Trend", "Covariate Effect", "Reals"],
            ),
            seasonal_degree=args.get("seasonal_degree", 0),
            seasonal_smoother_length=args.get("seasonal_smoother_length", 7),
            alpha=args.get("alpha", 1),
            min_multiplicative_observation=args.get("min_multiplicative_observation", 1),
        )
    )

    export_directory = get_dataset_directory(app, args["export_directory"])
    export_name = args["export_dataset_name"]
    decomp_model.facet(
        facet_type=facet_types.OUTPUT.EXPORTED, name=export_name
    ).arguments(
        dataset_name=export_name,
        file_type="csv",
        header=True,
        directory=export_directory.directory_id if export_directory else "",
    )

    return flow_builder.build()
