from utils.dataset_util import get_dataset_directory
from utils.model_util import ensure_model


def imputation_flow(ikigai, app, config):
    flow_builder = ikigai.builder
    facet_types = ikigai.facet_types
    model_types = ikigai.model_types

    model_type = model_types.IMPUTATION.Base
    model_name = config["arguments"].get("model_name", "imputation")
    ensure_model(app, model_name, model_type)

    primary_name = config["arguments"]["primary_dataset_name"]
    primary_ds = app.datasets[primary_name]

    primary_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name="primary_dataset"
    ).arguments(
        dataset_id=primary_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )

    model_fct = primary_facet.model_facet(
        facet_type=facet_types.MID.TIME_SERIES_ANALYSIS,
        model_type=model_type,
    ).arguments(
        model_name=model_name,
        version="",
        overwrite=True,
    ).parameters(
        time_column=config["arguments"]["time_col"],
        value_column=config["arguments"]["value_col"],
        identifier_columns=config["arguments"]["identifier_cols"],
    ).hyperparameters(
        rank=config["arguments"].get("rank", 10),
        rank_selection_method=config["arguments"].get("rank_selection_method", "donoho"),
        lag=config["arguments"].get("lag", 0),
        analysis_mode=config["arguments"].get("analysis_mode", "Multivariate"),
    )

    export_directory = get_dataset_directory(
        app, config["arguments"]["export_directory"]
    )
    export_name = config["arguments"]["export_dataset_name"]
    model_fct.facet(
        facet_type=facet_types.OUTPUT.EXPORTED, name=export_name
    ).arguments(
        dataset_name=export_name,
        file_type="csv",
        header=True,
        directory=export_directory.directory_id if export_directory else "",
    )

    return flow_builder.build()
