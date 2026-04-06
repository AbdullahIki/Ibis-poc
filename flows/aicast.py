from utils.dataset_util import get_dataset_directory, create_dummy_data
from utils.model_util import ensure_model


def aicast_flow(ikigai, app, config):

    flow_builder = ikigai.builder
    facet_types = ikigai.facet_types
    model_types = ikigai.model_types

    model_type=model_types.AI_CAST.base
    model_name = config["arguments"].get("model_name", "ai_cast_0")
    ensure_model(app, model_name, model_type)

    aux_name = config["arguments"]["aux_dataset"]
    primary_name = config["arguments"]["primary_dataset"]
    export_name = config["arguments"]["export_dataset_name"]

    # ensure datasets exist in the app
    create_dummy_data(app, aux_name)
    create_dummy_data(app, primary_name)

    aux_ds = app.datasets[aux_name]
    primary_ds = app.datasets[primary_name]

    aux_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name="aux"
    ).arguments(
        dataset_id=aux_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )


    primary_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name="primary"
    ).arguments(
        dataset_id=primary_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )

    model_facet = (
            flow_builder.model_facet(
                facet_type=facet_types.MID.AI_CAST,
                model_type=model_type,
                name=model_name,
            )
            # inputs with input_type (matches JSON arrows)
            .add_arrow(aux_facet, input_type="auxiliary")
            .add_arrow(primary_facet, input_type="primary")
            # arguments straight from JSON
            .arguments(
                model_name=model_name,
            ).hyperparameters(
                    type =  config["arguments"].get("type", "base"),
                    hierarchical_type =config["arguments"].get("hierarchical_type", "bottom_up"),
                    fill_missing_values = "zero",
                    drop_threshold = 1,
                    include_reals = True,
                    nonnegative = True,
                    return_all_levels = False,
                    enable_artifacts_saving= False,
                    holdout_period= config["arguments"].get("holdout_period", 10),
                    interval_to_predict= config["arguments"].get("interval_to_predict", 10),
                    metric= config["arguments"].get("metric", "weighted_mean_absolute_percentage_error"),
                    recalculate_metrics= False,
                    best_model_only= True,
                    computation_budget= 100,
                    confidence= 0.7,
                    enable_conformal_interval= False,
                    enable_parallel_processing= False,
                    eval_method= config["arguments"].get("eval_method", "cv"),
                    models_to_include= config["arguments"].get("models_to_include", ["Additive",
            "Sma", "Croston", "Lgmt1", "Tsfm0"]),
                    time_budget= config["arguments"].get("time_budget", 120),
                    enable_guardrails  = True, 
                    guardrail_type = "outlier_correction",
                    pruner= "statistical",
                    length_short_threshold = config["arguments"].get("interval_to_predict", 10) *4,

            )
            .parameters(
                identifier_columns=config["arguments"]["identifier_cols"],
                time_column=config["arguments"]["time_col"],
                value_column=config["arguments"]["value_col"],
                mode=config["arguments"].get("mode", "retrain_inference"),
            )
        )



    export_directory = get_dataset_directory(
        app, config["arguments"]["export_directory"]
    )
    export_name = config["arguments"]["export_dataset_name"]
    model_facet.facet(
        facet_type=facet_types.OUTPUT.EXPORTED, name=export_name
    ).arguments(
        dataset_name=export_name,
        file_type="csv",
        header=True,
        directory=export_directory.directory_id if export_directory else "",
    )

    return flow_builder.build()