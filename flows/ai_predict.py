from utils.dataset_util import get_dataset_directory
from utils.model_util import ensure_model, model_has_version


def ai_predict_flow(ikigai, app, config):
    """Build an aiPredict flow for classification or regression.

    If the model already has a trained version the mode is automatically
    switched to ``inference`` so the platform does not reject a second
    ``train`` attempt.

    The flow structure is:
        IMPORTED -> AI_PREDICT -> 3x EXPORTED
                                   ├─ predictions
                                   └─ feature_importance_metrics
                                   └─ performance_metrics
    """
    flow_builder = ikigai.builder
    facet_types = ikigai.facet_types
    model_types = ikigai.model_types

    args = config["arguments"]
    requested_mode = args.get("mode", "train")
    model_type = model_types.AI_PREDICT.Base
    model_name = args.get("model_name", "ai_predict")
    if args.get("overwrite_model", False) and requested_mode == "train":
        try:
            app.models[model_name].delete()
        except:
            pass
    ensure_model(app, model_name, model_type)
    if requested_mode == "train" and model_has_version(app, model_name):
        mode = "inference"
        print(f"[ai_predict] Model '{model_name}' already trained — switching to inference mode")
    else:
        mode = requested_mode

    primary_name = args["primary_dataset_name"]
    primary_ds = app.datasets[primary_name]

    # --- Import facet ---
    import_kwargs = dict(
        dataset_id=primary_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )
    data_types = args.get("data_types")
    if data_types:
        import_kwargs["data_types"] = data_types

    primary_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name=primary_name
    ).arguments(**import_kwargs)

    # holdout primary dataset
    holdout_kwargs = dict(
        dataset_id=primary_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )
    if data_types:
        holdout_kwargs["data_types"] = data_types

    holdout_primary_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name=f"{primary_name}_holdout"
    ).arguments(**holdout_kwargs)

    # --- AI Predict model facet ---
    model_fct = primary_facet.model_facet(
        facet_type=facet_types.MID.AI_PREDICT,
        model_type=model_type,
    ).arguments(
        model_name=model_name,
        version="",
    ).parameters(
        target_column=args["target_column"],
        mode=mode,
    ).hyperparameters(
        optimization_metric=args["optimization_metric"],
        models_to_include=args.get("models_to_include", [
            "light_gradient_boosting_machine",
            "categorical_boosting",
            "random_forest",
        ]),
        n_splits=args.get("n_splits", 3),
        random_seed=args.get("random_seed", 1),
        time_budget=args.get("time_budget", 100),
        best_model_only=args.get("best_model_only", False),
        eval_method=args.get("eval_method", "cv"),
        preprocess=args.get("preprocess", True),
        drop_threshold=args.get("drop_threshold", 0.9),
        confidence_interval=args.get("confidence_interval", 0.95),
        explain_predictions=args.get("explain_predictions", False),
        include_reals=args.get("include_reals", True),
        fill_missing_values=args.get("fill_missing_values", {
            "CATEGORICAL": "ffill",
            "NUMERIC": "average",
            "TEXT": "ffill",
            "TIME": "ffill",
        }),
        feature_importance_metrics=args.get(
            "feature_importance_metrics", ["estimator_base"]
        ),
    ).add_arrow(holdout_primary_facet, input_type="holdout")

    # --- 3 output branches from the model facet ---
    export_directory = get_dataset_directory(app, args["export_directory"])
    dir_id = export_directory.directory_id if export_directory else ""

    output_map = [
        ("predictions", args["predictions_name"]),
        ("feature_importance_metrics", args["feature_importance_name"]),
        ("performance_metrics", args["performance_metrics_name"]),
    ]

    for output_type, dataset_name in output_map:
        flow_builder.facet(
            facet_type=facet_types.OUTPUT.EXPORTED, name=dataset_name
        ).arguments(
            dataset_name=dataset_name,
            file_type="csv",
            header=True,
            directory=dir_id,
        ).add_arrow(model_fct, output_type=output_type)

    return flow_builder.build()
