from utils.dataset_util import get_dataset_directory
from utils.model_util import ensure_model, model_has_version


def ai_match_flow(ikigai, app, config):
    """Build an aiMatch (supervised) flow for entity matching.

    The flow connects two imported datasets (left and right) to the
    aiMatch model facet and exports two outputs: the matching result
    and the column mapping.

    Flow structure:
        IMPORTED (left)  ──left──►  AI_MATCH  ──matching_result──►  EXPORTED
        IMPORTED (right) ──right──►            ──column_mapping───►  EXPORTED
    """
    flow_builder = ikigai.builder
    facet_types = ikigai.facet_types
    model_types = ikigai.model_types

    args = config["arguments"]
    model_type = model_types.AI_MATCH.Supervised
    model_name = args.get("model_name", "ai_match")

    if args.get("overwrite_model", False):
        try:
            app.models[model_name].delete()
        except Exception:
            pass

    ensure_model(app, model_name, model_type)

    left_name = args["left_dataset_name"]
    right_name = args["right_dataset_name"]
    left_ds = app.datasets[left_name]
    right_ds = app.datasets[right_name]

    # --- Import facets ---
    left_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name=left_name
    ).arguments(
        dataset_id=left_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )

    right_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name=right_name
    ).arguments(
        dataset_id=right_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )

    # --- aiMatch model facet ---
    model_fct = (
        flow_builder.model_facet(
            facet_type=facet_types.MID.AI_MATCH,
            model_type=model_type,
            name=model_name,
        )
        .add_arrow(left_facet, table_identifier="left")
        .add_arrow(right_facet, table_identifier="right")
        .arguments(
            model_name=model_name,
            version="",
        )
        .hyperparameters(
            matching_threshold=args.get("matching_threshold", 0.5),
            include_similarity=args.get("include_similarity", False),
            column_mapping=args.get("column_mapping", True),
            sampling_amount=args.get("sampling_amount", 0.1),
            column_mapping_threshold=args.get("column_mapping_threshold", 0.5),
            left_include=args.get("left_include", []),
            right_include=args.get("right_include", []),
            left_exclude=args.get("left_exclude", []),
            right_exclude=args.get("right_exclude", []),
        )
    )

    # --- Output facets ---
    export_directory = get_dataset_directory(app, args.get("export_directory", []))
    dir_id = export_directory.directory_id if export_directory else ""

    output_map = [
        ("matching_result", args["matching_result_name"]),
        ("column_mapping", args["column_mapping_name"]),
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
