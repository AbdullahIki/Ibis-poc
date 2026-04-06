from utils.custom_facets_utils import get_custom_facet_by_name
from utils.dataset_util import get_dataset_directory, create_dummy_data
from utils.model_util import ensure_model


def pre_actual_with_buffer_flow(ikigai, app, config, headers):
        """
        Build the **preprocess** reconciliation flow:
          1. Import k forecast datasets
          2. ``prep_buffer_cf`` → actuals with buffers + new_id_order
          3. Export the actuals with buffers    
        config["arguments"] should contain:
            - forecast_dataset_names : list[str]
            - identifier_columns     : list[str]
            - time_column            : str
            - value_column           : str
            - export_dataset_name    : str
            - export_directory       : list[str]  (optional)
        """
        flow_builder = ikigai.builder
        facet_types = ikigai.facet_types

        args = config["arguments"]

        # ── 1. Import facets — one per forecast dataset ──────────────────────
        forecast_names = args["forecast_dataset_names"]  # list of str
        import_facets = []
        for i, ds_name in enumerate(forecast_names):
            create_dummy_data(app, ds_name)
            ds = app.datasets[ds_name]
            import_facet = flow_builder.facet(
                facet_type=facet_types.INPUT.IMPORTED, name=f"forecast_{i}"
            ).arguments(
                dataset_id=ds.dataset_id,
                file_type="csv",
                header=True,
                use_raw_file=False,
            )
            import_facets.append(import_facet)

        # ── 2. Custom facet — prepare_actuals_with_buffers ───────────────────
        prep_cf_id = get_custom_facet_by_name(
            ikigai, headers, "prep_buffer_cf"
        )["custom_facet_id"]

        prep_cf_facet = flow_builder.facet(
            facet_type=facet_types.MID.Custom_facet, name="Prepare Buffers"
        )
        for i, imp_facet in enumerate(import_facets):
            prep_cf_facet = prep_cf_facet.add_arrow(imp_facet, name=f"forecast_{i}")

        prep_cf_facet = (
            prep_cf_facet
            .arguments(custom_facet_id=prep_cf_id)
            ._update_arguments(
                arguments=[
                    {
                        "name": "identifier_columns",
                        "type": "str",
                        "value": ",".join(args["identifier_columns"]),
                    },
                ]
            )
        )

        # ── 3. Export facet ──────────────────────────────────────────────────
        export_directory = get_dataset_directory(
            app, args.get("export_directory", [])
        )
        export_name = args["export_dataset_name"]

        export_facet_1 = flow_builder.facet(
            facet_type=facet_types.OUTPUT.EXPORTED, name=export_name
        ).arguments(
            dataset_name=export_name,
            file_type="csv",
            header=True,
            directory=export_directory.directory_id if export_directory else "",
        ).add_arrow(prep_cf_facet, name="actuals")

        export_facet_2 = flow_builder.facet(
            facet_type=facet_types.OUTPUT.EXPORTED, name="id_order"
        ).arguments(
            dataset_name=export_name + "_id_order",
            file_type="csv",
            header=True,
            directory=export_directory.directory_id if export_directory else "",
        ).add_arrow(prep_cf_facet, name="new_id_order")

        return flow_builder.build()

def reconciliation_preprocess_flow(ikigai, app, config, headers):
    """
    Build the **preprocess** reconciliation flow:
      1. Import the actuals-with-buffers dataset (output of pre_actual_with_buffer_flow)
      2. Reconciliation model facet (mode = "preprocess")
      3. Import the new_id_order dataset (output of pre_actual_with_buffer_flow)
      4. Second set of imports for the original forecast datasets
      5. ``add_identifier_cf`` → merged forecasts with identifier column
      6. Export the merged forecast dataset

    config["arguments"] should contain:
        - actuals_dataset_name   : str  — name of the actuals+buffers dataset
        - id_order_dataset_name  : str  — name of the new_id_order dataset
        - forecast_dataset_names : list[str]
        - identifier_columns     : list[str]
        - new_id_order           : str  — comma-separated identifier columns
                                          (with buffers) for the recon facet
        - model_name             : str
        - time_column            : str
        - value_column           : str
        - export_dataset_name    : str
        - export_directory       : list[str]  (optional)
        - return_all_levels      : bool (default False)
        - nonnegative            : bool (default True)
        - fill_missing_values    : str  (default "zero")
        - drop_threshold         : float (default 1.0)
        - interval_to_predict    : int  (default 10)
    """
    flow_builder = ikigai.builder
    facet_types = ikigai.facet_types
    model_types = ikigai.model_types

    args = config["arguments"]

    model_type = model_types.Reconciliation.spatio_hierarchical
    model_name = args.get("model_name", "reconciliation_0")
    ensure_model(app, model_name, model_type)

    # ── 1. Import the actuals-with-buffers dataset ───────────────────────
    actuals_name = args["actuals_dataset_name"]
    create_dummy_data(app, actuals_name)
    actuals_ds = app.datasets[actuals_name]

    actuals_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name="actuals_with_buffers"
    ).arguments(
        dataset_id=actuals_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )

    # ── 2. Reconciliation model facet (preprocess) ───────────────────────
    reconciliation_facet = (
        flow_builder.model_facet(
            name=model_name,
            facet_type=facet_types.MID.TIME_SERIES_ANALYSIS,
            model_type=model_type,
        )
        .add_arrow(actuals_facet)
        .arguments(
            model_name=model_name,
            version="Ver 1",
            overwrite=True,
        )
        .parameters(
            identifier_columns=args["new_id_order"],
            time_column=args["time_column"],
            value_column=args["value_column"],
            mode=args.get("mode", "preprocess"),
        )
        .hyperparameters(
            return_all_levels=args.get("return_all_levels", False),
            nonnegative=args.get("nonnegative", True),
            fill_missing_values=args.get("fill_missing_values", "zero"),
            drop_threshold=args.get("drop_threshold", 1.0),
            interval_to_predict=args.get("interval_to_predict", 10),
        )
    )

    # ── 3. Import the new_id_order dataset ───────────────────────────────
    id_order_name = args["id_order_dataset_name"]
    create_dummy_data(app, id_order_name)
    id_order_ds = app.datasets[id_order_name]

    id_order_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name="new_id_order"
    ).arguments(
        dataset_id=id_order_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )

    # ── 4. Import facets for the original forecast datasets ──────────────
    forecast_names = args["forecast_dataset_names"]
    forecast_import_facets = []
    for i, ds_name in enumerate(forecast_names):
        ds = app.datasets[ds_name]
        imp = flow_builder.facet(
            facet_type=facet_types.INPUT.IMPORTED, name=f"forecast_{i}"
        ).arguments(
            dataset_id=ds.dataset_id,
            file_type="csv",
            header=True,
            use_raw_file=False,
        )
        forecast_import_facets.append(imp)

    # ── 5. Custom facet — add_identifier_column ──────────────────────────
    add_id_cf_id = get_custom_facet_by_name(
        ikigai, headers, "add_identifier_cf"
    )["custom_facet_id"]

    add_id_cf_facet = flow_builder.facet(
        facet_type=facet_types.MID.Custom_facet, name="Add Identifier"
    )
    add_id_cf_facet = add_id_cf_facet.add_arrow(
        reconciliation_facet, name="recon_pre"
    )
    add_id_cf_facet = add_id_cf_facet.add_arrow(
        id_order_facet, name="new_id_order"
    )
    for i, imp in enumerate(forecast_import_facets):
        add_id_cf_facet = add_id_cf_facet.add_arrow(
            imp, name=f"forecast_{i}"
        )

    add_id_cf_facet = (
        add_id_cf_facet
        .arguments(custom_facet_id=add_id_cf_id)
        ._update_arguments(
            arguments=[
                {
                    "name": "identifier_columns",
                    "type": "str",
                    "value": ",".join(args["identifier_columns"]),
                },
            ]
        )
    )

    # ── 6. Export facet ──────────────────────────────────────────────────
    export_directory = get_dataset_directory(
        app, args.get("export_directory", [])
    )
    export_name = args["export_dataset_name"]

    add_id_cf_facet.facet(
        facet_type=facet_types.OUTPUT.EXPORTED, name=export_name
    ).arguments(
        dataset_name=export_name,
        file_type="csv",
        header=True,
        directory=export_directory.directory_id if export_directory else "",
    )

    return flow_builder.build()


def reconciliation_reconcile_flow(ikigai, app, config):
    """
    Build the **reconcile** flow:
      1. Import the merged-forecast dataset (output of preprocess flow)
      2. Reconciliation model facet (mode = "reconcile")
      3. Export the reconciled result

    config["arguments"] should contain:
        - merged_dataset_name  : str  — name of the dataset exported by
                                        reconciliation_preprocess_flow
        - identifier_columns   : list[str]
        - model_name           : str
        - time_column          : str
        - value_column         : str
        - export_dataset_name  : str
        - export_directory     : list[str]  (optional)
        - return_all_levels    : bool (default False)
        - nonnegative          : bool (default True)
        - fill_missing_values  : str  (default "zero")
        - drop_threshold       : float (default 1.0)
        - interval_to_predict  : int  (default 10)
    """
    flow_builder = ikigai.builder
    facet_types = ikigai.facet_types
    model_types = ikigai.model_types

    args = config["arguments"]

    model_type = model_types.Reconciliation.spatio_hierarchical
    model_name = args.get("model_name", "reconciliation_0")
    ensure_model(app, model_name, model_type)

    # ── 1. Import the merged forecast dataset ────────────────────────────
    merged_name = args["merged_dataset_name"]
    create_dummy_data(app, merged_name)
    merged_ds = app.datasets[merged_name]

    imported_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name="merged_forecast"
    ).arguments(
        dataset_id=merged_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )

    # ── 2. Reconciliation model facet (reconcile) ────────────────────────
    reconciliation_facet = (
        flow_builder.model_facet(
            name=model_name,
            facet_type=facet_types.MID.TIME_SERIES_ANALYSIS,
            model_type=model_type,
        )
        .add_arrow(imported_facet)
        .arguments(
            model_name=model_name,
            version="Ver 1",
            overwrite=False,
        )
        .parameters(
            identifier_columns=["identifier"],
            time_column=args["time_column"],
            value_column=args["value_column"],
            mode="reconcile",
        )
        .hyperparameters(
            return_all_levels=args.get("return_all_levels", False),
            nonnegative=args.get("nonnegative", True),
            fill_missing_values=args.get("fill_missing_values", "zero"),
            drop_threshold=args.get("drop_threshold", 1.0),
            interval_to_predict=args.get("interval_to_predict", 10),
        )
    )

    # ── 3. Export facet ──────────────────────────────────────────────────
    export_directory = get_dataset_directory(
        app, args.get("export_directory", [])
    )
    export_name = args["export_dataset_name"]

    reconciliation_facet.facet(
        facet_type=facet_types.OUTPUT.EXPORTED, name=export_name
    ).arguments(
        dataset_name=export_name,
        file_type="csv",
        header=True,
        directory=export_directory.directory_id if export_directory else "",
    )

    return flow_builder.build()
