from utils.dataset_util import get_dataset_directory
from utils.model_util import ensure_model


def cohorts_flow(ikigai, app, config):
    flow_builder = ikigai.builder
    facet_types = ikigai.facet_types
    model_types = ikigai.model_types

    args = config["arguments"]
    primary_name = args["primary_dataset_name"]
    primary_ds = app.datasets[primary_name]

    embedding_dim = args.get("embedding_dimension", 16)
    n_clusters = args.get("n_clusters", 4)

    embedding_model_type = model_types.EMBEDDING.Base
    embedding_model_name = args.get("embedding_model_name", "embeddings")
    ensure_model(app, embedding_model_name, embedding_model_type)

    clustering_model_type = model_types.CLUSTERING.Base
    clustering_model_name = args.get("clustering_model_name", "clustering")
    ensure_model(app, clustering_model_name, clustering_model_type)

    # --- Import ---
    primary_facet = flow_builder.facet(
        facet_type=facet_types.INPUT.IMPORTED, name="primary_dataset"
    ).arguments(
        dataset_id=primary_ds.dataset_id,
        file_type="csv",
        header=True,
        use_raw_file=False,
    )

    # --- Embedding model ---
    embedding_facet = primary_facet.model_facet(
        facet_type=facet_types.MID.TIME_SERIES_ANALYSIS,
        model_type=embedding_model_type,
    ).arguments(
        model_name=embedding_model_name,
        version="",
        overwrite=True,
    ).parameters(
        identifier_column=",".join(args["identifier_cols"]),
        identifier_columns=args["identifier_cols"],
        time_column=args["time_col"],
        value_column=args["value_col"],
    ).hyperparameters(
        embedding_dimension=embedding_dim,
        use_scaling=args.get("use_scaling", True),
        window_size=args.get("window_size", 0),
    )

    # --- Copy embedding output so it can branch ---
    copy_facet = embedding_facet.facet(facet_type=facet_types.MID.COPY)

    # --- Clustering model on one copy ---
    target_columns = [f"embedding_{i}" for i in range(embedding_dim)]
    clustering_facet = (
        copy_facet.model_facet(
            facet_type=facet_types.MID.PREDICT,
            model_type=clustering_model_type,
        )
        .arguments(
            model_name=clustering_model_name,
            version="",
            overwrite=True,
        )
        .parameters(
            target_columns=target_columns,
        )
        .hyperparameters(
            algorithm=args.get("algorithm", "auto"),
            n_clusters=n_clusters,
        )
    )

    # --- Join the other copy with clustering assignments ---
    join_facet = (
        flow_builder.facet(facet_type=facet_types.MID.JOIN)
        .add_arrow(copy_facet, table_side="left")
        .add_arrow(clustering_facet, table_side="right")
        .arguments(
            option="join_by_rows",
            ignore_missing_values=True,
        )
    )

    # --- Rename Cluster_ID -> cohort ---
    rename_facet = join_facet.facet(
        facet_type=facet_types.MID.RENAME_COLUMNS,
    ).arguments(
        columns=[{"column_name": "Cluster_ID", "new_column_name": "cohort"}],
    )

    # --- Export ---
    export_directory = get_dataset_directory(app, args["export_directory"])
    export_name = args["export_dataset_name"]
    rename_facet.facet(
        facet_type=facet_types.OUTPUT.EXPORTED, name=export_name
    ).arguments(
        dataset_name=export_name,
        file_type="csv",
        header=True,
        directory=export_directory.directory_id if export_directory else "",
    )

    return flow_builder.build()
