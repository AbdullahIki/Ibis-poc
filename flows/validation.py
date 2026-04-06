from utils.custom_facets_utils import get_custom_facet_by_name
from utils.dataset_util import create_dummy_data, get_dataset_directory
import yaml

def validation_flow(ikigai, app, config, headers):
        flow_builder = ikigai.builder
        facet_types = ikigai.facet_types

        # Get dataset names from config
        primary_name = config["arguments"]["primary_dataset_name"]
        aux_name = config["arguments"]["aux_dataset_name"]
        # business_rules_name = config["arguments"]["business_rules_name"]
        aux_name = config["arguments"]["aux_dataset_name"]
        meta_data_name = config["arguments"]["meta_data_name"]

        # Create dummy dataset
        create_dummy_data(app, primary_name)
        primary_ds = app.datasets[primary_name]
        aux_ds = app.datasets[aux_name]
        # business_rules_ds = app.datasets[business_rules_name]
        meta_data_ds = app.datasets[meta_data_name]

        # Import primary dataset
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
        meta_data_facet = flow_builder.facet(
            facet_type=facet_types.INPUT.IMPORTED, name="meta_data"
        ).arguments(
            dataset_id=meta_data_ds.dataset_id,
            file_type="csv",
            header=True,
            use_raw_file=False,
        )
        # Create YoY forecast using custom facet
        validation_cf = get_custom_facet_by_name(
            ikigai, headers, "Validate_time_series_datasets"
        )["custom_facet_id"]

        validation_cf_facet = (
            flow_builder.facet(
                facet_type=facet_types.MID.Custom_facet, name="Create YoY Forecast"
            )
            .add_arrow(primary_facet, name="primary")
            .add_arrow(aux_facet, name="aux")
            .add_arrow(meta_data_facet, name="meta_data")
            .arguments(custom_facet_id=validation_cf)._update_arguments(
                arguments=[
                    {
                        "name": "time_col",
                        "type": "str",
                        "value": config["arguments"]["time_col"],
                    },
                    {
                        "name": "value_col",
                        "type": "str",
                        "value": config["arguments"]["value_col"],
                    },
                    {
                        "name": "identifier_cols",
                        "type": "list_str",
                        "value": config["arguments"]["identifier_cols"],
                    },
            ])
        )

        # Export YoY forecast dataset
        export_directory = get_dataset_directory(
            app, config["arguments"]["export_directory"]
        )
        export_name = config["arguments"]["export_dataset_name"]
        export_facet = (
            flow_builder.facet(facet_type=facet_types.OUTPUT.EXPORTED, name=export_name)
            .add_arrow(validation_cf_facet)
            .arguments(
                dataset_name=export_name,
                file_type="csv",
                header=True,
                directory=export_directory.directory_id if export_directory else "",
            )
        )

        return flow_builder.build()

