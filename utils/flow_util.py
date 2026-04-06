from utils.info_util import format_output

def get_flow_directory(app, directory_path):
    """Get a directory object given its path. if the directory does not exist, create it.

    Args:
        app (App class): Ikigai app instance.
        directory_path (str): Path of the directory.

    Returns:
        Directory: Directory object if found, else None.
    """
    if len(directory_path) == 0:
        return None
    directories = app.flow_directories()
    if directory_path[0] not in directories:
        # Directory not found, create it
        app.flow_directory.new(name=directory_path[0]).build()

    current = app.flow_directories()[directory_path[0]]
    for directory in directory_path[1:]:
        children = current.directories()
        if directory not in children:
            # Directory not found, create it
            app.flow_directory.new(name=directory).parent(current).build()
        current = current.directories()[directory]
    return current


def run_flow(app, flow_name, flow_def, output_names, high_volume=False):
    """Shared helper: build/update flow, run it, and return output DataFrames."""
    flow = build_or_update_flow(app, flow_name, flow_def)
    flow.update_high_volume_preference(high_volume)
    run_log = flow.run()
    if run_log.status != "SUCCESS":
        raise Exception(
            f"Flow {flow_name} failed. Exiting. exception: {run_log.data}"
        )
    return format_output(app, output_names)




def set_aicast_mode(app, flow_name: str, mode: str):
    """
    Set the mode of an AiCast flow.

    Args:
        app: Ikigai app instance
        flow_name: Full flow name (e.g., "4. AiCast - (High) - amr direct")
        mode: Either "train" or "retrain_inference"

    Returns:
        bool: True if successful, False otherwise
    """
    if mode not in ("train", "retrain_inference"):
        print(f"Error: mode must be 'train' or 'retrain_inference', got '{mode}'")
        return False

    print(f"Getting flow: {flow_name}")
    flow = app.flows[flow_name]

    print(f"Getting flow definition...")
    flow_def = flow.describe().get("definition", {})

    # Find and update AiCast facets
    updated = 0
    for facet in flow_def.get("facets", []):
        args = facet.get("arguments", {})
        if "model_name" in args:
            old_mode = args.get("parameters", {}).get("mode", "N/A")
            if "parameters" not in args:
                args["parameters"] = {}
            args["parameters"]["mode"] = mode
            print(
                f"  Facet '{facet.get('name', facet.get('facet_id'))}': {old_mode} -> {mode}"
            )
            updated += 1

    if updated == 0:
        print("No AiCast facets found in this flow.")
        return False

    print(f"Updating flow definition...")
    flow.update_definition(flow_def)

    print(f"Done! Updated {updated} facet(s) to mode='{mode}'")
    return True


def set_mode_batch(app, chunks: list, flow_types: list, mode: str):
    """
    Set mode for multiple flows at once.

    Args:
        app: Ikigai app instance
        chunks: List of chunks (e.g., ["amr direct", "apac direct"])
        flow_types: List of "high" and/or "low"
        mode: Either "train" or "retrain_inference"
    """
    flow_patterns = {
        "high": "4. AiCast - (High) - {chunk}",
        "low": "5. AiCast - (Low) - {chunk}",
    }

    for chunk in chunks:
        for flow_type in flow_types:
            flow_name = flow_patterns[flow_type].format(chunk=chunk)
            print(f"\n--- {flow_name} ---")
            try:
                flow = app.flows[flow_name]
                flow_def = flow.describe().get("definition", {})

                for facet in flow_def.get("facets", []):
                    args = facet.get("arguments", {})
                    if "model_name" in args:
                        if "parameters" not in args:
                            args["parameters"] = {}
                        args["parameters"]["mode"] = mode

                flow.update_definition(flow_def)
                print(f"  Set to {mode}")
            except Exception as e:
                print(f"  Error: {e}")

    print("\nDone!")

def build_or_update_flow(app, flow_name, flow_def):
    """
    Build a new flow or update an existing one with the given definition.

    Args:
        app: Ikigai app instance
        flow_name: Name of the flow to build or update
        flow_def: Flow definition to use

    Returns:
        Flow object that was built or updated
    """
    exists = app.flows.search(flow_name).get(flow_name) is not None
    if not exists:
        flow = app.flow.new(name=flow_name).definition(flow_def)
        flow.build()
    else:
        flow = app.flows[flow_name]
        flow.update_definition(flow_def)
    return app.flows[flow_name]

