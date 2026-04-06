import pandas as pd


def upload_dataset(app, name: str, df: pd.DataFrame):
    """Upload *df* as dataset *name*, creating or updating as needed.

    Returns the Ikigai Dataset object.
    """
    exists = app.datasets.search(name).get(name) is not None
    if not exists:
        dataset = app.dataset.new(name=name).df(df).build()
    else:
        app.datasets[name].edit_data(data=df)
        dataset = app.datasets[name]
    return dataset


def create_dummy_data(app, dataset_name, directory=None):
    """Create dummy data.

    Args:
        app (App class): Ikigai app instance.
    """
    try:
        dataset = app.datasets[dataset_name]
        # Dataset already exists -- no need to create dummy data.
        return
    except Exception:
        print(f"Creating dummy dataset: {dataset_name}")
        pass
    # create dummy data with one columns "a" and value 0
    data = pd.DataFrame({"a": [0]})
    directory = app.dataset_directories[directory] if directory else None
    if directory is not None:
        app.dataset.new(dataset_name).df(data).directory(directory).build()
    else:
        app.dataset.new(dataset_name).df(data).build()


def get_dataset_directory(app, directory_path):
    """Get a directory object given its path. if the directory does not exist, create it.

    Args:
        app (App class): Ikigai app instance.
        directory_path (list): Path of the directory.

    Returns:
        Directory: Directory object if found, else None.
    """
    if len(directory_path) == 0:
        return None
    directories = app.dataset_directories()
    if directory_path[0] not in directories:
        # Directory not found, create it
        app.dataset_directory.new(name=directory_path[0]).build()

    current = app.dataset_directories()[directory_path[0]]
    for directory in directory_path[1:]:
        children = current.directories()
        if directory not in children:
            # Directory not found, create it
            app.dataset_directory.new(name=directory).parent(current).build()
        current = current.directories()[directory]
    return current


def copy_dataset(
    source_project,
    dest_project,
    dataset_name,
    dest_directory_path=None,
    overwrite=False,
):
    """Copy a dataset from source project to destination project.

    Args:
        source_project: Source Ikigai project/app instance.
        dest_project: Destination Ikigai project/app instance.
        dataset_name (str): Name of the dataset to copy.
        dest_directory_path (str or list, optional): Destination directory path (e.g., 'Forecasts/batches' or ['Forecasts', 'batches']). If None, dataset is created at root.
        overwrite (bool): Whether to overwrite if dataset exists in destination.

    Returns:
        tuple: (success: bool, message: str)
    """
    # Get or create destination directory (if specified)
    dest_dir = None
    if dest_directory_path:
        if isinstance(dest_directory_path, str):
            dest_directory_path = dest_directory_path.split("/")

        dest_dir = get_dataset_directory(dest_project, dest_directory_path)
        if dest_dir is None:
            return False, "Could not create destination directory"

    # Check if already exists in dest
    if not overwrite:
        try:
            dest_project.datasets[dataset_name]
            return False, "Already exists"
        except (KeyError, Exception):
            pass  # Dataset doesn't exist, continue

    # Get from source
    try:
        df = source_project.datasets[dataset_name].df()
    except (KeyError, Exception) as e:
        return False, f"Not found in source: {type(e).__name__}"

    # Delete if overwrite
    if overwrite:
        try:
            dest_project.datasets[dataset_name].delete()
        except (KeyError, Exception):
            pass  # Doesn't exist, continue

    # Copy to dest
    try:
        builder = dest_project.dataset.new(dataset_name).df(df)
        if dest_dir:
            builder = builder.directory(dest_dir)
        builder.build()
        return True, f"Copied ({len(df)} rows)"
    except Exception as e:
        return False, f"Upload failed: {str(e)}"
