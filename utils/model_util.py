def ensure_model(app, model_name, model_type):
    """Get or create a model in the app by name and type.

    Returns the Model object.
    """
    models = app.models()
    if model_name in models:
        return models[model_name]
    return app.model.new(model_name).model_type(model_type=model_type).build()


def model_has_version(app, model_name):
    """Return True if the named model already has at least one trained version."""
    try:
        model = app.models[model_name]
        return bool(model.versions())
    except Exception:
        return False
