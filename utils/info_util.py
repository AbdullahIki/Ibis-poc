def app_info(app):
    info = app.describe()
    app_info = info["app"]
    components = info["components"]

    print("=" * 50)
    print(f"  📱 App: {app_info['name']}")
    print("=" * 50)
    print(f"  ID:          {app_info['app_id']}")
    print(f"  Owner:       {app_info['owner']}")
    print(f"  Description: {app_info.get('description') or '(none)'}")
    print(f"  Created:     {app_info['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Modified:    {app_info['modified_at'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Last Used:   {app_info['last_used_at'].strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

    # Show components that have items
    active = {k: v for k, v in components.items() if v}
    if active:
        print("  📦 Components:")
        for name, items in active.items():
            print(f"    • {name.replace('_', ' ').title()}: {len(items)}")
    else:
        print("  📦 Components: (empty)")

    print("=" * 50)
    return 

def format_output(app, dataset_names):
    """
    Format the output datasets into a dictionary.

    Args:
        app: Ikigai app instance
        dataset_names: List of dataset names to format

    Returns:
        dict: Dictionary with dataset names as keys and their contents as values
    """
    output = {}
    for name in dataset_names:
        dataset = app.datasets[name]
        output[name] = dataset.df()
    return output


EXPLAINABILITY_METRICS_CONTEXT = """\
The dataset contains explainability metrics for the forecast. The metrics are defined as follows:

- ly_value / l2y_value: Actual quantity for the same period 1 or 2 years ago (year-over-year ratios).
- cy_recent_N_avg / ly_recent_N_avg / l2y_recent_N_avg: Mean of the last N actual periods (e.g. 7 days) for current year, last year, and two years ago.
- cy_qtd_avg / ly_qtd_avg / l2y_qtd_avg: Mean of all actuals in the quarter-to-date for current year, last year, and two years ago.
- run_rate: Mean forecasted value per period within the quarter.
- run_rate_over_qtd_avg: run_rate / cy_qtd_avg — whether the forecast is running hotter or cooler than what has been observed this quarter.
- qoq_cy_forecast / qoq_ly_forecast / qoq_l2y_forecast: Quarter-over-quarter ratio (current_quarter / previous_quarter) for current year, last year, and two years ago.
- abs_1_minus_run_rate_over_qtd_avg: |1 - run_rate_over_qtd_avg| — how different the forecast run rate is from the quarter-to-date average as an absolute value, where a lower value means the forecast is more in line with what has been observed this quarter.

When asked to compare, only compare the metrics  'qoq_cy_forecast', 'qoq_ly_forecast', 'qoq_l2y_forecast','abs_1_minus_run_rate_over_qtd_avg',  "yoy_forecast", "yoy_recent_7_avg", "yo2y_recent_7_avg", "yoy_recent_14_avg", "yoy_qtd_avg"
       """