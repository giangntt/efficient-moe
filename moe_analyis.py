import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    import os
    import json
    return alt, json, mo, os, pd


@app.cell
def _(json, os):
    def load_threshold_data(results_folder: str) -> dict:
        """
        Loads JSON data from files in the specified results folder.

        Each JSON file is expected to contain a 'config' key with a
        'dynamic_routing_threshold' sub-key. The full JSON data is stored
        in a dictionary using this threshold as the key.

        Args:
            results_folder (str): The path to the folder containing JSON result files.

        Returns:
            dict: A dictionary where keys are dynamic routing thresholds and values
                  are the full JSON data loaded from the corresponding file.
                  Returns an empty dictionary if the folder does not exist or
                  no valid data is found.
        """
        loaded_data = {}
        if os.path.exists(results_folder) and os.path.isdir(results_folder):
            for filename in os.listdir(results_folder):
                if filename.endswith(".json"):
                    filepath = os.path.join(results_folder, filename)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            if 'config' in data and 'dynamic_routing_threshold' in data['config']:
                                threshold_key = data['config']['dynamic_routing_threshold']
                                loaded_data[threshold_key] = data
                            else:
                                print(f"Warning: 'config' or 'dynamic_routing_threshold' key not found in {filename}")
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from {filename}")
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
        else:
            print(f"The folder '{results_folder}' does not exist or is not a directory.")
        return loaded_data

    results_folder = "results"
    loaded_threshold_data = load_threshold_data(results_folder)

    loaded_threshold_data
    return (loaded_threshold_data,)


@app.cell
def _(loaded_threshold_data):
    loaded_threshold_data[0.1]
    return


@app.function
def extract_mmlu_categories(loaded_threshold_data):
    unique_mmlu_categories = set()

    for threshold_str, data in loaded_threshold_data.items():
        results = data.get('results', {})

        # Add overall MMLU category
        if 'mmlu' in results:
            unique_mmlu_categories.add('MMLU Overall')

        # Add MMLU subcategories
        for cat_key, cat_data in results.items():
            if cat_key.startswith('mmlu_') and 'acc,none' in cat_data:
                alias = cat_data.get('alias', cat_key).split('-')[-1]
                unique_mmlu_categories.add(alias.strip())

    mmlu_categories_list = sorted(list(unique_mmlu_categories))
    return mmlu_categories_list


@app.cell
def _(loaded_threshold_data):
    mmlu_categories_list = extract_mmlu_categories(loaded_threshold_data)
    mmlu_categories_list
    return


@app.cell
def _():
    # Grouping MMLU sub-categories into broader topics for analysis.
    # These groupings are based on the standard MMLU benchmark structure.
    mmlu_category_groups = {
        "STEM": [
            "abstract_algebra", "anatomy", "astronomy", "college_biology",
            "college_chemistry", "college_computer_science", "college_mathematics",
            "college_medicine", "college_physics", "computer_security",
            "conceptual_physics", "econometrics", "electrical_engineering",
            "elementary_mathematics", "high_school_biology", "high_school_chemistry",
            "high_school_computer_science", "high_school_mathematics",
            "high_school_physics", "high_school_statistics", "machine_learning",
            "medical_genetics", "nutrition", "professional_medicine", "virology",
            "clinical_knowledge"
        ],
        "Humanities": [
            "formal_logic", "high_school_european_history", "high_school_geography",
            "high_school_us_history", "high_school_world_history", "international_law",
            "jurisprudence", "logical_fallacies", "moral_disputes",
            "moral_scenarios", "philosophy", "prehistory", "professional_law",
            "world_religions"
        ],
        "Social Sciences": [
            "business_ethics", "global_facts", "high_school_government_and_politics",
            "high_school_macroeconomics", "high_school_microeconomics",
            "high_school_psychology", "human_aging", "human_sexuality",
            "management", "marketing", "professional_accounting",
            "professional_psychology", "public_relations", "security_studies",
            "sociology", "us_foreign_policy"
        ],
        "Other": [
            "miscellaneous"
        ],
        "Overall": [
            "MMLU Overall"
        ]
    }

    mmlu_category_groups
    return (mmlu_category_groups,)


@app.cell
def _(alt, loaded_threshold_data, mo, pd):
    # Prepare data for plotting
    plot_data = []
    for threshold_str, data in loaded_threshold_data.items():
        try:
            threshold = float(threshold_str)
            results = data.get('results', {})

            # Overall MMLU
            if 'mmlu' in results:
                try:
                    plot_data.append({
                        'threshold': threshold,
                        'category': 'MMLU Overall',
                        'accuracy': float(results['mmlu']['acc,none']),
                        'stderr': float(results['mmlu']['acc_stderr,none'])
                    })
                except (ValueError, KeyError, TypeError) as e:
                    print(f"Could not process overall MMLU for threshold {threshold}: {e}")


            # All MMLU subcategories
            for cat_key, cat_data in results.items():
                # Check if it's an MMLU subcategory (starts with 'mmlu_' but is not the overall 'mmlu' key)
                if cat_key.startswith('mmlu_') and 'acc,none' in cat_data:
                    try:
                        # Extract alias, removing leading ' - ' if present
                        alias = cat_data.get('alias', cat_key).split('-')[-1]
                        plot_data.append({
                            'threshold': threshold,
                            'category': alias.strip(),
                            'accuracy': float(cat_data['acc,none']),
                            'stderr': float(cat_data['acc_stderr,none'])
                        })
                    except (ValueError, KeyError, TypeError) as e:
                        print(f"Could not process {cat_key} for threshold {threshold}: {e}")
        except (ValueError, TypeError) as e:
            print(f"Skipping data for threshold '{threshold_str}' due to error: {e}")
            continue

    if not plot_data:
        mo.md("No valid MMLU accuracy data found for plotting.")
    else:
        df_mmlu_acc = pd.DataFrame(plot_data)

        # Create the Altair chart
        chart = alt.Chart(df_mmlu_acc).mark_line(point=True).encode(
            x=alt.X('threshold:Q', title='Dynamic Routing Threshold'),
            y=alt.Y('accuracy:Q', title='Accuracy', scale=alt.Scale(zero=False)),
            color=alt.Color('category:N', title='MMLU Category'),
            tooltip=[
                alt.Tooltip('threshold', title='Threshold'),
                alt.Tooltip('category', title='Category'),
                alt.Tooltip('accuracy', title='Accuracy', format='.3f'),
                alt.Tooltip('stderr', title='Std. Error', format='.3f')
            ]
        ).properties(
            title='MMLU Accuracy Across Different Dynamic Routing Thresholds'
        )

        # Add error bars
        error_bars = alt.Chart(df_mmlu_acc).mark_errorbar(extent='stderr').encode(
            x='threshold:Q',
            y='accuracy:Q',
            yError='stderr:Q',
            color='category:N',
            tooltip=[
                alt.Tooltip('threshold', title='Threshold'),
                alt.Tooltip('category', title='Category'),
                alt.Tooltip('accuracy', title='Accuracy', format='.3f'),
                alt.Tooltip('stderr', title='Std. Error', format='.3f')
            ]
        )

        # Combine the line chart and error bars
        combined_chart = (chart + error_bars).interactive()
        combined_chart
    return chart, combined_chart, df_mmlu_acc


@app.cell
def _(combined_chart):
    combined_chart
    return


@app.cell
def _(loaded_threshold_data):
    loaded_threshold_data[0.25]
    return


@app.cell
def _(mmlu_category_groups):
    mmlu_category_groups.keys()
    return


@app.cell
def _(alt, mo):
    def create_mmlu_category_charts(df_mmlu_acc, mmlu_category_groups):
        category_charts = []

        if df_mmlu_acc is not None and not df_mmlu_acc.empty:
            for group_name, subcategories_list in mmlu_category_groups.items():
                # Filter the DataFrame for the current higher-level category's subcategories
                filtered_df = df_mmlu_acc[df_mmlu_acc['category'].isin(subcategories_list)]

                if not filtered_df.empty:
                    # Create the base chart for the current group
                    base_chart = alt.Chart(filtered_df).encode(
                        x=alt.X('threshold:Q', title='Dynamic Routing Threshold'),
                        y=alt.Y('accuracy:Q', title='Accuracy', scale=alt.Scale(zero=False)),
                        color=alt.Color('category:N', title='MMLU Subcategory'),
                        tooltip=[
                            alt.Tooltip('threshold', title='Threshold'),
                            alt.Tooltip('category', title='Subcategory'),
                            alt.Tooltip('accuracy', title='Accuracy', format='.3f'),
                            alt.Tooltip('stderr', title='Std. Error', format='.3f')
                        ]
                    ).properties(
                        title=f'{group_name} MMLU Accuracy Across Thresholds'
                    )

                    # Line chart
                    line_chart = base_chart.mark_line(point=True)

                    # Error bars
                    error_bars = base_chart.mark_errorbar(extent='stderr').encode(
                        yError='stderr:Q'
                    )

                    # Combine line chart and error bars, make interactive
                    combined_group_chart = (line_chart + error_bars).interactive()
                    category_charts.append(combined_group_chart)
        else:
            # Assuming 'mo' (marimo) is available in the notebook environment
            category_charts.append(mo.md("No MMLU accuracy data available to create category-specific charts."))

        return category_charts
    return (create_mmlu_category_charts,)


@app.cell
def _(create_mmlu_category_charts, df_mmlu_acc, mmlu_category_groups):
    category_charts = create_mmlu_category_charts(df_mmlu_acc, mmlu_category_groups)
    category_charts
    return (category_charts,)


@app.cell
def _(category_charts, mo):
    mo.vstack(category_charts)
    return


@app.cell
def _(alt, chart, loaded_threshold_data, mo, pd):
    """
    Processes expert activation data and creates a plot showing the
    average number of activated experts versus the dynamic routing threshold.
    """

    def _extract_expert_activation_data(loaded_threshold_data):
        """
        Extracts expert activation data from the loaded threshold data.

        Args:
            loaded_threshold_data (dict): A dictionary containing data for various thresholds.

        Returns:
            list: A list of dictionaries, each containing 'threshold' and 'average_experts'.
        """
        expert_data = []
        for threshold_str, data in loaded_threshold_data.items():
            try:
                threshold = float(threshold_str)
                report = data.get('expert_activation_report', {})
                avg_experts = report.get('overall_average')

                if avg_experts is not None:
                    expert_data.append({
                        'threshold': threshold,
                        'average_experts': float(avg_experts),
                    })
            except (ValueError, TypeError) as e:
                print(f"Could not process expert activation data for threshold '{threshold_str}': {e}")
                continue
        return expert_data

    def _create_expert_activation_chart(df_expert_activation):
        """
        Creates an Altair chart showing average activated experts vs. dynamic routing threshold.

        Args:
            df_expert_activation (pd.DataFrame): DataFrame with 'threshold' and 'average_experts' columns.

        Returns:
            alt.Chart: An Altair chart object.
        """
        chart = alt.Chart(df_expert_activation).mark_line(point=True).encode(
            x=alt.X('threshold:Q', title='Dynamic Routing Threshold'),
            y=alt.Y('average_experts:Q', title='Average Activated Experts'),
            tooltip=[
                alt.Tooltip('threshold', title='Threshold'),
                alt.Tooltip('average_experts', title='Avg. Activated Experts', format='.2f')
            ]
        ).properties(
            title='Average Activated Experts vs. Dynamic Routing Threshold'
        ).interactive()
        return chart

    expert_data_list = _extract_expert_activation_data(loaded_threshold_data)

    if expert_data_list:
        df_expert_activation = pd.DataFrame(expert_data_list)
        expert_chart = _create_expert_activation_chart(df_expert_activation)
        chart
    else:
        mo.md("No expert activation data foundto plot.")
    expert_chart
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
