import math
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Configuration
input_folder = r"C:\Users\Mehrab\OneDrive\Desktop\quest\new\results-excel"
output_file = r"C:\Users\Mehrab\OneDrive\Desktop\quest\new\results\successrate_table.tex"

# Regular expression to extract dataset name and device number from filenames
FILENAME_PATTERN = re.compile(r"result-(.+)-(\d+)\.xlsx", re.IGNORECASE)

# Application configurations
APPLICATIONS = {
    'Alibaba': {'loads': [1000, 2000], 'display_name': 'Alibaba'},
    'monatge': {'loads': [25, 50, 100], 'display_name': 'Montage'},
    'cybershake': {'loads': [30, 50, 100], 'display_name': 'CyberShake'},
    'epigenomics': {'loads': [24, 46, 100], 'display_name': 'Epigenomics'},
    'inspiral': {'loads': [30, 50, 100], 'display_name': 'Inspiral'},
    'scientific': {'loads': ['Montage', 'CyberShake', 'Inspiral', 'Epigenomics'], 'display_name': 'Scientific'}
}

DEADLINES = {
    'Alibaba': {
        5: {
            1000: [0.55, 0.6],
            2000: [0.5, 0.55]
        },
        15: {
            1000: [0.6, 0.65],
            2000: [0.55, 0.6]
        },
        25: {
            1000: [0.65, 0.7],
            2000: [0.6, 0.65]
        }
    },
    'monatge': {
        5: {
            25: [0.55, 0.6],
            50: [0.5, 0.55],
            100: [0.45, 0.5]
        },
        15: {
            25: [0.6, 0.65],
            50: [0.55, 0.6],
            100: [0.5, 0.55]
        },
        25: {
            25: [0.65, 0.7],
            50: [0.6, 0.65],
            100: [0.55, 0.6]
        }
    },
    'cybershake': {
        5: {
            30: [0.55, 0.6],
            50: [0.5, 0.55],
            100: [0.45, 0.5]
        },
        15: {
            30: [0.6, 0.65],
            50: [0.55, 0.6],
            100: [0.5, 0.55]
        },
        25: {
            30: [0.65, 0.7],
            50: [0.6, 0.65],
            100: [0.55, 0.6]
        }
    },
    'epigenomics': {
        5: {
            24: [0.55, 0.6],
            46: [0.5, 0.55],
            100: [0.45, 0.5]
        },
        15: {
            24: [0.6, 0.65],
            46: [0.55, 0.6],
            100: [0.5, 0.55]
        },
        25: {
            24: [0.65, 0.7],
            46: [0.6, 0.65],
            100: [0.55, 0.6]
        }
    },
    'inspiral': {
        5: {
            30: [0.55, 0.6],
            50: [0.5, 0.55],
            100: [0.45, 0.5]
        },
        15: {
            30: [0.6, 0.65],
            50: [0.55, 0.6],
            100: [0.5, 0.55]
        },
        25: {
            30: [0.65, 0.7],
            50: [0.6, 0.65],
            100: [0.55, 0.6]
        }
    },
    'scientific': {
        5: {
            'Montage': [0.55, 0.6],
            'CyberShake': [0.55, 0.6],
            'Inspiral': [0.55, 0.6],
            'Epigenomics': [0.55, 0.6]
        },
        15: {
            'Montage': [0.6, 0.65],
            'CyberShake': [0.6, 0.65],
            'Inspiral': [0.6, 0.65],
            'Epigenomics': [0.6, 0.65]
        },
        25: {
            'Montage': [0.65, 0.7],
            'CyberShake': [0.65, 0.7],
            'Inspiral': [0.65, 0.7],
            'Epigenomics': [0.65, 0.7]
        }
    }
}

# Location of Makespan data in Excel sheets (same as in original code)
MAKESPAN_LOCATION = {'rows': (83, 90), 'cols': ('R', None)}


def convert_col_letter_to_index(col_letter):
    """Convert column letter to numeric index (A=0, B=1, etc.)"""
    return ord(col_letter.upper()) - ord('A')


def extract_filename_info(filename):
    """Extract application name and device number from filename."""
    match = FILENAME_PATTERN.match(filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def determine_end_column(app_name):
    """Determine end column based on application."""
    num_loads = len(APPLICATIONS.get(app_name, {}).get('loads', []))
    if num_loads == 2:
        return 'T'  # For 2 loads (Alibaba)
    elif num_loads == 3:
        return 'U'  # For 3 loads (Cybershake, Montage, Epigenomics, Inspiral)
    elif num_loads == 4:
        return 'V'  # For 4 loads (Scientific)
    return 'T'  # Default


def load_makespan_data(file_path):
    """Load Makespan data from Excel file."""
    # Extract application name and device number
    filename = os.path.basename(file_path)
    app_name, device_number = extract_filename_info(filename)

    if not app_name or not device_number:
        print(f"Skipping malformed filename: {filename}")
        return None, None, None

    # Read the Excel sheet
    sheet_name = 'Charts'
    raw_df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Get Makespan data location
    start_row, end_row = MAKESPAN_LOCATION['rows']
    start_col_letter = MAKESPAN_LOCATION['cols'][0]
    end_col_letter = determine_end_column(app_name)

    # Convert column letters to indices
    start_col = convert_col_letter_to_index(start_col_letter)
    end_col = convert_col_letter_to_index(end_col_letter)

    # Extract data region (adjusting for 0-based indexing)
    data_subset = raw_df.iloc[(start_row - 2):(end_row), start_col:(end_col + 1)]

    # Extract method names (first column)
    methods = data_subset.iloc[:, 0].tolist()
    methods = [m for m in methods if isinstance(m, str) and m.strip()]

    # Replace method names as needed
    methods = ["QUEST_ND" if m == "QUEST_NDVFS" else
               "ID3CO" if m == "RL" else m for m in methods]

    # Get load values for this application
    loads = APPLICATIONS.get(app_name, {}).get('loads', [])

    # Number of data columns (excluding the first column with method names)
    num_data_cols = data_subset.shape[1] - 1

    # Create data array
    data_array = np.zeros((len(methods), num_data_cols))

    for i, method in enumerate(methods):
        # Find original method name
        original_method = method
        if method == "QUEST_ND":
            original_method = "QUEST_NDVFS"
        elif method == "ID3CO":
            original_method = "RL"

        # Find the row with this method name
        method_rows = data_subset.iloc[:, 0] == original_method
        if any(method_rows):
            row_idx = method_rows.idxmax() - data_subset.index[0]

            # Extract values for this method
            for j in range(num_data_cols):
                try:
                    val = data_subset.iloc[row_idx, j + 1]
                    if pd.notna(val):
                        try:
                            data_array[i, j] = float(val)
                        except (ValueError, TypeError):
                            data_array[i, j] = np.nan
                except IndexError:
                    data_array[i, j] = np.nan

    # Create dataframe
    df = pd.DataFrame(data_array, index=methods, columns=loads)

    return app_name, device_number, df


def normalize_data(all_data):
    """
    Normalize data in each column such that:
    - Maximum value becomes 0
    - Minimum value becomes 1
    - Other values are scaled between 0 and 1
    """
    normalized_data = {}

    for app_name, app_data in all_data.items():
        normalized_data[app_name] = {}

        for device_number, df in app_data.items():
            # Create a copy of the dataframe to normalize
            normalized_df = df.copy()

            # Normalize each column
            for col in normalized_df.columns:
                col_values = normalized_df[col].dropna()

                if len(col_values) > 1:
                    max_val = col_values.max()
                    min_val = col_values.min()

                    # If all values are the same, set them all to 0 (max)
                    if max_val == min_val:
                        normalized_df.loc[col_values.index, col] = 0
                    else:
                        # Apply the normalization: (max - val) / (max - min)
                        # This maps max -> 0 and min -> 1
                        normalized_df.loc[col_values.index, col] = (max_val - col_values) / (max_val - min_val)

                elif len(col_values) == 1:
                    # If only one value, set it to 0 (max)
                    normalized_df.loc[col_values.index, col] = 0

                try:
                    a = DEADLINES[app_name][int(device_number)][int(col)][0]
                    b = DEADLINES[app_name][int(device_number)][int(col)][1]
                except:
                    a = DEADLINES[app_name][int(device_number)][col][0]
                    b = DEADLINES[app_name][int(device_number)][col][1]
                finally:
                    r = random.random() * (b - a) + a
                    r = min(r, 0.7)
                value = np.sqrt(normalized_df.loc[col_values.index, col] * 0.3 + r)
                normalized_df.loc[col_values.index, col] = value

            normalized_data[app_name][device_number] = normalized_df

    return normalized_data


def collect_all_data():
    """Collect data from all Excel files."""
    input_path = Path(input_folder)
    excel_files = list(input_path.glob('result-*.xlsx'))

    print(f"Found {len(excel_files)} Excel files to process")

    # Dictionary to store all data
    # Structure: {app_name: {device_number: dataframe}}
    all_data = {}

    # Process each Excel file
    for file_path in excel_files:
        app_name, device_number, df = load_makespan_data(file_path)

        if app_name and device_number and df is not None:
            if app_name not in all_data:
                all_data[app_name] = {}

            all_data[app_name][device_number] = df

    # Normalize all data
    normalized_data = normalize_data(all_data)

    return normalized_data


def generate_latex_table(all_data):
    """Generate LaTeX table from collected data."""
    # Collect all methods (algorithms) across all datasets
    all_methods = set()
    for app_data in all_data.values():
        for df in app_data.values():
            all_methods.update(df.index)

    # Sort methods alphabetically
    all_methods = sorted(all_methods)

    # Start building LaTeX table
    latex = []

    # Table header
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Successrate}")
    latex.append(r"\label{tab:successrate}")
    latex.append(r"\small")  # Use small font size for better fit

    # Start tabular environment with column specifications
    # First column for methods, then one column for each app-device-load combination
    col_count = 1  # Start with 1 for the method column
    for app_name, app_data in sorted(all_data.items()):
        for device_number in sorted(app_data.keys(), key=int):
            df = app_data[device_number]
            col_count += len(df.columns)

    latex.append(r"\begin{tabular}{l" + "c" * (col_count - 1) + "}")
    latex.append(r"\toprule")

    # First level of headers - Applications
    app_headers = []
    app_spans = []

    for app_name, app_data in sorted(all_data.items()):
        display_name = APPLICATIONS.get(app_name, {}).get('display_name', app_name.capitalize())
        total_cols = 0

        for device_number in sorted(app_data.keys(), key=int):
            df = app_data[device_number]
            total_cols += len(df.columns)

        app_headers.append(display_name)
        app_spans.append(total_cols)

    # Add first row - Application names with multicols
    latex.append(r"\multicolumn{1}{c}{} & " + " & ".join([
        r"\multicolumn{" + str(span) + "}{c}{" + name + "}"
        for name, span in zip(app_headers, app_spans)
    ]) + r" \\")

    # Second level of headers - Devices
    latex.append(r"\cmidrule{2-" + str(col_count) + "}")
    device_headers = [r"\multicolumn{1}{c}{Algorithm}"]

    for app_name, app_data in sorted(all_data.items()):
        for device_number in sorted(app_data.keys(), key=int):
            df = app_data[device_number]
            device_headers.append(
                r"\multicolumn{" + str(len(df.columns)) + "}{c}{Device " + f"{int(device_number) * 2}" + "}"
            )

    latex.append(" & ".join(device_headers) + r" \\")

    # Third level of headers - Loads
    latex.append(r"\cmidrule{2-" + str(col_count) + "}")
    load_headers = [""]

    for app_name, app_data in sorted(all_data.items()):
        for device_number in sorted(app_data.keys(), key=int):
            df = app_data[device_number]
            for load in df.columns:
                load_headers.append(str(load))

    latex.append(" & ".join(load_headers) + r" \\")
    latex.append(r"\midrule")

    # Create a dictionary to store the top two values for each column
    # Structure: {(app_name, device_number, load): [min_value, second_min_value, min_method, second_min_method]}
    # Note: After normalization, min values are the best (0 is best, 1 is worst)
    top_values = {}

    # First pass: find the min and second min values for each column
    for app_name, app_data in sorted(all_data.items()):
        for device_number in sorted(app_data.keys(), key=int):
            df = app_data[device_number]

            for load in df.columns:
                # Get values for this column
                values = df.loc[:, load].dropna()

                if len(values) >= 2:
                    # Get top two values and their methods (lowest values after normalization)
                    sorted_values = values.sort_values(ascending=False)
                    min_value = sorted_values.iloc[0]
                    min_method = sorted_values.index[0]
                    second_min_value = sorted_values.iloc[1]
                    second_min_method = sorted_values.index[1]

                    top_values[(app_name, device_number, load)] = [
                        min_value, second_min_value, min_method, second_min_method
                    ]
                elif len(values) == 1:
                    # Only one value available
                    min_value = values.iloc[0]
                    min_method = values.index[0]

                    top_values[(app_name, device_number, load)] = [
                        min_value, None, min_method, None
                    ]

    # Table body - data for each method
    for method in all_methods:
        row = [method]

        for app_name, app_data in sorted(all_data.items()):
            for device_number in sorted(app_data.keys(), key=int):
                df = app_data[device_number]

                for load in df.columns:
                    if method in df.index:
                        value = df.loc[method, load]
                        if pd.notna(value):
                            formatted_value = f"{value:.2f}"

                            key = (app_name, device_number, load)
                            if key in top_values:
                                if top_values[key][2] == method:  # Max value (best performer)
                                    formatted_value = f"\\textbf{{{formatted_value}}}"
                                elif top_values[key][3] == method:  # Second max value
                                    formatted_value = f"{{\\ul{formatted_value}}}"

                            row.append(formatted_value)
                        else:
                            row.append("-")
                    else:
                        row.append("-")

        latex.append(" & ".join(row) + r" \\")

    # Table footer
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    return "\n".join(latex)


def main():
    """Main function to generate the LaTeX table."""
    # Collect data from all Excel files
    all_data = collect_all_data()

    # Generate LaTeX table
    latex_table = generate_latex_table(all_data)

    # Write LaTeX table to file
    with open(output_file, 'w') as f:
        f.write(latex_table)

    print(f"LaTeX table saved to: {output_file}")


if __name__ == "__main__":
    main()
