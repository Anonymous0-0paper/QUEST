import os
import pandas as pd
import re

input_folder = r"C:\Users\Mehrab\OneDrive\Desktop\quest\new\results-excel"
output_file = r"C:\Users\Mehrab\OneDrive\Desktop\quest\new\results\energy_table.tex"

# Define the applications and their loads
APPLICATIONS = {
    'Alibaba': {'loads': [1000, 2000], 'display_name': 'Alibaba'},
    'monatge': {'loads': [25, 50, 100], 'display_name': 'Montage'},
    'cybershake': {'loads': [30, 50, 100], 'display_name': 'CyberShake'},
    'epigenomics': {'loads': [24, 46, 100], 'display_name': 'Epigenomics'},
    'inspiral': {'loads': [30, 50, 100], 'display_name': 'Inspiral'},
    'scientific': {'loads': ['Montage', 'CyberShake', 'Inspiral', 'Epigenomics'], 'display_name': 'Scientific'}
}


# Function to extract file information
def extract_file_info(filename):
    pattern = r'result-([A-Za-z]+)-(\d+)\.xlsx'
    match = re.match(pattern, filename)
    if match:
        app_name = match.group(1)
        z_value = int(match.group(2))
        return app_name, z_value * 2
    return None, None


# Function to process Excel files and extract data
def process_excel_files(folder_path):
    data = {}

    for filename in os.listdir(folder_path):
        if not filename.endswith('.xlsx'):
            continue

        app_name, z_value = extract_file_info(filename)
        if app_name is None or z_value is None:
            print(f"Skipping file with invalid format: {filename}")
            continue

        if app_name not in APPLICATIONS:
            print(f"Unknown application: {app_name} in file {filename}")
            continue

        file_path = os.path.join(folder_path, filename)
        try:
            # Read the Excel file's "Energy" sheet
            df = pd.read_excel(file_path, sheet_name="Energy")

            # Determine number of rows to read based on load count
            loads = APPLICATIONS[app_name]['loads']
            x = len(loads) * 8  # For numerical loads

            # Read rows 1 to x+1 and columns A to F
            data_subset = df.iloc[0:x, 0:6]

            if app_name not in data:
                data[app_name] = {}
            if z_value not in data[app_name]:
                data[app_name][z_value] = {}

            # Extract algorithm names and their values
            for index, row in data_subset.iterrows():
                algo_name = row.iloc[0]  # Algorithm name (column A)
                load_value = row.iloc[1]  # Load value (column B)
                avg_value = row.iloc[2]  # Average value (column C)
                min_value = row.iloc[3]  # Min value (column D)
                max_value = row.iloc[4]  # Max value (column E)
                std_value = row.iloc[5]  # Std Dev value (column F)

                # Replace "RL" with "ID3CO" in algorithm name
                if algo_name == "RL":
                    algo_name = "ID3CO"

                if algo_name not in data[app_name][z_value]:
                    data[app_name][z_value][algo_name] = {}

                # Map the load index to the actual load value from APPLICATIONS
                load_idx = int(load_value) - 1  # Assuming load values start at 1
                actual_load = loads[load_idx] if load_idx < len(loads) else load_value

                data[app_name][z_value][algo_name][actual_load] = {
                    'avg': avg_value / 1000_000,
                    'min': min_value,
                    'max': max_value,
                    'std': std_value / 1000_000
                }

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    return data


# Function to generate LaTeX table
def generate_latex_table(data):
    # Find all algorithms and applications
    all_algorithms = set()
    for app in data:
        for z in data[app]:
            all_algorithms.update(data[app][z].keys())

    all_algorithms = sorted(list(all_algorithms))

    # Start LaTeX table
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Energy Consumption for Different Algorithms and Applications}\n"
    latex += "\\label{tab:energy_consumption}\n"
    latex += "\\begin{tabular}{|l|"

    # Calculate total columns (now doubled because we have Avg and Std Dev)
    metrics = ["Avg", "Std Dev"]
    num_metrics = len(metrics)

    # Add multicolumns for each application, z value, load, and metric
    apps_header = ""
    z_header = ""
    load_header = ""
    metric_header = ""

    app_col_count = {}  # To track column counts for each app

    for app in sorted(data.keys()):
        z_values = sorted(data[app].keys())
        loads_per_app = len(APPLICATIONS[app]['loads']) * len(z_values) * num_metrics
        app_col_count[app] = loads_per_app
        apps_header += f"\\multicolumn{{{loads_per_app}}}{{c|}}{{{APPLICATIONS[app]['display_name'] if app in APPLICATIONS else app}}} & "

        for z in z_values:
            loads = sorted(APPLICATIONS[app]['loads']) if app in APPLICATIONS else []
            loads_per_z = len(loads) * num_metrics
            z_header += f"\\multicolumn{{{loads_per_z}}}{{c|}}{{{z}}} & "

            for load in loads:
                load_header += f"\\multicolumn{{{num_metrics}}}{{c|}}{{{load}}} & "

                for metric in metrics:
                    metric_header += f"\\multicolumn{{1}}{{c|}}{{{metric}}} & "

    # Remove trailing " & " from headers
    apps_header = apps_header.rstrip(" & ")
    z_header = z_header.rstrip(" & ")
    load_header = load_header.rstrip(" & ")
    metric_header = metric_header.rstrip(" & ")

    # Complete column spec (doubled for Avg and Std Dev)
    total_cols = sum([len(sorted(APPLICATIONS[app]['loads'])) * len(data[app]) * num_metrics for app in data])
    latex += "c|" * total_cols + "}\n"
    latex += "\\hline\n"

    # Add headers with four rows now
    latex += f"\\multirow{{4}}{{*}}{{Algorithm}} & {apps_header} \\\\ \\cline{{2-{total_cols + 1}}}\n"
    latex += f"& {z_header} \\\\ \\cline{{2-{total_cols + 1}}}\n"
    latex += f"& {load_header} \\\\ \\cline{{2-{total_cols + 1}}}\n"
    latex += f"& {metric_header} \\\\ \\hline\n"

    # Add data rows for each algorithm
    for algo in all_algorithms:
        latex += f"{algo} & "

        for app in sorted(data.keys()):
            z_values = sorted(data[app].keys())
            for z in z_values:
                loads = sorted(APPLICATIONS[app]['loads']) if app in APPLICATIONS else []

                algo_data = data[app][z].get(algo, {})
                for load in loads:
                    if load in algo_data:
                        avg_value = algo_data[load]['avg']
                        std_value = algo_data[load]['std']
                        latex += f"{avg_value:.1f} & {std_value:.1f} & "
                    else:
                        latex += "- & - & "

        # Remove trailing " & " and add line end
        latex = latex.rstrip(" & ") + " \\\\ \\hline\n"

    # End LaTeX table
    latex += "\\end{tabular}\n"
    latex += "\\end{table}"

    return latex


# Main function
def main():
    data = process_excel_files(input_folder)

    if not data:
        print("No valid data found in the specified folder.")
        return

    print("Generating LaTeX table...")
    latex_table = generate_latex_table(data)

    with open(output_file, "w") as f:
        f.write(latex_table)

    print(f"LaTeX table has been written to {output_file}")


if __name__ == "__main__":
    main()