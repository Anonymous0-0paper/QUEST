import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import re
from pathlib import Path

# Configuration
input_folder = r"C:\Users\Mehrab\OneDrive\Desktop\quest\new\results-excel"
output_root = r"C:\Users\Mehrab\OneDrive\Desktop\quest\new\results\Heatmap"

# Categories for the three types of heatmaps
CATEGORIES = ["AoI", "Energy", "Makespan"]

# Regular expression to extract base name and number from filenames
FILENAME_PATTERN = re.compile(r"result-(.+)-(\d+)\.xlsx", re.IGNORECASE)

# Y-axis labels for the loads
# Each dataset has different load values
LOAD_CONFIGS = {
    'Alibaba': [1000, 2000],
    'cybershake': [30, 50, 100],
    'monatge': [25, 50, 100],
    'epigenomics': [24, 46, 100],
    'inspiral': [30, 50, 100],
    'scientific': ['Montage', 'CyberShake', 'Inspiral', 'Epigenomics']  # Special case
}

# End columns based on the number of loads
END_COLUMNS = {
    2: 'T',  # For 2 loads (Alibaba)
    3: 'U',  # For 3 loads (Cybershake, Montage, Epigenomics, Inspiral)
    4: 'V'  # For 4 loads (Scientific)
}

# Data locations in the Excel sheet - starting rows and columns are fixed
DATA_LOCATIONS = [
    {'rows': (19, 25), 'cols': ('R', None)},  # Location 1: rows 19-25, columns R-? (AoI)
    {'rows': (53, 59), 'cols': ('R', None)},  # Location 2: rows 53-59, columns R-? (Energy)
    {'rows': (84, 90), 'cols': ('R', None)}  # Location 3: rows 84-90, columns R-? (Makespan)
]


def convert_col_letter_to_index(col_letter):
    """Convert column letter to numeric index (A=0, B=1, etc.)"""
    return ord(col_letter.upper()) - ord('A')


def extract_filename_info(filename):
    """Extract base name and number from filename."""
    match = FILENAME_PATTERN.match(filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def get_dataset_loads(base_name):
    """Get the appropriate loads for the dataset."""
    return LOAD_CONFIGS.get(base_name, ['Load 1', 'Load 2'])


def determine_end_column(num_loads):
    """Determine end column based on number of loads."""
    return END_COLUMNS.get(num_loads, 'T')  # Default to 'T' if not found


def load_data(file_path, sheet_name, location_idx=0):
    """Load the Excel file and extract data from specified location."""
    # Read the entire Excel sheet
    try:
        raw_df = pd.read_excel(file_path, sheet_name=sheet_name)
    except Exception as e:
        print(f"Error reading Excel file {file_path}, sheet {sheet_name}: {str(e)}")
        return pd.DataFrame()  # Return empty dataframe on error

    # Get the specified data location
    if location_idx >= len(DATA_LOCATIONS):
        print(f"Location index {location_idx} out of range")
        return pd.DataFrame()

    location = DATA_LOCATIONS[location_idx]
    start_row, end_row = location['rows']
    start_col_letter = location['cols'][0]

    # Extract base name to determine which loads to use
    filename = os.path.basename(file_path)
    base_name, _ = extract_filename_info(filename)

    if not base_name:
        # Default loads if filename doesn't match pattern
        dataset_loads = ['Load 1', 'Load 2']
    else:
        # Get the appropriate loads for this dataset
        dataset_loads = get_dataset_loads(base_name)

    # Determine end column based on number of loads
    num_loads = len(dataset_loads)
    end_col_letter = determine_end_column(num_loads)

    print(f"Processing {base_name} with {num_loads} loads, using columns {start_col_letter}-{end_col_letter}")

    # Convert column letters to indices
    start_col = convert_col_letter_to_index(start_col_letter)
    end_col = convert_col_letter_to_index(end_col_letter)

    # Extract data from the specified region
    # Adjusting for 0-based indexing (Excel row 19 = index 18)
    data_subset = raw_df.iloc[(start_row - 2):(end_row), start_col:(end_col + 1)]

    # Extract method names (first column)
    methods = data_subset.iloc[:, 0].tolist()
    methods = [m for m in methods if isinstance(m, str) and m.strip()]

    # Replace "QUEST_NDVFS" with "QUEST_ND" in the method names
    methods = [ ("QUEST_ND" if m == "QUEST_NDVFS" else "ID3CO" if m == "RL" else m) for m in methods]

    # Number of data columns (excluding the first column with method names)
    num_data_cols = data_subset.shape[1] - 1

    # Create dataframe with proper orientation
    # Each row is a data point, each column is a method
    data_array = np.zeros((num_data_cols, len(methods)))

    for i, method in enumerate(methods):
        # Find the original method name (before replacing QUEST_NDVFS)
        original_method = method
        if method == "QUEST_ND":
            original_method = "QUEST_NDVFS"
        elif method == "ID3CO":
            original_method = "RL"

        # Find the row with this method name
        method_rows = data_subset.iloc[:, 0] == original_method
        if any(method_rows):
            # Get the row index
            row_idx = method_rows.idxmax() - data_subset.index[0]

            # Extract values for this method (excluding the method name column)
            for j in range(num_data_cols):
                try:
                    val = data_subset.iloc[row_idx, j + 1]
                    # Try to convert to float
                    if pd.notna(val):
                        try:
                            data_array[j, i] = float(val)
                        except (ValueError, TypeError):
                            data_array[j, i] = np.nan
                except IndexError:
                    data_array[j, i] = np.nan

    # Use the appropriate loads as row labels
    row_labels = []
    for j in range(num_data_cols):
        if j < len(dataset_loads):
            row_labels.append(str(dataset_loads[j]))
        else:
            row_labels.append(f'Load {j + 1}')

    # Create dataframe with data points as rows and methods as columns
    data_array = data_array - 1
    df = pd.DataFrame(data_array, index=row_labels, columns=methods)

    return df


def create_heatmap(df, figsize=(16, 8), cmap='viridis'):
    """Create a single heatmap with percentage values."""
    plt.figure(figsize=figsize)

    # Convert values to percentages for display (values are already 0-1 scale)
    df_percentage = df * 100

    # Create the heatmap with percentage values
    ax = plt.gca()

    # Create annotation with percentage signs directly
    annot_matrix = df_percentage.map(lambda x: f'{x:.1f}%' if pd.notna(x) else '')

    hm = sns.heatmap(df_percentage, ax=ax, cmap=cmap, annot=annot_matrix,
                     fmt='',  # Empty format since annotations are pre-formatted
                     cbar_kws={'label': 'Improvement (%)'},
                     annot_kws={"size": 28})

    # Set axis labels
    ax.set_ylabel('Load', fontsize=26)

    # Set tick parameters
    ax.tick_params(labelsize=26)

    # Make y-axis labels horizontal instead of vertical
    plt.yticks(rotation=0)

    # Rotate x-axis labels (algorithm names) by 45 degrees
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Set colorbar properties
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=26)
    cbar.set_label('Improvement (%)', fontsize=26)

    plt.tight_layout(pad=3.0)

    return plt.gcf()


def save_heatmap(df, base_name, number, category_idx):
    """Save a heatmap to the appropriate category folder."""
    # Skip if dataframe is empty
    if df.empty:
        print(f"No valid data for {base_name}, {CATEGORIES[category_idx]}, skipping...")
        return

    # Create output folder path
    category = CATEGORIES[category_idx]
    export_dir = Path(output_root) / category / number
    export_dir.mkdir(parents=True, exist_ok=True)

    # Create output file path
    output_file = export_dir / f"{base_name}.pdf"

    # Create and save the heatmap
    fig = create_heatmap(df, cmap='viridis')

    # Save the figure
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Heatmap saved: {output_file}")


def process_excel_file(file_path):
    """Process a single Excel file and generate all heatmaps."""
    # Extract base name and number from filename
    filename = os.path.basename(file_path)
    base_name, number = extract_filename_info(filename)

    if not base_name or not number:
        print(f"Skipping malformed filename: {filename}")
        return

    print(f"Processing file: {filename}, Base: {base_name}, Number: {number}")

    # Skip processing for Scientific except for specific loads files
    if base_name == 'Scientific' and number not in ['1', '2', '3', '4', '5']:
        print(f"Skipping Scientific file with non-standard number: {number}")
        return

    # Process each category
    sheet_name = 'Charts'  # Assuming all files use the same sheet name

    for i, category in enumerate(CATEGORIES):
        try:
            # Load data for this category
            df = load_data(file_path, sheet_name, i)

            # Save heatmap to appropriate folder
            save_heatmap(df, base_name, number, i)
        except Exception as e:
            print(f"Error processing {filename} for {category}: {str(e)}")


def main():
    """Main function to process all Excel files in the input folder."""
    # Get list of all Excel files in the input folder
    input_path = Path(input_folder)
    excel_files = list(input_path.glob('result-*.xlsx'))

    print(f"Found {len(excel_files)} Excel files to process")

    # Process each Excel file
    for file_path in excel_files:
        process_excel_file(file_path)


if __name__ == "__main__":
    main()