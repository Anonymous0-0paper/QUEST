import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import os
import re
from pathlib import Path

# Configuration
input_folder = r"C:\Users\mehrab\Desktop\QUEST\QUEST\result\v3\results-excel"
output_root = r"C:\Users\mehrab\Desktop\QUEST\QUEST\result\v4\results"

# Categories for the three types of heatmaps
CATEGORIES = ["AoI", "Energy", "Makespan"]

# Scale mapping
SCALES = {
    '5': 'Small',
    '15': 'Medium',
    '25': 'Large'
}

# Regular expression to extract base name and number from filenames
FILENAME_PATTERN = re.compile(r"result-(.+)-(\d+)\.xlsx", re.IGNORECASE)

# Workload configurations with their loads
WORKLOAD_CONFIGS = {
    'alibaba': [1000, 2000],
    'cybershake': [30, 50, 100],
    'montage': [25, 50, 100],
    'epigenomics': [24, 46, 100],
    'inspiral': [30, 50, 100]
}

# Algorithm names (in order) - QUEST removed since it's the baseline
ALGORITHMS = ['QUEST_ND', 'Fuzzy', 'NSGA3', 'MQGA', 'Greedy', 'MOPSO', 'ID3CO']

# Data locations in the Excel sheet
DATA_LOCATIONS = [
    {'rows': (19, 25), 'cols': ('R', None)},  # AoI
    {'rows': (53, 59), 'cols': ('R', None)},  # Energy
    {'rows': (84, 90), 'cols': ('R', None)}  # Makespan
]

# End columns based on the number of loads
END_COLUMNS = {
    2: 'T',  # For 2 loads (Alibaba)
    3: 'U',  # For 3 loads
    4: 'V'  # For 4 loads (Scientific)
}


def convert_col_letter_to_index(col_letter):
    """Convert column letter to numeric index (A=0, B=1, etc.)"""
    return ord(col_letter.upper()) - ord('A')


def extract_filename_info(filename):
    """Extract base name and number from filename."""
    match = FILENAME_PATTERN.match(filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def load_single_file_data(file_path, category_idx):
    """Load data from a single Excel file for a specific category."""
    filename = os.path.basename(file_path)
    base_name, scale_num = extract_filename_info(filename)

    if not base_name:
        return None, None, None

    # Get workload config
    workload_key = base_name.lower()
    if workload_key not in WORKLOAD_CONFIGS:
        print(f"Unknown workload: {base_name}")
        return None, None, None

    loads = WORKLOAD_CONFIGS[workload_key]
    num_loads = len(loads)

    # Determine columns
    location = DATA_LOCATIONS[category_idx]
    start_row, end_row = location['rows']
    start_col_letter = 'R'
    end_col_letter = END_COLUMNS.get(num_loads, 'T')

    start_col = convert_col_letter_to_index(start_col_letter)
    end_col = convert_col_letter_to_index(end_col_letter)

    try:
        raw_df = pd.read_excel(file_path, sheet_name='Charts')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None

    # Extract data
    data_subset = raw_df.iloc[(start_row - 2):(end_row), start_col:(end_col + 1)]

    # Build result dictionary: {algorithm: [values for each load]}
    result = {}

    for idx in range(len(data_subset)):
        method_name = data_subset.iloc[idx, 0]
        if not isinstance(method_name, str):
            continue
        method_name = method_name.strip()

        # Rename methods
        if method_name == "QUEST_NDVFS":
            method_name = "QUEST_ND"
        elif method_name == "RL":
            method_name = "ID3CO"

        # Get values for each load
        values = []
        for j in range(1, num_loads + 1):
            try:
                val = data_subset.iloc[idx, j]
                if pd.notna(val):
                    values.append(float(val) - 1)  # Subtract 1 as in original code
                else:
                    values.append(np.nan)
            except (IndexError, ValueError):
                values.append(np.nan)

        result[method_name] = values

    return base_name, loads, result


def collect_all_data_for_scale(input_folder, scale_num, category_idx):
    """Collect data from all workloads for a specific scale and category."""
    input_path = Path(input_folder)
    pattern = f"result-*-{scale_num}.xlsx"
    files = list(input_path.glob(pattern))

    all_data = {}  # {workload_name: {algorithm: [values]}}
    workload_loads = {}  # {workload_name: [load1, load2, ...]}

    for file_path in files:
        base_name, loads, data = load_single_file_data(file_path, category_idx)
        if base_name and data:
            all_data[base_name.lower()] = data
            workload_loads[base_name.lower()] = loads

    return all_data, workload_loads


def build_combined_dataframe(all_data, workload_loads):
    """Build a combined DataFrame with multi-level columns."""
    # Define workload order (lowercase for matching)
    workload_order = ['alibaba', 'montage', 'cybershake', 'epigenomics', 'inspiral', 'scientific']

    # Display names (capitalized)
    display_names = {
        'alibaba': 'Alibaba',
        'montage': 'Montage',
        'cybershake': 'CyberShake',
        'epigenomics': 'Epigenomics',
        'inspiral': 'Inspiral',
        'scientific': 'Scientific'
    }

    # Filter to only existing workloads
    existing_workloads = [w for w in workload_order if w in all_data]

    # Build multi-level columns
    columns = []
    for workload in existing_workloads:
        loads = workload_loads[workload]
        for load in loads:
            columns.append((display_names[workload], str(load)))

    # Create MultiIndex for columns
    multi_columns = pd.MultiIndex.from_tuples(columns, names=['Workload', 'Load'])

    # Build data matrix
    data_matrix = []
    for algo in ALGORITHMS:
        row = []
        for workload in existing_workloads:
            if algo in all_data[workload]:
                row.extend(all_data[workload][algo])
            else:
                # Fill with NaN if algorithm not found
                row.extend([np.nan] * len(workload_loads[workload]))
        data_matrix.append(row)

    # Create DataFrame
    df = pd.DataFrame(data_matrix, index=ALGORITHMS, columns=multi_columns)

    return df


def create_combined_heatmap(df, category, scale_name, figsize=(13, 3)):
    """Create a combined heatmap with multi-level column headers."""
    # Set Times New Roman font globally
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    fig, ax = plt.subplots(figsize=figsize)

    # Convert to percentage
    df_percentage = df * 100

    # Create annotation matrix
    annot_matrix = df_percentage.map(lambda x: f'{x:.1f}%' if pd.notna(x) else '')

    # Get min and max for symmetric log normalization
    vmin = df_percentage.min().min()
    vmax = df_percentage.max().max()

    # Use SymLogNorm for better color distribution
    norm = mcolors.SymLogNorm(linthresh=10, linscale=0.5, vmin=vmin, vmax=vmax)

    # REVERSED colormap: Green for LOW values, Red for HIGH values
    cmap = 'RdYlGn_r'  # _r means reversed

    # More ticks for colorbar - logarithmic spacing, regular numbers
    colorbar_ticks = [-40, -20, -10, -5, 0, 5, 10, 20, 40, 80, 150, 300]

    # Create heatmap - disable default linewidths
    # pad=0.01 reduces distance between heatmap and colorbar
    hm = sns.heatmap(df_percentage, ax=ax, cmap=cmap, norm=norm,
                     annot=annot_matrix, fmt='',
                     cbar_kws={'label': 'Improvement (%)', 'ticks': colorbar_ticks, 'pad': 0.01},
                     annot_kws={"size": 14, "fontfamily": "serif"},
                     linewidths=0, linecolor='none')

    # Get workload groups
    workloads = df.columns.get_level_values(0).unique()
    n_rows = len(df)
    n_cols = len(df.columns)

    # Draw proper grid lines manually
    # Horizontal lines between rows
    for i in range(n_rows + 1):
        ax.axhline(y=i, color='white', linewidth=1)

    # Vertical lines for each cell
    for j in range(n_cols + 1):
        ax.axvline(x=j, color='white', linewidth=1)

    # Calculate workload boundaries and draw thick separator lines
    current_pos = 0
    workload_boundaries = [0]
    for workload in workloads:
        n_wl_cols = len([c for c in df.columns if c[0] == workload])
        current_pos += n_wl_cols
        workload_boundaries.append(current_pos)

    # Draw thick vertical lines between workload groups
    for pos in workload_boundaries[1:-1]:
        ax.axvline(x=pos, color='black', linewidth=1)

    # Draw border around entire heatmap
    ax.axhline(y=0, color='black', linewidth=2)
    ax.axhline(y=n_rows, color='black', linewidth=2)
    ax.axvline(x=0, color='black', linewidth=2)
    ax.axvline(x=n_cols, color='black', linewidth=2)

    # Remove bottom x-axis
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.tick_params(axis='x', bottom=False)

    # Calculate positions for workload labels and load labels
    workload_label_positions = []
    load_labels = []
    load_positions = []

    current_pos = 0
    for workload in workloads:
        cols_for_wl = [c for c in df.columns if c[0] == workload]
        n_wl_cols = len(cols_for_wl)

        # Workload label at center
        workload_label_positions.append((current_pos + n_wl_cols / 2, workload))

        # Load labels for each column
        for i, col in enumerate(cols_for_wl):
            load_positions.append(current_pos + i + 0.5)
            load_labels.append(str(col[1]))

        current_pos += n_wl_cols

    # Remove default top axis
    ax.tick_params(axis='x', top=False, labeltop=False)

    # Add workload labels at TOP of heatmap (increased gap from 1.08 to 1.14)
    for pos, label in workload_label_positions:
        ax.text(pos, 1.12, label, ha='center', va='bottom', fontsize=16,
                transform=ax.get_xaxis_transform(), fontfamily='serif')

    # Add load labels below workload labels (but still at top)
    for pos, label in zip(load_positions, load_labels):
        ax.text(pos, 1.02, label, ha='center', va='bottom', fontsize=14,
                transform=ax.get_xaxis_transform(), fontfamily='serif')
        # Add tick mark below load number
        ax.plot([pos, pos], [1.0, 1.015], color='black', linewidth=0.8,
                transform=ax.get_xaxis_transform(), clip_on=False)

    # Set y-axis properties - no bold
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.tick_params(axis='y', labelsize=16)
    for label in ax.get_yticklabels():
        label.set_fontfamily('serif')
        label.set_fontweight('normal')
    plt.yticks(rotation=0)

    # Set colorbar properties
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Improvement (%)', fontsize=16, fontfamily='serif')
    # Use regular number format (not scientific notation)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}' if x == int(x) else f'{x:.1f}'))
    # Remove minor ticks to avoid extra tick marks
    cbar.ax.yaxis.set_minor_locator(plt.NullLocator())
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('serif')

    return fig


def save_combined_heatmap(df, category, scale_num, scale_name, output_root):
    """Save the combined heatmap."""
    if df.empty:
        print(f"No data for {category} - {scale_name}")
        return

    # Create output directory
    output_dir = Path(output_root) / category
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create and save heatmap
    fig = create_combined_heatmap(df, category, scale_name)

    output_file = output_dir / f"{category}_{scale_name}.pdf"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_file}")


def main():
    """Main function to generate all combined heatmaps."""
    print("Starting combined heatmap generation...")

    # Process each scale
    for scale_num, scale_name in SCALES.items():
        print(f"\nProcessing {scale_name} scale (files ending with -{scale_num})...")

        # Process each category
        for category_idx, category in enumerate(CATEGORIES):
            print(f"  Processing {category}...")

            # Collect all data for this scale and category
            all_data, workload_loads = collect_all_data_for_scale(
                input_folder, scale_num, category_idx
            )

            if not all_data:
                print(f"    No data found for {category} - {scale_name}")
                continue

            # Build combined DataFrame
            df = build_combined_dataframe(all_data, workload_loads)

            # Save heatmap
            save_combined_heatmap(df, category, scale_num, scale_name, output_root)

    print("\nDone!")


if __name__ == "__main__":
    main()