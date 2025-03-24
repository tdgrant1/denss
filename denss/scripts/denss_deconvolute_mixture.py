import os
import sys, time
import re
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import denss
from functools import partial

# --- Define Fraction Model Functions (MUST COME FIRST) ---
def linear_fraction_model(frame_index, m, b):
    """Linear fraction model: f(frame) = m * frame + b"""
    return m * frame_index + b


def exponential_fraction_model(frame_index, A, rate_constant):
    """Exponential decay fraction model: f(frame) = A * exp(-rate_constant * frame)"""
    return A * np.exp(-rate_constant * frame_index)


def no_constraint_fraction_model(frame_index):
    """No constraint model: fraction is a free parameter."""
    return None  # Indicate that fraction is directly from parameters


def sigmoid_fraction_model(frame_index, L=1.0, k=0.2, frame_mid=25):
    """Sigmoidal (logistic) fraction model."""
    return L / (1 + np.exp(-k * (frame_index - frame_mid)))

def generate_evolving_mixture_stack(profiles, fractions):
    """Generate a 2D array of mixture profiles."""
    mixture_stack = fractions.T @ profiles
    return mixture_stack

def calculate_cosine_similarity(profile1, profile2):
    """Calculates the cosine similarity between two scattering profiles."""
    norm_profile1 = np.linalg.norm(profile1)
    norm_profile2 = np.linalg.norm(profile2)

    # print(f"  calculate_cosine_similarity: norm_profile1 = {norm_profile1:.6e}, norm_profile2 = {norm_profile2:.6e}") # DEBUG PRINT

    if norm_profile1 == 0 or norm_profile2 == 0:
        print("  calculate_cosine_similarity: WARNING - Zero norm detected in profile, returning NaN.") # DEBUG PRINT
        return np.nan  # Return NaN if either profile has zero norm

    cosine_similarity = np.dot(profile1, profile2) / (norm_profile1 * norm_profile2)
    # print(f"  calculate_cosine_similarity: cosine_similarity = {cosine_similarity:.6e}") # DEBUG PRINT
    return cosine_similarity

def parse_command_line_args():
    """Parses command line arguments using argparse and returns a dictionary of arguments."""
    import argparse
    import os  # Make sure os is imported here as well, if needed

    # --- Argument Parser Setup (Comprehensive Command Line Options) ---
    parser = argparse.ArgumentParser(description="Deconvolute evolving mixture SAXS data using Shannon expansion.")

    # Essential Required Input Arguments

    # Replace the existing mixture_stack_file argument
    parser.add_argument("-m", "--mixture_stack", required=True,
                        help="Path to mixture stack input. Can be: \n"
                             "1. A single data file (.dat, .txt, .npy)\n"
                             "2. A directory (all .dat files will be loaded)\n"
                             "3. A text file containing list of data files\n"
                             "4. A wildcard pattern (e.g., 'data/*.dat')\n"
                             "5. A template with placeholder (e.g., 'data_{}.dat')")

    # Add optional parameters to control behavior when needed
    parser.add_argument("--q_column", type=int, default=0,
                        help="Column index for q values in data files (default: 0)")
    parser.add_argument("--I_column", type=int, default=1,
                        help="Column index for intensity values in data files (default: 1)")
    parser.add_argument("--sort_method", default="natural",
                        choices=["natural", "alpha", "time", "numeric"],
                        help="Method to sort files when loading from directory (default: natural)")

    parser.add_argument("--interpolation_mode", default="auto",
                        choices=["none", "common", "union", "reference", "auto"],
                        help="How to handle files with different q-points: " +
                             "'none': require identical q-points, " +
                             "'common': use common q-range, " +
                             "'union': use all q-points, " +
                             "'reference': interpolate to first file, " +
                             "'auto': try common, then reference")

    # Only needed for template-based loading
    parser.add_argument("--range_start", type=int,
                        help="Starting frame number (only needed for template with {} placeholder)")
    parser.add_argument("--range_end", type=int,
                        help="Ending frame number (only needed for template with {} placeholder)")
    parser.add_argument("--range_step", type=int, default=1,
                        help="Step size between frames (default: 1)")
    parser.add_argument("--range_padding", type=int,
                        help="Zero-padding for frame numbers (e.g., 3 -> '001')")

    parser.add_argument("-q", "--q_values_file",
                        help="Path to file containing q values (must correspond to mixture_stack rows).")
    parser.add_argument("-d", "--d_values", help="Comma-separated D values for unknown components.")

    # Flexible Initial Guess Profiles Input
    parser.add_argument("-u", "--unknown_profiles_init_input", default="auto",
                        help="Comma-separated frame indices (integers), file paths (strings), or 'auto' for automatic initial guess profiles. Defaults to 'auto'.")

    # Flexible Known Profiles Input
    parser.add_argument("-k", "--known_profiles_files", default=None,
                        help="Comma-separated paths to known profiles (q, I). Optional.")
    parser.add_argument("--known_profile_types", default=None,
                        help="Comma-separated types of known profiles. Defaults to 'generic'.")
    parser.add_argument("--known_profile_names", default=None,
                        help="Comma-separated names for known profiles (for plotting). Optional.")

    # Optimization Settings (Optional, with defaults)
    parser.add_argument("--optimization_method", default='L-BFGS-B',
                        help="Optimization method (scipy.optimize.minimize). Default: L-BFGS-B")
    parser.add_argument("--maxiter", type=int, default=1e12, help="Maximum iterations for optimization. Default: 100")
    parser.add_argument("--ftol", type=float, default=1e-16,
                        help="Function tolerance for optimization convergence. Default: 1e-8")
    parser.add_argument("--maxfun", type=int, default=1e12, help="Maximum function evaluations. Default: 100000")
    parser.add_argument("--maxls", type=int, default=50, help="Maximum line search iterations. Default: 50")
    parser.add_argument("--eps", type=float, default=1e-8, help="Step size for numerical derivatives. Default: 1e-8")

    # Penalty Settings (Optional, with defaults)
    parser.add_argument("--fractions_weight", type=float, default=1.0,
                        help="Weight for fraction sum penalty. Default: 1.0")
    parser.add_argument("--profile_similarity_weight", type=float, default=10.0,
                        help="Weight for profile similarity penalty. Default: 10.0")
    parser.add_argument("--water_penalty_weight", type=float, default=0.0,
                        help="Weight for water penalty. Default: 0.0")
    parser.add_argument("--water_peak_range", default="1.9,2.0",
                        help="q-range for water peak penalty (e.g., '1.9,2.0'). Default: 1.9,2.0")
    parser.add_argument("--target_similarity", type=str, default=None,
                        help="Comma-separated target profile similarity to water for each unknown component. Optional.")

    # Output and Visualization Options (Optional)
    parser.add_argument("-o", "--output_params_file", default='optimization_params_flexible_fractions.npy',
                        help="Path to save optimized parameters (.npy file). Default: optimization_params_flexible_fractions.npy")
    parser.add_argument("-v", "--visualize", action="store_true",
                        help="Enable interactive visualization during optimization.")
    parser.add_argument("--plot_update_frequency", type=int, default=10,
                        help="Plot update frequency (iterations). Default: 10")
    parser.add_argument("--fit_frame_number", default=None, help="Frame number for plotting example fit.")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Enable verbose output. Default: True")  # Default verbose to True
    parser.add_argument("--no-verbose", action="store_false", dest='verbose',
                        help="Disable verbose output.")  # Option to disable verbose output

    args = parser.parse_args()

    # --- Convert argparse.Namespace to dictionary ---
    parsed_args_dict = vars(args) # Use vars() to convert Namespace to dict

    return parsed_args_dict # Return the dictionary


# Loading files for this requires flexibility and ease of use for the user, which is a tough combination to strike
# Here we will utilize multiple different functions for different possible use cases

def load_mixture_stack_from_directory(directory_path, q_column=0, I_column=1,
                                      file_pattern="*.dat", sort_method="natural"):
    """
    Load mixture stack from all matching files in a directory.

    Parameters:
    -----------
    directory_path : str
        Path to directory containing .dat files
    q_column : int
        Column index for q values (default: 0)
    I_column : int
        Column index for intensity values (default: 1)
    file_pattern : str
        Glob pattern to match files (default: "*.dat")
    sort_method : str
        Method to sort files: "natural" (human-like sorting),
        "alpha" (alphabetical), "time" (modification time),
        or "numeric" (extract numbers from filenames)

    Returns:
    --------
    tuple
        (q_values, mixture_stack) as numpy arrays
    """
    import glob
    import os
    import re

    # Get list of files matching pattern
    files = glob.glob(os.path.join(directory_path, file_pattern))

    if not files:
        raise ValueError(f"No files matching '{file_pattern}' found in {directory_path}")

    # Sort files based on requested method
    if sort_method == "alpha":
        files.sort()
    elif sort_method == "time":
        files.sort(key=os.path.getmtime)
    elif sort_method == "numeric":
        def extract_number(filename):
            numbers = re.findall(r'\d+', os.path.basename(filename))
            return int(numbers[0]) if numbers else 0

        files.sort(key=extract_number)
    elif sort_method == "natural":
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split(r'(\d+)', os.path.basename(s))]

        files.sort(key=natural_sort_key)
    else:
        raise ValueError(f"Unknown sort method: {sort_method}")

    # Load first file to get q values and determine dimensions
    first_data = np.genfromtxt(files[0], invalid_raise=False, usecols=(0, 1, 2))
    q_values = first_data[:, q_column]
    n_q_bins = len(q_values)
    n_frames = len(files)

    # Initialize mixture stack
    mixture_stack = np.zeros((n_frames, n_q_bins))

    # Load intensity data from each file
    for i, file_path in enumerate(files):
        data = np.genfromtxt(file_path, invalid_raise=False, usecols=(0, 1, 2))
        if data.shape[0] != n_q_bins:
            raise ValueError(f"File {file_path} has different number of q points than the first file")
        mixture_stack[i, :] = data[:, I_column]

    return q_values, mixture_stack


def load_mixture_stack_from_list(file_list_path, q_column=0, I_column=1):
    """
    Load mixture stack from a list of files specified in a text file.

    Parameters:
    -----------
    file_list_path : str
        Path to text file containing list of .dat files (one per line)
    q_column : int
        Column index for q values (default: 0)
    I_column : int
        Column index for intensity values (default: 1)

    Returns:
    --------
    tuple
        (q_values, mixture_stack) as numpy arrays
    """
    import os

    # Read the list of files
    with open(file_list_path, 'r') as f:
        files = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if not files:
        raise ValueError(f"No files listed in {file_list_path}")

    # Handle relative paths - make them relative to the list file location
    list_dir = os.path.dirname(os.path.abspath(file_list_path))
    files = [os.path.join(list_dir, f) if not os.path.isabs(f) else f for f in files]

    # Check all files exist
    missing_files = [f for f in files if not os.path.exists(f)]
    if missing_files:
        raise ValueError(f"Missing files: {', '.join(missing_files)}")

    # Load first file to get q values and determine dimensions
    first_data = np.genfromtxt(files[0], invalid_raise=False, usecols=(0, 1, 2))
    q_values = first_data[:, q_column]
    n_q_bins = len(q_values)
    n_frames = len(files)

    # Initialize mixture stack
    mixture_stack = np.zeros((n_frames, n_q_bins))

    # Load intensity data from each file
    for i, file_path in enumerate(files):
        data = np.genfromtxt(file_path, invalid_raise=False, usecols=(0, 1, 2))
        if data.shape[0] != n_q_bins:
            raise ValueError(f"File {file_path} has different number of q points than the first file")
        mixture_stack[i, :] = data[:, I_column]

    return q_values, mixture_stack


def load_mixture_stack_from_range(file_template, start, end, step=1,
                                  padding=None, q_column=0, I_column=1):
    """
    Load mixture stack from sequentially numbered files.

    Parameters:
    -----------
    file_template : str
        Template with {} placeholder for number (e.g., "data_{}.dat")
    start : int
        Starting frame number
    end : int
        Ending frame number (inclusive)
    step : int
        Step size between frames (default: 1)
    padding : int or None
        Zero-padding for frame numbers (e.g., 3 -> "001") or None for no padding
    q_column : int
        Column index for q values (default: 0)
    I_column : int
        Column index for intensity values (default: 1)

    Returns:
    --------
    tuple
        (q_values, mixture_stack) as numpy arrays
    """
    import os

    # Generate list of files
    files = []
    for i in range(start, end + 1, step):
        if padding is not None:
            num_str = str(i).zfill(padding)
        else:
            num_str = str(i)
        file_path = file_template.format(num_str)
        files.append(file_path)

    # Check all files exist
    missing_files = [f for f in files if not os.path.exists(f)]
    if missing_files:
        raise ValueError(f"Missing files: {', '.join(missing_files)}")

    # Load first file to get q values and determine dimensions
    first_data = np.genfromtxt(files[0], invalid_raise=False, usecols=(0, 1, 2))
    q_values = first_data[:, q_column]
    n_q_bins = len(q_values)
    n_frames = len(files)

    # Initialize mixture stack
    mixture_stack = np.zeros((n_frames, n_q_bins))

    # Load intensity data from each file
    for i, file_path in enumerate(files):
        data = np.genfromtxt(file_path, invalid_raise=False, usecols=(0, 1, 2))
        if data.shape[0] != n_q_bins:
            raise ValueError(f"File {file_path} has different number of q points than the first file")
        mixture_stack[i, :] = data[:, I_column]

    return q_values, mixture_stack


def detect_and_load_mixture_stack(input_path, q_values_file=None, q_column=0, I_column=1,
                                  sort_method="natural", range_start=None, range_end=None,
                                  range_step=1, range_padding=None):
    """
    Smart detection of input type and loading of mixture stack.

    Parameters:
    -----------
    input_path : str
        Path to mixture stack input (could be file, directory, list file, or pattern)
    q_values_file : str or None
        Path to file containing q values, or None to extract from data files
    q_column : int
        Column index for q values in data files (default: 0)
    I_column : int
        Column index for intensity values in data files (default: 1)
    sort_method : str
        Method for sorting files in directory mode
    range_start, range_end : int or None
        Start and end frame numbers for template mode
    range_step : int
        Step size between frames for template mode (default: 1)
    range_padding : int or None
        Zero-padding for frame numbers in template mode

    Returns:
    --------
    tuple
        (q_values, mixture_stack) as numpy arrays
    """
    import os
    import glob
    import numpy as np

    # First, check if we have a separate q_values file
    q_values = None
    if q_values_file and os.path.isfile(q_values_file):
        q_values = np.genfromtxt(q_values_file, invalid_raise=False, usecols=(0,))
        print(f"Loading q values from: {q_values_file}")

    # Case 1: Input is a .npy file
    if input_path.endswith('.npy') and os.path.isfile(input_path):
        mixture_stack = np.load(input_path)
        print(f"Loading .npy mixture stack from {input_path}")

        # If we already have q values, use those
        if q_values is not None:
            return q_values, mixture_stack
        else:
            # Otherwise assume first column is q
            q = mixture_stack[:, 0].copy()
            mixture_stack = mixture_stack[:, 1:]
            return q, mixture_stack

    # Case 2: Input is a directory
    if os.path.isdir(input_path):
        print(f"Detected directory input. Loading all .dat files from {input_path}")
        files = sorted(glob.glob(os.path.join(input_path, "*.dat")))
        if not files:
            raise ValueError(f"No .dat files found in directory: {input_path}")

        # Process files (using helper function)
        q, mixture_stack = process_data_files(
            files,
            q_values,
            q_column,
            I_column,
            sort_method,
            interpolation_mode=parsed_args.get("interpolation_mode", "auto")
        )
        return q, mixture_stack

    # Case 3: Input is a text file with list of files
    if os.path.isfile(input_path) and (input_path.endswith('.txt') or input_path.endswith('.list')):
        with open(input_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line and not first_line.startswith('#'):
                # Try relative path resolution
                base_dir = os.path.dirname(input_path)
                first_file = first_line
                if not os.path.isabs(first_line):
                    first_file = os.path.join(base_dir, first_line)

                if os.path.exists(first_file):
                    print(f"Detected list file input. Loading files listed in {input_path}")
                    with open(input_path, 'r') as f:
                        files = [line.strip() for line in f if line.strip() and not line.startswith('#')]

                    # Make relative paths absolute
                    files = [os.path.join(base_dir, f) if not os.path.isabs(f) else f for f in files]

                    # Process files
                    q, mixture_stack = process_data_files(files, q_values, q_column, I_column, sort_method)
                    return q, mixture_stack

    # Case 4: Input contains wildcard characters
    if '*' in input_path or '?' in input_path:
        matching_files = sorted(glob.glob(input_path))
        if matching_files:
            print(f"Detected wildcard pattern. Loading {len(matching_files)} matching files")
            q, mixture_stack = process_data_files(matching_files, q_values, q_column, I_column, sort_method)
            return q, mixture_stack

    # Case 5: Input contains curly braces placeholder
    if '{}' in input_path and range_start is not None and range_end is not None:
        print(f"Detected range template. Loading files from {range_start} to {range_end}")
        files = []
        for i in range(range_start, range_end + 1, range_step):
            if range_padding is not None:
                num_str = str(i).zfill(range_padding)
            else:
                num_str = str(i)
            file_path = input_path.format(num_str)
            if os.path.isfile(file_path):
                files.append(file_path)
            else:
                print(f"Warning: File not found: {file_path}")

        if not files:
            raise ValueError(f"No files found for template: {input_path} with range {range_start}-{range_end}")

        # Process files
        q, mixture_stack = process_data_files(files, q_values, q_column, I_column, sort_method)
        return q, mixture_stack

    # Case 6: Input is a single data file (fallback)
    if os.path.isfile(input_path):
        print(f"Loading single file mixture stack from {input_path}")
        mixture_stack = np.genfromtxt(input_path)

        # If we already have q values, use those
        if q_values is not None:
            return q_values, mixture_stack
        else:
            # Assume first column is q
            q = mixture_stack[:, 0].copy()
            mixture_stack = mixture_stack[:, 1:] if mixture_stack.ndim > 1 else np.array([mixture_stack[1:]])
            return q, mixture_stack

    # If we get here, we couldn't determine the input type
    raise ValueError(
        f"Could not determine input type for '{input_path}'. "
        "Please provide a valid file, directory, file list, or pattern."
    )


def load_saxs_file_generic(file_path, q_column=0, I_column=1):
    """
    A generic SAXS data file loader that adapts to different file formats.

    This function intelligently determines the data section of SAXS files
    by analyzing line patterns and content, handling various formats from
    different instruments and processing software.

    Parameters:
    -----------
    file_path : str
        Path to the SAXS data file
    q_column : int
        Index of the q column in the data section (default: 0)
    I_column : int
        Index of the intensity column in the data section (default: 1)

    Returns:
    --------
    tuple
        (q_values, intensities) as numpy arrays
    """
    # Read all lines from the file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Analysis variables
    data_lines = []
    numeric_pattern = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
    comment_markers = ['#', '!', '//', 'REMARK', '*', '/*', 'HEADER', '{']

    # Analyze file structure
    possible_data_blocks = []
    current_block = []
    current_block_column_count = None

    for i, line in enumerate(lines):
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Skip obvious comment or metadata lines
        if any(line.startswith(marker) for marker in comment_markers):
            continue

        # Check if line contains mostly numeric data
        fields = line.split()
        numeric_fields = 0

        # Skip lines with too few fields to contain q and I values
        if len(fields) < 2:
            continue

        # Count numeric fields
        for field in fields:
            if re.match(numeric_pattern, field):
                numeric_fields += 1

        # Consider a line to be data if at least 70% of fields are numeric
        is_data_line = numeric_fields >= max(2, int(0.7 * len(fields)))

        if is_data_line:
            # If starting a new block or continuing with same column count
            if not current_block or current_block_column_count == len(fields):
                current_block.append(line)
                current_block_column_count = len(fields)
            else:
                # Column count changed - end current block and start new one
                if len(current_block) > 5:  # Only keep blocks with sufficient lines
                    possible_data_blocks.append((current_block, current_block_column_count))
                current_block = [line]
                current_block_column_count = len(fields)

    # Add the last block if it exists
    if current_block and len(current_block) > 5:
        possible_data_blocks.append((current_block, current_block_column_count))

    # No valid data blocks found
    if not possible_data_blocks:
        raise ValueError(f"Could not identify data section in file: {file_path}")

    # Select the largest data block
    selected_block = max(possible_data_blocks, key=lambda x: len(x[0]))
    data_text = '\n'.join(selected_block[0])

    # Safety check for column indices
    col_count = selected_block[1]
    if max(q_column, I_column) >= col_count:
        raise ValueError(
            f"Requested columns ({q_column}, {I_column}) exceed available columns ({col_count}) in file: {file_path}")

    # Parse the data using StringIO
    from io import StringIO
    import numpy as np

    try:
        data = np.genfromtxt(StringIO(data_text), invalid_raise=False)

        # Extract q and intensity values
        q_values = data[:, q_column]
        intensities = data[:, I_column]

        # Validate data - check for NaN values
        if np.isnan(q_values).any() or np.isnan(intensities).any():
            # Find first non-NaN section and use that
            valid_indices = ~(np.isnan(q_values) | np.isnan(intensities))
            if np.sum(valid_indices) < 10:
                raise ValueError(f"Too few valid data points in file: {file_path}")

            q_values = q_values[valid_indices]
            intensities = intensities[valid_indices]

        return q_values, intensities

    except Exception as e:
        # If standard approach fails, try a more aggressive parsing
        try:
            # Try to extract just numeric values from each line
            all_numeric_values = []
            for line in selected_block[0]:
                numeric_vals = re.findall(numeric_pattern, line)
                all_numeric_values.append(numeric_vals)

            # Find mode of the number of values per line
            lengths = [len(vals) for vals in all_numeric_values]
            from collections import Counter
            most_common_length = Counter(lengths).most_common(1)[0][0]

            # Filter to lines with the most common length
            filtered_numeric_values = [vals for vals in all_numeric_values if len(vals) == most_common_length]

            # Convert to numpy array
            data_array = np.array(filtered_numeric_values, dtype=float)

            # Extract q and I columns safely
            safe_q_column = min(q_column, data_array.shape[1] - 1)
            safe_I_column = min(I_column, data_array.shape[1] - 1)

            return data_array[:, safe_q_column], data_array[:, safe_I_column]

        except Exception as nested_e:
            raise ValueError(
                f"Failed to parse file after multiple attempts: {file_path}\nOriginal error: {e}\nSecondary error: {nested_e}")

def process_data_files(files, q_values=None, q_column=0, I_column=1, sort_method="natural",
                       interpolation_mode="auto"):
    """
    Helper function to process a list of data files into a mixture stack.
    Now with support for files with different q-grids.

    Parameters:
    -----------
    files : list
        List of file paths to process
    q_values : ndarray or None
        Pre-loaded q values or None to extract from files
    q_column : int
        Column index for q values
    I_column : int
        Column index for intensity values
    sort_method : str
        Sorting method if needed
    interpolation_mode : str
        How to handle files with different q-points:
        - "none": Require all files to have identical q-points
        - "common": Find common q-range and truncate all files
        - "union": Use union of all q-points and interpolate
        - "reference": Interpolate all files to match first file
        - "auto": Try common first, then reference if needed

    Returns:
    --------
    tuple
        (q_values, mixture_stack) as numpy arrays
    """
    import numpy as np
    import re
    import os
    from scipy.interpolate import interp1d

    # Apply sorting if needed
    if sort_method == "alpha":
        files = sorted(files)
    elif sort_method == "time":
        files = sorted(files, key=os.path.getmtime)
    elif sort_method == "numeric":
        def extract_number(filename):
            numbers = re.findall(r'\d+', os.path.basename(filename))
            return int(numbers[0]) if numbers else 0

        files = sorted(files, key=extract_number)
    elif sort_method == "natural":
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower()
                    for text in re.split(r'(\d+)', os.path.basename(s))]

        files = sorted(files, key=natural_sort_key)

    # First survey all files to understand q-grid differences
    all_q_grids = []
    all_q_min = float('inf')
    all_q_max = float('-inf')
    common_q_min = float('-inf')
    common_q_max = float('inf')

    print(f"Surveying {len(files)} files to determine q-grid compatibility...")

    for file_path in files:
        try:
            # Use our generic parser instead of genfromtxt directly
            file_q, file_I = load_saxs_file_generic(file_path, q_column, I_column)

            # Track min/max for all files for union q-range
            all_q_min = min(all_q_min, file_q.min())
            all_q_max = max(all_q_max, file_q.max())

            # Track min/max for common q-range intersection
            common_q_min = max(common_q_min, file_q.min())
            common_q_max = min(common_q_max, file_q.max())

            all_q_grids.append(file_q)
        except Exception as e:
            print(f"Warning: Error reading file {file_path}: {e}")

    # Check if all q-grids are the same
    first_q_grid = all_q_grids[0]
    all_same = True

    for q_grid in all_q_grids[1:]:
        if len(q_grid) != len(first_q_grid) or not np.allclose(q_grid, first_q_grid):
            all_same = False
            break

    # If all q-grids are the same and "none" mode selected, proceed normally
    if all_same and (interpolation_mode == "none" or interpolation_mode == "auto"):
        print("All files have identical q-grids. Proceeding with direct loading.")

        # Use provided q_values or from first file
        if q_values is None:
            q_values = first_q_grid

        # Initialize mixture stack
        n_q_bins = len(q_values)
        n_frames = len(files)
        mixture_stack = np.zeros((n_frames, n_q_bins))

        # Load intensity data from each file
        for i, file_path in enumerate(files):
            # data = np.genfromtxt(file_path)
            data_q, data_I = load_saxs_file_generic(file_path, q_column, I_column)
            # mixture_stack[i, :] = data[:, I_column]
            mixture_stack[i, :] = data_I

        return q_values, mixture_stack

    # If q-grids differ or interpolation mode forced, handle with interpolation
    print(f"Files have different q-grids. Using '{interpolation_mode}' interpolation mode.")

    # Choose target q-grid based on selected mode
    if interpolation_mode == "reference" or (interpolation_mode == "auto" and (common_q_max <= common_q_min)):
        # Use first file's q-grid as reference
        target_q = first_q_grid
        print(
            f"Using reference q-grid from first file with {len(target_q)} points: {target_q[0]:.6f} to {target_q[-1]:.6f}")

    elif interpolation_mode == "common":
        # Use intersection of all q-ranges with spacing from first file
        if common_q_max <= common_q_min:
            print(f"Warning: No common q-range found. Falling back to reference mode.")
            target_q = first_q_grid
        else:
            # Find typical spacing from first file
            typical_spacing = np.median(np.diff(first_q_grid))
            # Create new q-grid with same spacing in common range
            n_points = int((common_q_max - common_q_min) / typical_spacing) + 1
            target_q = np.linspace(common_q_min, common_q_max, n_points)
            print(f"Using common q-range with {len(target_q)} points: {target_q[0]:.6f} to {target_q[-1]:.6f}")

    elif interpolation_mode == "union":
        # Use union of all q-ranges
        # Find typical spacing from all files
        all_spacings = []
        for q_grid in all_q_grids:
            if len(q_grid) > 1:
                all_spacings.append(np.median(np.diff(q_grid)))
        typical_spacing = np.median(all_spacings)

        # Create new q-grid with same spacing in union range
        n_points = int((all_q_max - all_q_min) / typical_spacing) + 1
        target_q = np.linspace(all_q_min, all_q_max, n_points)
        print(f"Using union q-range with {len(target_q)} points: {target_q[0]:.6f} to {target_q[-1]:.6f}")

    else:
        # Default to first file as reference
        target_q = first_q_grid
        print(f"Unknown interpolation mode '{interpolation_mode}'. Using reference q-grid.")

    # Initialize mixture stack with target q-grid
    n_q_bins = len(target_q)
    n_frames = len(files)
    mixture_stack = np.zeros((n_frames, n_q_bins))

    # Load and interpolate intensity data from each file
    print(f"Interpolating data from {len(files)} files to a common q-grid with {n_q_bins} points...")

    for i, file_path in enumerate(files):
        try:
            data = np.genfromtxt(file_path)
            file_q = data[:, q_column]
            file_I = data[:, I_column]

            # Create interpolation function
            interp_func = interp1d(
                file_q, file_I,
                kind='linear',
                bounds_error=False,
                fill_value=(file_I[0], file_I[-1])  # Extrapolate with end values
            )

            # Apply interpolation to get intensity on target q-grid
            interpolated_I = interp_func(target_q)

            # Check for NaN or inf values (can happen with extrapolation)
            if np.any(~np.isfinite(interpolated_I)):
                # Replace any non-finite values with nearest valid values
                mask = ~np.isfinite(interpolated_I)
                valid_indices = np.where(~mask)[0]
                if len(valid_indices) > 0:
                    for idx in np.where(mask)[0]:
                        nearest_valid_idx = valid_indices[np.argmin(np.abs(valid_indices - idx))]
                        interpolated_I[idx] = interpolated_I[nearest_valid_idx]
                else:
                    # If no valid values, use zeros
                    interpolated_I = np.zeros_like(interpolated_I)
                    print(f"Warning: File {file_path} has no valid intensity values after interpolation")

            # Store in mixture stack
            mixture_stack[i, :] = interpolated_I

            if i % 100 == 0 or i == len(files) - 1:
                print(f"  Processed {i + 1}/{len(files)} files")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            # Fill with zeros for this frame
            mixture_stack[i, :] = np.zeros(n_q_bins)

    return target_q, mixture_stack



def setup_deconvolution(parsed_args):
    """
    Parses command-line arguments, loads data files, preprocesses data,
    and returns a dictionary of parameters for ShannonDeconvolution class.
    Now removes hardcoded water.dat loading and relies on known profiles input for water profile.
    """

    # --- Data Loading from Parsed Arguments Dictionary ---
    q, mixture_stack = detect_and_load_mixture_stack(
        parsed_args["mixture_stack"],  # Your original argument name, may need to adjust
        q_values_file=parsed_args["q_values_file"],
        q_column=parsed_args.get("q_column", 0),  # Use .get() to handle missing keys
        I_column=parsed_args.get("I_column", 1),
        sort_method=parsed_args.get("sort_method", "natural"),
        range_start=parsed_args.get("range_start"),
        range_end=parsed_args.get("range_end"),
        range_step=parsed_args.get("range_step", 1),
        range_padding=parsed_args.get("range_padding")
    )

    n_frames = mixture_stack[0]

    d_values_str = parsed_args["d_values"].split(',')
    unknown_component_Ds = [float(d) for d in d_values_str]

    initial_sasrec_frames = None
    unknown_profiles_init_input = parsed_args["unknown_profiles_init_input"]

    unknown_component_profiles_iq_init = None  # Initialize as None, will be extracted or loaded below

    if unknown_profiles_init_input.lower() == "auto":
        initial_sasrec_frames = None
        unknown_component_profiles_iq_init = None  # Keep as None, auto-extract in class
        print("Using automatic frame selection for initial guess profiles.")
        if len(unknown_component_Ds) == 1:
            initial_sasrec_frames = [n_frames // 2]  # Middle frame for 1 component
        elif len(unknown_component_Ds) == 2:
            initial_sasrec_frames = [0, n_frames - 1]  # First and last for 2 components
        elif len(unknown_component_Ds) >= 3:
            initial_sasrec_frames = [int(round(f)) for f in np.linspace(0, n_frames - 1,
                                                                        len(unknown_component_Ds))]  # Evenly spaced for 3+
        else:
            initial_sasrec_frames = []  # Should not happen, but handle just in case
        print(f"  Automatically selected frames for initial Sasrecs: {initial_sasrec_frames}")

        if initial_sasrec_frames:  # If auto-selected frames are available, extract profiles from mixture_stack
            for frame_index in initial_sasrec_frames:
                initialization_profile = np.copy(mixture_stack[frame_index])
                initialization_Iq = np.vstack(
                    (q, initialization_profile, np.zeros_like(initialization_profile))).T  # Create (q, I, errors) array with zeros for errors
                unknown_component_profiles_iq_init.append(
                    initialization_Iq)  # Append NumPy array to the list


    elif unknown_profiles_init_input:
        initial_sasrec_frames = []
        unknown_component_profiles_iq_init_files_or_frames = [x.strip() for x in
                                                                unknown_profiles_init_input.split(
                                                                    ',')]  # More descriptive name
        unknown_component_profiles_iq_init = []  # Initialize list to store loaded profiles or frame-extracted profiles
        for input_item in unknown_component_profiles_iq_init_files_or_frames:  # More descriptive variable name
            try:
                frame_index = int(input_item)
                initial_sasrec_frames.append(frame_index)
                initialization_profile = np.copy(mixture_stack[frame_index])  # Extract profile from mixture stack
                initialization_Iq = np.vstack(
                    (q, initialization_profile, np.zeros_like(initialization_profile))).T  # Create (q, I, errors) array with zeros for errors
                unknown_component_profiles_iq_init.append(initialization_Iq)  # Append NumPy array to the list
                print(f"  Using frame {frame_index} from mixture stack for initial guess profile.")

            except ValueError:  # If not integer, assume it's a file path
                file_path = input_item
                initial_sasrec_frames.append(file_path)
                initialization_profile_iq = np.genfromtxt(file_path, invalid_raise=False, usecols=(0, 1, 2))  # Load profile from file
                unknown_component_profiles_iq_init.append(initialization_profile_iq)  # Append NumPy array (loaded from file)
                print(f"  Loading initial guess profile from file: {file_path}")
        print(f"Using user-provided initial Sasrec inputs (frames or files): {initial_sasrec_frames}")

    else:  # Should not happen, "auto" is default
        pass  # Auto frame selection already handled above

    known_profiles_files = parsed_args["known_profiles_files"]
    known_profiles_iq = None
    known_profile_types_str = parsed_args["known_profile_types"]
    known_profile_types = None
    known_profile_names_str = parsed_args["known_profile_names"]
    known_profile_names = None

    if known_profiles_files:
        known_profiles_files_list = known_profiles_files.split(',')
        known_profiles_iq = [np.genfromtxt(f, invalid_raise=False, usecols=(0, 1, 2)) for f in known_profiles_files_list]
        if known_profile_types_str:
            known_profile_types = known_profile_types_str.split(',')
        else:
            known_profile_types = ['generic'] * len(known_profiles_files_list)
        if known_profile_names_str:
            known_profile_names = known_profile_names_str.split(',')
        else:
            known_profile_names = [f"known_{i + 1}" for i in
                                   range(len(known_profiles_files_list))]
    else:
        known_profiles_iq = None
        known_profile_types = None
        known_profile_names = None

    water_I = None # Initialize water_I to None - it will now be dynamically found from known profiles if type='water'

    water_peak_range_str = parsed_args["water_peak_range"].split(',')
    water_peak_range = tuple(float(qr) for qr in water_peak_range_str)
    fractions_weight = parsed_args["fractions_weight"]
    profile_similarity_weight = parsed_args["profile_similarity_weight"]
    water_penalty_weight = parsed_args["water_penalty_weight"]

    target_similarity_str = parsed_args["target_similarity"]
    target_similarity = None
    if target_similarity_str:
        target_similarity = [float(ts) for ts in target_similarity_str.split(',')]

    optimization_options = {
        'maxiter': parsed_args["maxiter"],
        'ftol': parsed_args["ftol"],
        'maxfun': parsed_args["maxfun"],
        'maxls': parsed_args["maxls"],
        'eps': parsed_args["eps"]
    }

    params_dict = { # Create params_dict dictionary
        'mixture_stack': mixture_stack,
        'q': q,
        'unknown_component_Ds': unknown_component_Ds,
        'unknown_component_profiles_iq_init': unknown_component_profiles_iq_init, # Pass LOADED initial profiles (always NumPy arrays now)
        'known_profiles_iq': known_profiles_iq, # Pass loaded known profiles
        'known_profile_types': known_profile_types,
        'known_profile_names': known_profile_names,
        'alpha': 1e-8,
        'extrapolate_sasrec': False,
        'initial_sasrec_frames': initial_sasrec_frames,
        'water_peak_range': water_peak_range,
        'target_similarity': target_similarity,
        'fractions_weight': fractions_weight,
        'water_penalty_weight': water_penalty_weight,
        'profile_similarity_weight': profile_similarity_weight,
        'optimization_method': parsed_args["optimization_method"],
        'optimization_options': optimization_options,
        'use_basin_hopping': False,
        'basin_hopping_options': None,
        'parameter_bounds': None,
        'callback': None,
        'update_visualization': parsed_args["visualize"],
        'plot_update_frequency': parsed_args["plot_update_frequency"],
        'fit_frame_number': parsed_args["fit_frame_number"],
        'verbose': parsed_args["verbose"]
    }

    return params_dict # Return the params_dict dictionary

class ShannonDeconvolution:
    def __init__(self,
                 unknown_component_profiles_iq_init,  # NOW EXPECTS list of NumPy arrays (or None)
                 unknown_component_Ds,
                 mixture_stack,  # NOW EXPECTS NumPy array
                 q,  # NOW EXPECTS NumPy array
                 known_profiles_iq=None,  # NOW EXPECTS list of NumPy arrays (or None)
                 known_profile_names=None,
                 known_profile_types=None,
                 alpha=1e-8,
                 extrapolate_sasrec=False,
                 initial_sasrec_frames=None,
                 water_peak_range=(1.9, 2.0),
                 target_similarity=None,
                 fractions_weight=1.0,
                 water_penalty_weight=0.0,
                 profile_similarity_weight=10.0,
                 optimization_method='L-BFGS-B',
                 optimization_options=None,
                 use_basin_hopping=False,
                 basin_hopping_options=None,
                 parameter_bounds=None,
                 callback=None,
                 update_visualization=False,
                 plot_update_frequency=10,
                 fit_frame_number=None,
                 verbose=True):
        """
        Initializes the ShannonDeconvolution class.
        Now expects NumPy arrays and lists directly as arguments, no file paths.
        """
        # --- Input Validation (same as before) ---
        if mixture_stack is None or not isinstance(mixture_stack, np.ndarray) or mixture_stack.ndim != 2:
            raise ValueError("mixture_stack must be a 2D numpy array.")
        if q is None or not isinstance(q, np.ndarray) or q.ndim != 1:
            raise ValueError("q must be a 1D numpy array.")
        if unknown_component_Ds is None or not isinstance(unknown_component_Ds, list):
            raise ValueError("unknown_component_Ds must be a list of D values.")
        if len(unknown_component_Ds) < 1:
            raise ValueError("At least one D value must be provided for unknown components.")
        if initial_sasrec_frames is not None and not isinstance(initial_sasrec_frames, list):
            raise TypeError("initial_sasrec_frames must be a list of frame indices or file paths, or None/'auto'.")
        if known_profiles_iq is not None and not isinstance(known_profiles_iq, list):
            raise TypeError("known_profiles_iq must be a list of known profile arrays.")
        if known_profile_types is not None:
            if not isinstance(known_profile_types, list):
                raise TypeError("known_profile_types must be a list of strings.")
            if known_profiles_iq and len(known_profile_types) != len(known_profiles_iq):
                raise ValueError("Length of known_profile_types must match known_profiles_iq.")
        if known_profile_names is not None:
            if not isinstance(known_profile_names, list):
                raise TypeError("known_profile_names must be a list of strings.")
            if known_profiles_iq and len(known_profile_names) != len(known_profiles_iq):
                raise ValueError("Length of known_profile_names must match known_profiles_iq.")
        if target_similarity is not None and not isinstance(target_similarity, list):
            raise TypeError("target_similarity must be a list of floats (or None).")
        if optimization_options is not None and not isinstance(optimization_options, dict):
            raise TypeError("optimization_options must be a dictionary.")
        if basin_hopping_options is not None and not isinstance(basin_hopping_options, dict):
            raise TypeError("basin_hopping_options must be a dictionary.")
        if parameter_bounds is not None and not isinstance(parameter_bounds, list):
            raise TypeError("parameter_bounds must be a list of tuples.")
        if callback is not None and not callable(callback):
            raise TypeError("callback must be a callable function.")

        # --- Store parameters as attributes ---
        self.unknown_component_profiles_iq_init = unknown_component_profiles_iq_init  # NOW expects NumPy arrays directly
        self.unknown_component_Ds = unknown_component_Ds
        self.mixture_stack = mixture_stack  # NOW expects NumPy array directly
        self.n_frames = mixture_stack.shape[0]
        self.q = q  # NOW expects NumPy array directly
        self.n_q = len(q)
        self.known_profiles_iq = known_profiles_iq  # NOW expects NumPy arrays directly
        self.known_profile_names = known_profile_names
        self.known_profile_types = known_profile_types
        self.unknown_component_names = [f"component_{i + 1}" for i in range(len(unknown_component_Ds))]
        self.n_components = len(self.unknown_component_names)
        self.alpha = alpha
        self.extrapolate_sasrec = extrapolate_sasrec
        self.initial_sasrec_frames = initial_sasrec_frames
        self.water_peak_range = water_peak_range
        self.target_similarity = target_similarity
        self.fractions_weight = fractions_weight
        self.water_penalty_weight = water_penalty_weight
        self.profile_similarity_weight = profile_similarity_weight
        self.optimization_method = optimization_method
        self.optimization_options = optimization_options
        self.use_basin_hopping = use_basin_hopping
        self.basin_hopping_options = basin_hopping_options
        self.parameter_bounds = parameter_bounds
        self.callback = callback
        self.update_visualization = update_visualization
        self.plot_update_frequency = plot_update_frequency
        self.verbose = verbose

        # --- Initialize internal attributes ---
        self.unknown_component_sasrecs = []
        self.unknown_component_Bns = []
        self.unknown_component_nsh = []
        self.initial_params = None
        self.known_profiles_Iq = []
        self.unknown_component_profiles_iq_init_loaded = []
        self.plot_counter = 0
        if fit_frame_number is None:
            self.fit_frame_number = mixture_stack.shape[0] // 2
        else:
            self.fit_frame_number = int(fit_frame_number)

        # --- Initialization Steps (Data is assumed to be loaded and preprocessed already) ---
        self._load_profiles()
        self._initialize_known_profiles()  # Process known profiles (no file loading here anymore)
        self._preprocess_profiles() # Perform any necessary interpolation onto common q grid here
        self._initialize_sasrecs_and_params()  # Initialize Sasrecs (automatic frame selection still inside)
        self.parameter_bounds = self._create_parameter_bounds_In_only()

        # --- Initialize target similarity using the new method ---
        self._initialize_target_similarity()

        if self.verbose:
            print("ShannonDeconvolution object initialized.")
            print(f"  Unknown Components: {self.unknown_component_names}")
            if self.known_profile_names:
                print(f"  Known Profiles:     {self.known_profile_names}")
            if self.parameter_bounds is not None:
                print(f"  Parameter bounds are set with {len(self.parameter_bounds)} bounds.")
            if self.target_similarity is not None:  # Indicate if target_similarity penalty is enabled
                print("  Profile similarity penalty to water is ENABLED.")
            else:
                print("  Profile similarity penalty to water is DISABLED (no water profile type provided).")

        self.update_visualization = update_visualization

        if self.update_visualization:  # Initialize plots only if update_visualization is True
            # --- Plotting Initialization ---
            self._initialize_plot()

    def _load_profiles(self):
        """Loads known profiles from files and assigns types."""
        self.known_profiles_iq_loaded = []
        self.known_profile_names = self.known_profile_names or [f"known_{i + 1}" for i in range(
            len(self.known_profiles_iq or []))]  # Default known profile names, handle None case

        if self.known_profiles_iq is not None:  # Known profiles are optional
            if self.known_profile_types is None:
                self.known_profile_types = ['generic'] * len(
                    self.known_profiles_iq)  # Default to 'generic' types if not provided
            elif len(self.known_profile_types) != len(self.known_profiles_iq):
                raise ValueError("Length of known_profile_types must match known_profiles_iq.")
            if len(self.known_profile_names) != len(self.known_profiles_iq):
                raise ValueError("Length of known_profile_names must match known_profiles_iq.")

            for i_known_profile, profile_iq in enumerate(self.known_profiles_iq):
                profile_file_path = profile_iq  # now profile_iq is just the file path string from self.known_profiles_iq
                if isinstance(profile_file_path, str):
                    try:
                        loaded_profile = np.genfromtxt(profile_file_path, invalid_raise=False, usecols=(0, 1, 2))  # Load from file path
                        if loaded_profile.shape[1] < 2:
                            raise ValueError(
                                f"Known profile file '{profile_file_path}' must have at least 2 columns: [q, I, (errors), ...].")
                        self.known_profiles_iq_loaded.append(loaded_profile)  # Append loaded profile array
                        print(
                            f"  Loaded known profile '{self.known_profile_names[i_known_profile]}' from: {profile_file_path}, type: {self.known_profile_types[i_known_profile]}")
                    except FileNotFoundError:
                        raise FileNotFoundError(f"Known profile file not found: {profile_file_path}")
                    except Exception as e:  # Catch other potential loading errors
                        raise ValueError(f"Error loading known profile file '{profile_file_path}': {e}")

                elif isinstance(profile_file_path,
                                np.ndarray):  # For array inputs (less common via CLI, but for flexibility)
                    if profile_file_path.shape[1] < 2:
                        raise ValueError("Known profiles (arrays) must have at least 2 columns: [q, I, (errors), ...].")
                    self.known_profiles_iq_loaded.append(profile_file_path)
                    print(
                        f"  Using known profile '{self.known_profile_names[i_known_profile]}' from provided array, type: {self.known_profile_types[i_known_profile]}")

                else:
                    raise TypeError("Known profiles must be specified as file paths (string) or NumPy arrays.")
        else:
            self.known_profiles_iq_loaded = []  # Initialize as empty list if no known profiles provided
            self.known_profile_types = []
            self.known_profile_names = []

    def _interpolate_profile_to_q_grid(self, source_q, source_I, target_q, profile_name="profile"):
        """
        Helper function to interpolate a profile to a target q-grid.

        Parameters:
        -----------
        source_q : ndarray
            Source q values
        source_I : ndarray
            Source intensity values
        target_q : ndarray
            Target q values for interpolation
        profile_name : str
            Name of the profile for logging

        Returns:
        --------
        ndarray
            Interpolated intensity values
        """
        from scipy.interpolate import interp1d

        # Ensure q values are sorted
        sort_idx = np.argsort(source_q)
        source_q_sorted = source_q[sort_idx]
        source_I_sorted = source_I[sort_idx]

        # Check if q ranges are compatible
        if source_q_sorted[0] > target_q[0] or source_q_sorted[-1] < target_q[-1]:
            if self.verbose:
                print(
                    f"  WARNING: {profile_name} q-range ({source_q_sorted[0]:.5f}-{source_q_sorted[-1]:.5f}) does not fully cover target q-range ({target_q[0]:.5f}-{target_q[-1]:.5f})")
                print(f"  Extrapolation may cause artifacts in regions outside the profile's q-range.")

        # Use linear interpolation within range, nearest extrapolation outside
        interp_func = interp1d(source_q_sorted, source_I_sorted,
                               kind='linear',
                               bounds_error=False,
                               fill_value=(source_I_sorted[0], source_I_sorted[-1]))

        # Apply interpolation
        interpolated_I = interp_func(target_q)

        if self.verbose:
            print(f"  Successfully interpolated {profile_name} to {len(target_q)} points on target q-grid")

        return interpolated_I

    def _preprocess_profiles(self):
        """
        Preprocesses all profiles to ensure they're on the same q-grid as the mixture stack.
        This step happens before any other initialization to ensure consistent q-grids.
        """
        if self.verbose:
            print("\nPreprocessing profiles to ensure consistent q-grid...")
            print(f"Mixture stack q-grid: {len(self.q)} points from {self.q[0]:.5f} to {self.q[-1]:.5f}")

        # 1. Preprocess known profiles
        if self.known_profiles_iq_loaded:
            for i, profile_iq in enumerate(self.known_profiles_iq_loaded):
                profile_name = self.known_profile_names[i] if i < len(self.known_profile_names) else f"known_{i + 1}"

                # Extract q and I from the profile
                profile_q = profile_iq[:, 0]
                profile_I = profile_iq[:, 1]

                # Check if q-grid matches
                if len(profile_q) != len(self.q) or not np.allclose(profile_q, self.q):
                    if self.verbose:
                        print(f"  Interpolating known profile '{profile_name}' to mixture stack q-grid...")

                    # Interpolate to the mixture stack q-grid
                    interpolated_I = self._interpolate_profile_to_q_grid(
                        profile_q, profile_I, self.q, f"known profile '{profile_name}'")

                    # Replace the profile with interpolated version
                    # Create a new array with the mixture stack q values and interpolated intensities
                    interpolated_profile = np.zeros((len(self.q), profile_iq.shape[1]))
                    interpolated_profile[:, 0] = self.q
                    interpolated_profile[:, 1] = interpolated_I
                    if profile_iq.shape[1] > 2:  # If there are error values
                        # Interpolate errors too
                        profile_errors = profile_iq[:, 2]
                        interpolated_errors = self._interpolate_profile_to_q_grid(
                            profile_q, profile_errors, self.q, f"errors for known profile '{profile_name}'")
                        interpolated_profile[:, 2] = interpolated_errors

                    # Update the stored profile
                    self.known_profiles_iq_loaded[i] = interpolated_profile
                else:
                    if self.verbose:
                        print(f"  Known profile '{profile_name}' already on correct q-grid.")

        # 2. Preprocess unknown component initialization profiles
        if self.unknown_component_profiles_iq_init is not None:
            self.unknown_component_profiles_iq_init_loaded = []  # Clear any previous values

            for i, profile_iq in enumerate(self.unknown_component_profiles_iq_init):
                component_name = self.unknown_component_names[i] if i < len(
                    self.unknown_component_names) else f"component_{i + 1}"

                # Handle different input types
                if isinstance(profile_iq, np.ndarray):
                    # It's a preloaded array
                    profile_q = profile_iq[:, 0]
                    profile_I = profile_iq[:, 1]

                    # Check if q-grid matches
                    if len(profile_q) != len(self.q) or not np.allclose(profile_q, self.q):
                        if self.verbose:
                            print(f"  Interpolating initial profile for '{component_name}' to mixture stack q-grid...")

                        # Interpolate to the mixture stack q-grid
                        interpolated_I = self._interpolate_profile_to_q_grid(
                            profile_q, profile_I, self.q, f"initial profile for '{component_name}'")

                        # Create a new array with the mixture stack q values and interpolated intensities
                        interpolated_profile = np.zeros((len(self.q), 3))  # q, I, error
                        interpolated_profile[:, 0] = self.q
                        interpolated_profile[:, 1] = interpolated_I
                        interpolated_profile[:, 2] = np.ones_like(self.q)  # Default errors to 1.0

                        # If original profile had errors, interpolate those too
                        if profile_iq.shape[1] > 2:
                            profile_errors = profile_iq[:, 2]
                            interpolated_errors = self._interpolate_profile_to_q_grid(
                                profile_q, profile_errors, self.q, f"errors for '{component_name}'")
                            interpolated_profile[:, 2] = interpolated_errors

                        # Store the interpolated profile
                        self.unknown_component_profiles_iq_init_loaded.append(interpolated_profile)
                    else:
                        if self.verbose:
                            print(f"  Initial profile for '{component_name}' already on correct q-grid.")
                        # Still need to ensure it has the right format (q, I, error)
                        if profile_iq.shape[1] < 3:
                            # Add error column if missing
                            full_profile = np.zeros((len(self.q), 3))
                            full_profile[:, :profile_iq.shape[1]] = profile_iq
                            full_profile[:, 2] = np.ones_like(self.q)  # Default errors to 1.0
                            self.unknown_component_profiles_iq_init_loaded.append(full_profile)
                        else:
                            # It's already in the right format
                            self.unknown_component_profiles_iq_init_loaded.append(profile_iq)

                elif isinstance(profile_iq, (int, str)):
                    # It's a frame index or file path - will be handled in _initialize_sasrecs_and_params
                    self.unknown_component_profiles_iq_init_loaded.append(profile_iq)

                else:
                    # Unsupported type
                    if self.verbose:
                        print(
                            f"  WARNING: Unsupported type for initial profile of '{component_name}': {type(profile_iq)}")
                    self.unknown_component_profiles_iq_init_loaded.append(None)

        if self.verbose:
            print("Profile preprocessing complete.")

    def _initialize_sasrecs_and_params(self):
        """Initializes Sasrecs for unknown components and sets initial parameters."""
        n_components = len(self.unknown_component_Ds)

        # Handle automatic frame selection if needed
        if self.initial_sasrec_frames is None:
            # Default frames: evenly spaced
            initial_sasrec_frames = np.linspace(0, self.mixture_stack.shape[0] - 1, n_components)
            self.initial_sasrec_frames = [int(round(frame)) for frame in initial_sasrec_frames]
        elif len(self.initial_sasrec_frames) != n_components:
            raise ValueError("Length of initial_sasrec_frames must match the number of unknown components.")

        # Find water profile for subtraction (if available)
        water_profile_I = None
        for i_known_profile, profile_type in enumerate(self.known_profile_types):
            if profile_type == 'water':
                water_profile_I = self.known_profiles_Iq[i_known_profile]
                break

        # Process each component
        for i_component in range(n_components):
            # Get the initialization profile (already on the correct q-grid from _preprocess_profiles)
            if i_component < len(self.unknown_component_profiles_iq_init_loaded):
                initialization_source = self.unknown_component_profiles_iq_init_loaded[i_component]

                # Handle different input types
                if isinstance(initialization_source, np.ndarray):
                    # It's already a preprocessed array
                    initialization_Iq = initialization_source
                    if self.verbose:
                        print(f"  Using preprocessed profile for component {i_component + 1}")

                elif isinstance(initialization_source, (int, str)):
                    # It's a frame index or was a file path
                    if isinstance(initialization_source, int) or initialization_source.isdigit():
                        # It's a frame index
                        frame_index = int(initialization_source)
                        initialization_profile = np.copy(self.mixture_stack[frame_index])
                        if self.verbose:
                            print(f"  Using frame {frame_index} for component {i_component + 1}")
                    else:
                        # It must be a string but not a digit string - assume it was a file path
                        # We should have already loaded and preprocessed it, but just in case:
                        if self.verbose:
                            print(
                                f"  WARNING: File path found in initialization source, should have been preprocessed: {initialization_source}")
                        # Use a default frame as fallback
                        frame_index = self.initial_sasrec_frames[i_component]
                        initialization_profile = np.copy(self.mixture_stack[frame_index])

                    # Subtract water background if available
                    if water_profile_I is not None:
                        water_peak_mini = denss.find_nearest_i(self.q, self.water_peak_range[0])
                        water_peak_maxi = denss.find_nearest_i(self.q, self.water_peak_range[1])

                        initialization_profile_water_peak_int = np.mean(
                            initialization_profile[water_peak_mini:water_peak_maxi])
                        water_peak_int = np.mean(water_profile_I[water_peak_mini:water_peak_maxi])

                        if water_peak_int > 0:
                            water_background_sf = initialization_profile_water_peak_int / water_peak_int
                            initialization_profile -= water_profile_I * (water_background_sf * 0.99)

                    # Create a complete Iq array
                    initialization_Iq = np.vstack((
                        self.q,
                        initialization_profile,
                        np.ones_like(initialization_profile)  # Default errors to 1.0
                    )).T

                else:
                    # Unsupported type or None - use a default frame
                    if self.verbose:
                        print(
                            f"  WARNING: Unsupported initialization source type for component {i_component + 1}, using default frame")
                    frame_index = self.initial_sasrec_frames[i_component]
                    initialization_profile = np.copy(self.mixture_stack[frame_index])

                    # Create a complete Iq array
                    initialization_Iq = np.vstack((
                        self.q,
                        initialization_profile,
                        np.ones_like(initialization_profile)  # Default errors to 1.0
                    )).T

            else:
                # No initialization profile provided, use default frame
                if self.verbose:
                    print(f"  No initialization provided for component {i_component + 1}, using default frame")
                frame_index = self.initial_sasrec_frames[i_component]
                initialization_profile = np.copy(self.mixture_stack[frame_index])

                # Create a complete Iq array
                initialization_Iq = np.vstack((
                    self.q,
                    initialization_profile,
                    np.ones_like(initialization_profile)  # Default errors to 1.0
                )).T

            # Store the final initialization profile
            if i_component < len(self.unknown_component_profiles_iq_init_loaded):
                self.unknown_component_profiles_iq_init_loaded[i_component] = initialization_Iq
            else:
                self.unknown_component_profiles_iq_init_loaded.append(initialization_Iq)

            # Create the Sasrec object
            sasrec = denss.Sasrec(
                initialization_Iq,
                D=self.unknown_component_Ds[i_component],
                qc=self.q,
                alpha=0,
                extrapolate=False
            )
            self.unknown_component_sasrecs.append(sasrec)

            if self.verbose:
                print(f"  Component {self.unknown_component_names[i_component]}: Nsh = {sasrec.n}")

        # Extract B matrices and Shannon channels
        self.unknown_component_Bns = [sasrec.B for sasrec in self.unknown_component_sasrecs]
        self.unknown_component_nsh = [sasrec.n for sasrec in self.unknown_component_sasrecs]

        # Create initial parameters
        self.initial_params = self._dict2params_In_only(self._create_params_dict_In_only())

    def _initialize_known_profiles(self):
        """
        Initializes known profiles, extracting and interpolating I(q) values to match the mixture stack q-grid.
        """
        self.known_profiles_Iq = []
        if not self.known_profiles_iq_loaded:
            return  # Nothing to do if no known profiles were loaded

        # Print original profiles info
        if self.verbose:
            print("\nInterpolating known profiles to mixture stack q-grid:")
            print(f"  Mixture stack q-grid has {len(self.q)} points from {self.q[0]:.5f} to {self.q[-1]:.5f}")

        for i, profile_iq in enumerate(self.known_profiles_iq_loaded):
            profile_name = self.known_profile_names[i] if i < len(self.known_profile_names) else f"known_{i + 1}"
            profile_type = self.known_profile_types[i] if i < len(self.known_profile_types) else "generic"

            # Get q and I values from the loaded profile
            profile_q = profile_iq[:, 0]  # First column is q
            profile_I = profile_iq[:, 1]  # Second column is intensity

            # Print info about this profile before interpolation
            if self.verbose:
                print(
                    f"  Profile '{profile_name}' (type: {profile_type}) has {len(profile_q)} q-points from {profile_q[0]:.5f} to {profile_q[-1]:.5f}")

            # Check if q ranges are compatible
            if profile_q[0] > self.q[0] or profile_q[-1] < self.q[-1]:
                print(
                    f"  WARNING: Profile '{profile_name}' q-range ({profile_q[0]:.5f}-{profile_q[-1]:.5f}) does not fully cover mixture stack q-range ({self.q[0]:.5f}-{self.q[-1]:.5f})")
                print(f"  Extrapolation may cause artifacts in regions outside the profile's q-range.")

            # Interpolate to match the mixture stack q-grid
            from scipy.interpolate import interp1d

            # Ensure q values are sorted (just in case)
            sort_idx = np.argsort(profile_q)
            profile_q_sorted = profile_q[sort_idx]
            profile_I_sorted = profile_I[sort_idx]

            # Use linear interpolation for values inside the range
            # Use nearest extrapolation for values outside (safer than linear extrapolation)
            interp_func = interp1d(profile_q_sorted, profile_I_sorted,
                                   kind='linear',
                                   bounds_error=False,
                                   fill_value=(profile_I_sorted[0], profile_I_sorted[-1]))

            # Apply interpolation to get profile on mixture stack q-grid
            interpolated_I = interp_func(self.q)

            # Store the interpolated profile
            self.known_profiles_Iq.append(interpolated_I)

            # Print confirmation of successful interpolation
            if self.verbose:
                print(f"  Successfully interpolated '{profile_name}' to {len(self.q)} points on mixture stack q-grid")

    def _initialize_target_similarity(self):
        """Calculates and initializes the target_similarity, if not provided."""
        if self.target_similarity is None:  # Only calculate target_similarity if not provided via command line/config
            print("Calculating target_similarity...")  # Informative message
            n_components = len(self.unknown_component_Ds)
            target_similarity_calculated = np.zeros(n_components)
            initial_profiles = self._calculate_profiles_from_params(self.initial_params) # Get initial profiles from self
            water_I = None

            for i_known_profile, profile_type in enumerate(self.known_profile_types): # Find water profile dynamically
                if profile_type == 'water':
                    water_I = self.known_profiles_Iq[i_known_profile] # Found water profile
                    print("  _initialize_target_similarity: Using dynamically loaded water profile for target_similarity calculation.")
                    break

            if water_I is not None: # Calculate target similarity only if water profile is found
                for i_component in range(n_components):
                    profile_similarity_to_water = calculate_cosine_similarity(initial_profiles[i_component], water_I)
                    target_similarity_calculated[i_component] = profile_similarity_to_water
                    # target_similarity_calculated[i_component] += np.random.normal(loc=0, scale=0.0012)
                self.target_similarity = target_similarity_calculated # Assign calculated target_similarity to self
            else: # If no water profile, disable target similarity penalty by setting target_similarity to None
                self.target_similarity = None # Disable target similarity penalty
                print("Warning: No 'water' type known profile provided. Profile similarity penalty disabled.") # Warning message for disabled penalty
        else: # If target_similarity is provided via command line/config, use it directly
            print("Using user-provided target_similarity from command line/config.") # Informative message
            # if command line given target similarity is a value, use it directly
            # if it is a .dat file, calculate the similarity compared to water
            # if it is a .pdb file, calculate the profile first with pdb2mrc, then the similarity to water

    def _create_params_dict_In_only(self):
        """Creates initial parameter dictionary from Sasrec In values."""
        params_dict = {}
        params_dict['component_In'] = []
        for i in range(len(self.unknown_component_sasrecs)):
            params_dict['component_In'].append(self.unknown_component_sasrecs[i].In)
        return params_dict

    def _dict2params_In_only(self, params_dict):
        """Converts parameter dictionary back to params array."""
        params = []
        for i in range(len(params_dict['component_In'])):
            params.append(params_dict['component_In'][i])
        params = np.concatenate(params)
        return params

    def _params2dict_In_only(self, params):
        """Parses params array into a dictionary for In parameters."""
        params_dict = {}
        params_dict['component_In'] = []
        start_index = 0
        for i in range(len(self.unknown_component_nsh)):
            nsh = self.unknown_component_nsh[i]
            end_index = start_index + nsh
            params_dict['component_In'].append(params[start_index:end_index])
            start_index = end_index
        return params_dict

    def Ish2Iq(self, Ish, Bn):
        """Calculate I(q) from intensities at Shannon points."""
        I = 2 * np.einsum('n,nq->q', Ish, Bn)
        return I

    def Ish2P(self, Ish, Sn):
        """Calculate P(r) from intensities at Shannon points."""
        P = np.einsum('n,nr->r', Ish, Sn)
        return P

    def _calculate_profiles_from_params(self, params):
        """
        Calculates estimated profiles from the parameter vector.

        Args:
            params (np.ndarray): Parameter vector containing Shannon coefficients (In)
                                 for unknown components.

        Returns:
            np.ndarray: Estimated profiles matrix (n_total_profiles x n_qbins).
        """
        n_qbins = self.mixture_stack.shape[1]
        n_unknown_components = len(self.unknown_component_nsh)
        n_known_profiles = len(self.known_profiles_Iq)
        n_total_profiles = n_unknown_components + n_known_profiles
        estimated_profiles = np.zeros((n_total_profiles, n_qbins))

        # 1. Parse parameters (Shannon coefficients for unknown components)
        params_dict = self._params2dict_In_only(params)

        # 2. Calculate profiles:
        #   a. Reconstruct "unknown" component profiles from In
        for i_unknown in range(n_unknown_components):
            Ish = params_dict['component_In'][i_unknown]
            Bn = self.unknown_component_Bns[i_unknown]
            estimated_profiles[i_unknown] = self.Ish2Iq(Ish, Bn)

        #   b. Use "known" profiles directly
        for i_known in range(n_known_profiles):
            estimated_profiles[n_unknown_components + i_known] = self.known_profiles_Iq[i_known]

        return estimated_profiles

    def _calculate_fractions(self, estimated_profiles):
        """
        Calculates estimated fractions for each frame using non-negative least squares.

        Args:
            estimated_profiles (np.ndarray): Estimated profiles matrix (n_total_profiles x n_qbins).

        Returns:
            np.ndarray: Estimated fractions matrix (n_total_profiles x n_frames).
        """
        n_frames = self.mixture_stack.shape[0]
        estimated_fractions = np.zeros((len(estimated_profiles), n_frames))  # Initialize fractions here

        # 3. Least Squares Fraction Calculation for each frame (moved from calc_score_from_In)
        estimated_profiles_for_lstsq = estimated_profiles.T  # Transpose for lstsq
        for i_frame in range(n_frames):
            target_data_frame = self.mixture_stack[i_frame, :]
            fractions_frame, residuals_frame_lstsq = optimize.nnls(estimated_profiles_for_lstsq, target_data_frame)
            estimated_fractions[:, i_frame] = fractions_frame

        return estimated_fractions

    def _calculate_residuals(self, calculated_mixture):
        """
        Calculates residuals between the mixture stack and the calculated mixture.

        Args:
            calculated_mixture (np.ndarray): Calculated mixture stack (N_frames x N_qbins).

        Returns:
            float: Sum of residuals, normalized by the number of frames.
        """
        n_frames = self.mixture_stack.shape[0]
        residuals = np.sum(np.sqrt((self.mixture_stack - calculated_mixture)**2))
        if residuals < 10: # Switch to log scale residuals when residuals are good
            residuals = np.sum(np.sqrt((np.log10(self.mixture_stack) - np.log10(calculated_mixture)) ** 2))
        residuals /= n_frames
        return residuals

    def _calculate_fraction_sum_penalty(self, estimated_fractions):
        """
        Calculates the sum-to-one penalty for the unknown component fractions.

        Args:
            estimated_fractions (np.ndarray): Estimated fractions matrix (n_total_profiles x n_frames).

        Returns:
            float: Sum-to-one fraction penalty value.
        """
        n_frames = self.mixture_stack.shape[0]
        n_unknown_components = len(self.unknown_component_nsh)

        fractions_sum_per_frame = np.sum(estimated_fractions[:n_unknown_components], axis=0)
        fractions_error = np.sum(np.sqrt((np.ones(n_frames) - fractions_sum_per_frame)**2))
        fractions_weight = self.fractions_weight  # Use the class attribute for weight
        fraction_sum_penalty = fractions_error * fractions_weight
        return fraction_sum_penalty

    def _calculate_water_penalty(self, estimated_fractions):
        """Calculates the water background intensity penalty."""
        water_penalty = 0.0

        # First check if any profile has type='water' - if not, return 0 immediately
        has_water_profile = False
        for profile_type in self.known_profile_types:
            if profile_type == 'water':
                has_water_profile = True
                break

        if not has_water_profile or self.water_penalty_weight <= 0:
            return 0.0  # No water profile or zero weight, skip calculation

        if self.water_peak_range is not None and self.water_penalty_weight > 0:
            n_frames = self.mixture_stack.shape[0]
            n_unknown_components = len(self.unknown_component_nsh)

            # Find indices for water peak range
            water_peak_mini_index = denss.find_nearest_i(self.q, self.water_peak_range[0])
            water_peak_maxi_index = denss.find_nearest_i(self.q, self.water_peak_range[1])

            # Ensure we have a valid range
            if water_peak_mini_index >= water_peak_maxi_index:
                if self.verbose and self.plot_counter == 0:  # Only print once
                    print(
                        f"WARNING: Invalid water peak range. Min index ({water_peak_mini_index}) >= Max index ({water_peak_maxi_index})")
                return 0.0  # Skip water penalty calculation

            # Ensure the range contains at least one point
            if water_peak_maxi_index - water_peak_mini_index < 1:
                if self.verbose and self.plot_counter == 0:  # Only print once
                    print(f"WARNING: Water peak range too narrow. Needs at least one q point.")
                return 0.0  # Skip water penalty calculation

            # Safely calculate mean water intensity in the mixture stack
            try:
                target_water_int = np.nanmean(self.mixture_stack[:, water_peak_mini_index:water_peak_maxi_index],
                                              axis=1)
            except Exception as e:
                if self.verbose and self.plot_counter == 0:  # Only print once
                    print(f"WARNING: Error calculating target water intensity: {e}")
                return 0.0  # Skip water penalty calculation

            # Initialize water intensity array
            calculated_water_int = np.zeros(n_frames)
            water_profile_I = None  # Initialize water_profile_I to None

            # Find water profile (if available)
            for i_known_profile, profile_type in enumerate(self.known_profile_types):
                if profile_type == 'water':
                    water_profile_I = self.known_profiles_Iq[i_known_profile]
                    break

            if water_profile_I is not None:
                try:
                    # Calculate water component contribution
                    water_stack_component = water_profile_I[None, :] * estimated_fractions[n_unknown_components + 0][:,
                                                                       None]
                    # Safely calculate mean water intensity in calculated mixture
                    calculated_water_int = np.nanmean(
                        water_stack_component[:, water_peak_mini_index:water_peak_maxi_index], axis=1)

                    # Safely calculate water error (handling potential NaN values)
                    water_error = np.nansum(np.sqrt(np.square(np.nan_to_num(target_water_int - calculated_water_int))))
                    water_penalty = water_error * self.water_penalty_weight

                    # If we still have NaN, set penalty to zero
                    if np.isnan(water_penalty):
                        if self.verbose and self.plot_counter == 0:  # Only print once
                            print("WARNING: NaN detected in water penalty calculation. Setting penalty to zero.")
                        water_penalty = 0.0

                except Exception as e:
                    if self.verbose and self.plot_counter == 0:  # Only print once
                        print(f"WARNING: Error in water penalty calculation: {e}")
                    water_penalty = 0.0

        return water_penalty

    def _calculate_profile_similarity_penalty(self, estimated_profiles):
        """Calculates the profile similarity penalty (similarity to water)."""
        # First check if target_similarity is set and if any profile has type='water'
        if self.target_similarity is None or self.profile_similarity_weight <= 0:
            return 0.0  # Skip calculation if target_similarity not set or zero weight

        has_water_profile = False
        for profile_type in self.known_profile_types:
            if profile_type == 'water':
                has_water_profile = True
                break

        if not has_water_profile:
            return 0.0  # No water profile, skip calculation

        if self.target_similarity is not None: # water_profile_iq_loaded check removed - check only target_similarity now
            n_unknown_components = len(self.unknown_component_nsh)
            profile_similarity_to_water = np.zeros(n_unknown_components)
            water_I = None # Initialize water_I to None

            for i_known_profile, profile_type in enumerate(self.known_profile_types): # Find water profile
                if profile_type == 'water':
                    water_I = self.known_profiles_Iq[i_known_profile] # Found water profile, extract I(q) values
                    # print(f"  _calculate_profile_similarity_penalty: Found water profile for similarity penalty.") # DEBUG PRINT
                    break # Assuming only one water profile, exit loop once found

            if water_I is not None: # Proceed only if water profile is found
                # print(f"  _calculate_profile_similarity_penalty: Water profile norm = {np.linalg.norm(water_I):.6e}") # DEBUG PRINT
                for i_component in range(n_unknown_components):
                    estimated_profile = estimated_profiles[i_component]
                    # print(f"  _calculate_profile_similarity_penalty: Component {i_component} profile norm = {np.linalg.norm(estimated_profile):.6e}") # DEBUG PRINT
                    profile_similarity_to_water[i_component] = calculate_cosine_similarity(estimated_profile, water_I) # Calculate similarity
                    # print(f"  _calculate_profile_similarity_penalty: Component {i_component} similarity to water = {profile_similarity_to_water[i_component]:.6e}") # DEBUG PRINT

                profile_similarity_error = np.sum(np.sqrt((profile_similarity_to_water - self.target_similarity)**2)) # Calculate error

        profile_similarity_penalty = profile_similarity_error * self.profile_similarity_weight # Apply weight
        return profile_similarity_penalty

    def calc_score_from_In(self, params, update_viz=False):
        """Calculates the score function from In parameters."""
        n_frames = self.mixture_stack.shape[0]
        # print(params[:5])
        estimated_profiles = self._calculate_profiles_from_params(params) # Get profiles
        estimated_fractions = self._calculate_fractions(estimated_profiles) # Get fractions

        # 4. Calculate Mixture
        calculated_mixture = generate_evolving_mixture_stack(profiles=estimated_profiles, fractions=estimated_fractions)

        # 5. Calculate Residuals
        residuals = self._calculate_residuals(calculated_mixture)

        # --- 6. Penalties ---
        # a. Sum-to-One Fraction Constraint Penalty
        fraction_sum_penalty = self._calculate_fraction_sum_penalty(estimated_fractions)

        # b. Water Penalty
        water_penalty = self._calculate_water_penalty(estimated_fractions)

        # c. Profile Similarity Penalty (using the new method)
        profile_similarity_penalty = self._calculate_profile_similarity_penalty(estimated_profiles)

        # 7. Total Score (rest of the original score aggregation and output)
        score = residuals + fraction_sum_penalty + water_penalty + profile_similarity_penalty

        if self.verbose:
            sys.stdout.write(f"\rStep {self.plot_counter:06d} {residuals:.6e} {fraction_sum_penalty:.6e} {water_penalty:.6e} {profile_similarity_penalty:.6e} {score:.6e} ") # Updated verbose output for profile_similarity_penalty
            sys.stdout.flush()

        # 8. Visualization Update (placeholder for now)
        self.plot_counter += 1
        if update_viz and self.update_visualization and (self.plot_counter % self.plot_update_frequency == 0):
            self.update_plot_simple(estimated_profiles, estimated_fractions) # Placeholder call

        return score

    def _create_parameter_bounds_In_only(self):
        """
        Creates bounds for In parameters only (non-negative Shannon intensities).

        Returns:
            list of tuples: Bounds for In parameters.
        """
        n_components = len(self.unknown_component_nsh) # Use self.unknown_component_nsh
        bounds = []
        # Bounds for Shannon intensities (In)
        for i in range(n_components):
            for _ in range(self.unknown_component_nsh[i]): # Use self.unknown_component_nsh[i]
                bounds.append((0.0, None))  # Non-negative intensities
        if self.verbose: # Add verbose output if desired
            print(f"Parameter bounds created for {sum(self.unknown_component_nsh)} parameters.")
        return bounds

    def _initialize_plot(self):
        """Initializes the matplotlib plots for visualization."""
        global fig, ax
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))

        # global line_true_profiles_noisy, line_true_profiles, line_fitted_profiles
        global line_true_profiles_noisy, line_fitted_profiles
        line_true_profiles_noisy = []
        # line_true_profiles = []
        colors = ['blue', 'red', 'forestgreen', 'purple', 'gold', 'darkcyan']  # Colors for components
        markers = ['o', '^', 's', 'v', '*', 'D']  # Markers for noisy data

        # Static data (reference data - True profiles)
        ax[0, 0].set_title("I(q) Profiles")  # Generic title
        ax[0, 0].set_ylabel("I(q)")
        ax[0, 0].set_xlabel("q")
        ax[0, 0].semilogy()
        ax[0, 0].grid(True)

        profile_handles = []  # List to collect legend handles
        profile_labels = []  # List to collect legend labels

        # Plot True Noisy and True Profiles (if available) - Dynamic Legends
        if self.unknown_component_profiles_iq_init_loaded is not None:  # Check if true profiles are available
            for i_component in range(len(self.unknown_component_names)):
                component_name = self.unknown_component_names[i_component]
                noisy_line, = ax[0, 0].plot(self.q, self.unknown_component_profiles_iq_init_loaded[i_component][:, 1],
                                            marker=markers[i_component], linestyle='None', color='gray', alpha=0.3,
                                            mec='none', ms=4, label=f'Initial {component_name}')
                profile_handles.extend([noisy_line])  # Add handles for legend
                profile_labels.extend([f'Initial {component_name}'])

        # Dynamic data - Fitted Profiles (using dynamic component names)
        line_fitted_profiles = []  # List to hold lines for fitted profiles
        for i_component in range(len(self.unknown_component_names)):
            component_name = self.unknown_component_names[i_component]
            fitted_line, = ax[0, 0].plot(self.q, np.zeros_like(self.q), color=colors[i_component],
                                        label=f'Fitted {component_name} (D={self.unknown_component_Ds[i_component]})')
            line_fitted_profiles.append(fitted_line)  # Store fitted profile line handles
            profile_handles.append(fitted_line)  # Add fitted profile handle to legend
            profile_labels.append(f'Fitted {component_name} (D={self.unknown_component_Ds[i_component]})')  # Add fitted profile label to legend

        ax[0, 0].legend(profile_handles, profile_labels, loc='upper right')  # Create legend with dynamic labels

        # Fractions Subplot - Dynamic Labels and Legend
        global line_fractions_target, line_fractions_fitted  # Renamed to be more generic
        line_fractions_target = []  # List for target fraction lines
        line_fractions_fitted = []  # List for fitted fraction lines

        ax[0, 1].set_title("Fractions")  # Generic title
        ax[0, 1].set_ylabel("Fractions")
        ax[0, 1].set_xlabel("Frames")
        ax[0, 1].set_ylim([-0.1, 1.2])
        ax[0, 1].grid(True)

        fractions_handles = []  # List for fraction legend handles
        fractions_labels = []  # List for fraction legend labels

        for i_component in range(len(self.unknown_component_names)):  # Plot fitted fractions for unknown components
            component_name = self.unknown_component_names[i_component]
            fitted_fraction_line, = ax[0, 1].plot(np.arange(self.n_frames), np.zeros(self.n_frames),
                                                label=f'{component_name} fractions', color=colors[i_component])
            line_fractions_fitted.append(fitted_fraction_line)  # Store fitted fraction lines
            fractions_handles.append(fitted_fraction_line)  # Add fitted fraction lines to legend handles
            fractions_labels.append(f'{component_name} fractions')  # Add fitted fraction lines to legend labels

        if self.known_profile_names is not None:  # Plot fitted fractions for known profiles
            for i_known_profile in range(len(self.known_profile_names)):
                known_profile_name = self.known_profile_names[i_known_profile]
                fitted_fraction_line, = ax[0, 1].plot(np.arange(self.n_frames), np.zeros(self.n_frames),
                                                    label=f'{known_profile_name} fractions',
                                                    color=colors[self.n_components + i_known_profile])  # Use colors after component colors
                line_fractions_fitted.append(fitted_fraction_line)  # Store fitted fraction lines (using same list)
                fractions_handles.append(fitted_fraction_line)  # Add fitted fraction lines to legend handles
                fractions_labels.append(f'{known_profile_name} fractions')  # Add fitted fraction lines to legend labels

        ax[0, 1].legend(fractions_handles, fractions_labels, loc='upper center')  # Create fractions legend with dynamic labels

        # Third subplot - Example fit to data
        global line_example_fit_data, line_example_fit_total, line_example_fit_components  # Renamed
        line_example_fit_components = []  # List for example fit component lines

        ax[1, 0].set_title(f"Example Fit - Frame {self.fit_frame_number}")  # Generic title
        ax[1, 0].set_ylabel("I(q)")
        ax[1, 0].set_xlabel("q")
        ax[1, 0].semilogy()
        ax[1, 0].grid(True)

        example_fit_handles = []  # Legend handles for example fit
        example_fit_labels = []  # Legend labels for example fit
        line_example_fit_data, = ax[1, 0].plot(self.q, self.mixture_stack[self.fit_frame_number], 'o', color='gray', alpha=0.3,
                                            mec='none', ms=4, label=f'Frame {self.fit_frame_number} Data')  # Data line - generic label
        example_fit_handles.append(line_example_fit_data)  # Add data handle to legend
        example_fit_labels.append(f'Frame {self.fit_frame_number} Data')  # Add data label to legend

        line_example_fit_total, = ax[1, 0].plot(self.q, np.zeros(self.n_q), color='black', linestyle='-', label='Total Fit')  # Total fit - generic label
        example_fit_handles.append(line_example_fit_total)  # Add total fit handle
        example_fit_labels.append('Total Fit')  # Add total fit label

        # Plot Example Fit Components - Dynamic Labels
        for i_component in range(len(self.unknown_component_names)):
            component_name = self.unknown_component_names[i_component]
            component_line, = ax[1, 0].plot(self.q, np.zeros(self.n_q), color=colors[i_component],
                                            label=f'Fitted {component_name}')
            line_example_fit_components.append(component_line)  # Store component fit lines
            example_fit_handles.append(component_line)  # Add component fit handles to legend
            example_fit_labels.append(f'Fitted {component_name}')  # Component fit labels

        if self.known_profile_names is not None:  # Plot Example Fit for known profiles
            for i_known_profile in range(len(self.known_profile_names)):
                known_profile_name = self.known_profile_names[i_known_profile]
                component_line, = ax[1, 0].plot(self.q, np.zeros(self.n_q), color=colors[self.n_components + i_known_profile],
                                                label=f'Fitted {known_profile_name}')  # Use colors after component colors
                line_example_fit_components.append(component_line)  # Store component fit lines (using same list)
                example_fit_handles.append(component_line)  # Add component fit handles to legend
                example_fit_labels.append(f'Fitted {known_profile_name}')  # Component fit labels

        ax[1, 0].legend(example_fit_handles, example_fit_labels, loc='upper center')  # Create example fit legend with dynamic labels

        # Residuals Subplot - Same as before
        global residuals_img
        initial_residuals = np.zeros((self.n_frames, len(self.q)))
        residuals_img = ax[1, 1].imshow(initial_residuals, aspect='auto', cmap='viridis',
                                        origin='lower', interpolation='nearest')
        ax[1, 1].set_title("Residuals")
        ax[1, 1].set_xlabel("q bin")
        ax[1, 1].set_ylabel("Frame")
        fig.colorbar(residuals_img, ax=ax[1, 1], label="Residual Magnitude")

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

        global plot_counter
        plot_counter = 0
        global last_update_time
        last_update_time = time.time()

    def update_plot_simple(self, estimated_profiles, estimated_fractions):
        """Updates the plot with current profiles and fractions."""
        global line_fitted_profiles, line_fractions_fitted, line_example_fit_components  # Updated globals

        # Update Fitted Profiles Data (using line_fitted_profiles list)
        for i_component in range(len(self.unknown_component_names)):
            line_fitted_profiles[i_component].set_ydata(estimated_profiles[i_component])

        # Update Fitted Fractions Data (using line_fractions_fitted list)
        for i_component in range(len(self.unknown_component_names)):  # Update fractions for unknown components
            line_fractions_fitted[i_component].set_ydata(estimated_fractions[i_component])

        # Update Fitted Fractions Data for known profiles (if any)
        if self.known_profile_names:
            for i_known_profile in range(len(self.known_profile_names)):
                line_fractions_fitted[len(self.unknown_component_names) + i_known_profile].set_ydata(
                    estimated_fractions[len(self.unknown_component_names) + i_known_profile])

        # Calculate and update the residuals image (same as before)
        calculated_mixture = generate_evolving_mixture_stack(
            profiles=estimated_profiles,
            fractions=estimated_fractions
        )
        frame_residuals = self.mixture_stack - calculated_mixture

        # Update Example Fit Data (using line_example_fit_components list)
        line_example_fit_total.set_ydata(calculated_mixture[self.fit_frame_number])
        for i_component in range(
                len(self.unknown_component_names)):  # Update component fit lines for unknown components
            line_example_fit_components[i_component].set_ydata(
                estimated_profiles[i_component] * estimated_fractions[i_component][self.fit_frame_number])

        if self.known_profile_names:  # Update component fit lines for known profiles (if any)
            for i_known_profile in range(len(self.known_profile_names)):
                line_example_fit_components[len(self.unknown_component_names) + i_known_profile].set_ydata(
                    estimated_profiles[len(self.unknown_component_names) + i_known_profile] *
                    estimated_fractions[len(self.unknown_component_names) + i_known_profile][self.fit_frame_number])

        # Update the image data (same as before)
        residuals_img.set_array(frame_residuals)
        rstd = np.std(frame_residuals)
        residuals_img.set_clim(vmin=-rstd, vmax=rstd)

        # Use draw_idle which is more efficient than full draw()
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    def run_optimization(self):
        """Runs the optimization process using scipy.optimize.minimize."""
        if self.verbose:
            print("\nStarting optimization...")

        # 1. Define objective function (partial function to pass fixed arguments)
        objective_function = partial(self.calc_score_from_In, update_viz=self.update_visualization) # Pass update_viz flag

        # 2. Optimization options
        optimization_options = self.optimization_options or {} # Use user-provided options or default to empty dict
        # default_options = {
        #     'maxiter': 100,     # Default max iterations (adjust as needed for testing)
        #     'maxfun': 1e5,
        #     'ftol': 1e-8,
        #     'maxls': 50,
        #     'eps': 1e-8
        # }
        # optimization_options = {**default_options, **optimization_options} # Merge default and user options, user options take precedence

        print(optimization_options)

        # 3. Callback function (initially None or a placeholder)
        callback = self.callback # Use callback from __init__, can be None

        # 4. Run optimization using scipy.optimize.minimize
        results = optimize.minimize(
            objective_function,      # Objective function
            self.initial_params,      # Initial guess parameters
            method=self.optimization_method, # Optimization method (e.g., 'L-BFGS-B')
            bounds=self.parameter_bounds,    # Parameter bounds
            callback=callback,          # Callback function
            options=optimization_options   # Optimization options
        )

        self.optimization_results = results # Store optimization results in the class instance

        if self.verbose:
            print("\nOptimization finished.")
            print("Optimization Results:")
            print(results) # Print full results for now

        return results # Or return specific parts of results if you prefer


if __name__ == '__main__':
    # --- Parse Command Line Arguments ---
    parsed_args = parse_command_line_args()

    # --- Set Up Deconvolution Parameters and Load Data using setup_deconvolution function ---
    params_dict = setup_deconvolution(parsed_args) # Call setup_deconvolution to get params_dict

    # --- Instantiate ShannonDeconvolution (passing params_dict) ---
    deconvolver = ShannonDeconvolution(**params_dict) # Instantiate ShannonDeconvolution using unpacked params_dict

    # --- Run Optimization ---
    optimization_results = deconvolver.run_optimization()

    # --- Output and Plotting (same as before) ---
    print("\n--- Optimization Results ---")
    print(optimization_results)
    if deconvolver.update_visualization:
        print("\nPlotting is persistent. Close the plot window to exit.")
        plt.show(block=True)
    else:
        print("\nPlotting was not enabled. Script finished.")

