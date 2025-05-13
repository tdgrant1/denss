import os
import sys, time
import re
import glob
import numpy as np
from scipy import optimize, interpolate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import denss
from functools import partial

from io import StringIO
from collections import Counter


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


def calculate_profile_similarity(profile1, profile2, apply_scaling=True, scaling_method='lsq',
                                 q_range=None, q_values=None):
    """
    Calculate similarity between two scattering profiles with optional scaling.

    Parameters:
    -----------
    profile1, profile2 : numpy.ndarray
        Intensity profiles to compare
    apply_scaling : bool
        Whether to scale profile2 to profile1 before similarity calculation
    scaling_method : str
        Method to use for scaling:
        - 'lsq': Least squares scaling factor (minimizes sum of squared differences)
        - 'mean': Scale by ratio of means
        - 'max': Scale by ratio of maximum values
    q_range : tuple or None
        Optional (min_q, max_q) range to use for scaling and similarity calc
    q_values : numpy.ndarray or None
        Q values corresponding to the profiles (required if q_range is specified)

    Returns:
    --------
    float
        Similarity value (cosine similarity after optimal scaling)
    float or None
        Scaling factor (if apply_scaling=True)
    """
    # Handle empty or all-zero profiles
    if len(profile1) == 0 or len(profile2) == 0:
        return np.nan, None

    if np.all(profile1 == 0) or np.all(profile2 == 0):
        return np.nan, None

    # Apply q_range restriction if specified
    if q_range is not None and q_values is not None:
        min_q, max_q = q_range
        mask = (q_values >= min_q) & (q_values <= max_q)
        p1 = profile1[mask]
        p2 = profile2[mask]
    else:
        p1 = profile1
        p2 = profile2

    # Calculate scaling factor if requested
    scale_factor = None
    if apply_scaling:
        if scaling_method == 'lsq':
            # Optimal least squares scaling: scale_factor = (p1 · p2)/(p2 · p2)
            # This minimizes sum((p1 - scale*p2)²)
            scale_factor = np.dot(p1, p2) / np.dot(p2, p2) if np.dot(p2, p2) > 0 else 1.0
        elif scaling_method == 'mean':
            # Scale by ratio of means
            mean1 = np.mean(p1)
            mean2 = np.mean(p2)
            scale_factor = mean1 / mean2 if mean2 > 0 else 1.0
        elif scaling_method == 'max':
            # Scale by ratio of maxima
            max1 = np.max(p1)
            max2 = np.max(p2)
            scale_factor = max1 / max2 if max2 > 0 else 1.0
        else:
            scale_factor = 1.0  # No scaling

        # Apply scaling to profile2
        p2_scaled = p2 * scale_factor
    else:
        # No scaling
        p2_scaled = p2

    # Calculate cosine similarity with scaled profiles
    norm_p1 = np.linalg.norm(p1)
    norm_p2_scaled = np.linalg.norm(p2_scaled)

    if norm_p1 == 0 or norm_p2_scaled == 0:
        return np.nan, scale_factor

    cosine_similarity = np.dot(p1, p2_scaled) / (norm_p1 * norm_p2_scaled)

    return cosine_similarity, scale_factor

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

    # Just add a single argument for error column
    parser.add_argument("--error_column", type=int, default=2,
                        help="Column index for error values in data files (default: 2)")

    parser.add_argument("--interpolation_mode", default="auto",
                        choices=["none", "common", "union", "reference", "auto"],
                        help="How to handle files with different q-points: " +
                             "'none': require identical q-points, " +
                             "'common': use common q-range, " +
                             "'union': use all q-points, " +
                             "'reference': interpolate to first file, " +
                             "'auto': try common, then reference")
    parser.add_argument("--force_uniform_grid", action="store_true", default=True,
                      help="Force resampling data to a uniform q-grid (default: True)")

    # Add NaN handling option
    parser.add_argument("--nan_handling", default="remove",
                        choices=["remove", "interpolate", "zero"],
                        help="How to handle NaN values in data: " +
                             "'remove': Remove points with NaNs, " +
                             "'interpolate': Replace NaNs with interpolated values, " +
                             "'zero': Replace NaNs with zeros (default: remove)")

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

    # HDF5-specific arguments
    parser.add_argument("--hdf5_q_dataset",
                        help="Path to q values dataset in HDF5 file (e.g., '/entry/data/q')")
    parser.add_argument("--hdf5_q_attribute",
                        help="Name of attribute containing q values (e.g., 'qgrid')")
    parser.add_argument("--hdf5_q_attribute_path",
                        help="Path to object containing the q attribute (default: file root)")
    parser.add_argument("--hdf5_I_dataset",
                        help="Path to intensity values dataset in HDF5 file (e.g., '/entry/data/I')")
    parser.add_argument("--intensity_index", default=0, type=lambda x: int(x) if x.isdigit() else x,
                        help="Index or field name for intensity values when stored with uncertainties")
    parser.add_argument("--uncertainty_index", type=lambda x: int(x) if x.isdigit() else x,
                        help="Index or field name for uncertainty values (optional)")

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

    # Add these arguments to your parser
    parser.add_argument("--buffer_frames",
                        help="Semi-colon separated list of comma-separated start,end frames for buffer regions. " +
                             "Example: '0,10;90,100' for frames 0-10 and 90-100")
    parser.add_argument("--n_buffer_components", type=int, default=2,
                        help="Number of SVD components to extract from buffer regions (default: 2)")

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
    parser.add_argument("--i0_constraint_weight", type=float, default=1.0,
                        help="Weight for I(0) constraint penalty to prevent scale ambiguity.")
    parser.add_argument("--fractions_weight", type=float, default=1.0,
                        help="Weight for fraction sum penalty. Default: 1.0")
    parser.add_argument("--profile_similarity_weight", type=float, default=1.0,
                        help="Weight for profile similarity penalty. Default: 10.0")
    parser.add_argument("--water_peak_range", default="1.9,2.0",
                        help="q-range for water peak (e.g., '1.9,2.0'). Default: 1.9,2.0")
    parser.add_argument("--target_similarity", type=str, default=None,
                        help="Comma-separated target profile similarity to water for each unknown component. "
                             "Can be floating point values (e.g., '0.2,0.3'), file paths (.dat or .pdb), or 'none' to "
                             "disable the constraint for specific components. "
                             "Examples: '0.2,none,0.3' or 'profile1.dat,none,0.1', where, for example, "
                             "profile1.dat could be a calculated scattering profile from a known pdb file."
                             "Requires at least one known_profile_type to be water.")
    # P(r) Smoothness Penalty
    parser.add_argument("--pr_smoothness_weight", type=float, default=0.0,
                        help="Weight for P(r) smoothness penalty. Higher values produce smoother P(r) functions. Default: 0.0")
    # high q I(q) Smoothness Penalty
    parser.add_argument("--iq_smoothness_weight", type=float, default=0.0,
                        help="Weight for I(q) smoothness penalty at high q. Higher values produce smoother I(q) functions. Default: 0.0")
    parser.add_argument("--iq_negative_weight", type=float, default=0.0,
                        help="Weight for negative I(q) penalty. Higher values ensure positive I(q) functions. Default: 0.0")

    parser.add_argument("--fraction_shape_constraints", default=None,
                        help="Comma-separated shape constraints for each component. Options: 'linear', 'exponential', 'sigmoid', 'smooth', 'none'. Default: none for all components.")
    parser.add_argument("--fractions_smoothness_weight", type=float, default=0.0,
                        help="Weight for fraction smoothness penalty. Higher values produce smoother fraction profiles. Default: 0.0")
    parser.add_argument("--fraction_shape_penalty_weight", type=float, default=1.0,
                        help="Weight for fraction shape constraint penalty. Default: 1.0")
    parser.add_argument("--global_smoothness", action="store_true", default=False,
                        help="Apply smoothness constraint to all components regardless of individual constraints. Default: False")


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


def load_mixture_stack_from_hdf5(file_path, q_dataset=None, q_attribute=None,
                                 q_attribute_path=None, I_dataset=None,
                                 intensity_index=0, uncertainty_index=None,
                                 handle_nans='remove'):
    """
    Load mixture stack from a single HDF5 file with support for combined intensity/uncertainty data
    and comprehensive NaN handling.

    Parameters:
    -----------
    file_path : str
        Path to HDF5 file containing SAXS data
    q_dataset : str
        Path to q values dataset (e.g., '/entry/data/q')
    q_attribute : str
        Name of attribute containing q values (e.g., 'qgrid')
    q_attribute_path : str
        Path to object with the attribute (default: root level)
    I_dataset : str
        Path to intensity dataset within HDF5 file (e.g., '/entry/data/I')
    intensity_index : int or str
        Index for intensity values (e.g., 0 for first dimension in a multi-dimensional array)
    uncertainty_index : int, str or None
        Index for uncertainty values (optional)
    handle_nans : str
        Strategy for handling NaN values: 'remove', 'interpolate', or 'zero' (default: 'remove')

    Returns:
    --------
    tuple
        (q_values, mixture_stack) as numpy arrays with NaNs handled
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("Loading HDF5 files requires the h5py package. Please install it using 'pip install h5py'.")

    import os
    import numpy as np
    from scipy.interpolate import interp1d

    if not os.path.isfile(file_path):
        raise ValueError(f"HDF5 file not found: {file_path}")

    print(f"Loading data from HDF5 file: {file_path}")

    with h5py.File(file_path, 'r') as f:
        # Load q values based on the provided parameters
        if q_dataset is not None:
            # Get q from dataset path
            if q_dataset in f:
                q_values = np.array(f[q_dataset][()])
                print(f"Loaded q values from dataset: {q_dataset}")
            else:
                raise ValueError(f"Dataset not found: {q_dataset}")
        elif q_attribute is not None:
            # Get q from attribute
            obj = f
            if q_attribute_path is not None:
                if q_attribute_path in f:
                    obj = f[q_attribute_path]
                else:
                    raise ValueError(f"Path not found: {q_attribute_path}")

            if q_attribute in obj.attrs:
                q_values = np.array(obj.attrs[q_attribute])
                location = "file root" if q_attribute_path is None else q_attribute_path
                print(f"Loaded q values from attribute '{q_attribute}' on {location}")
            else:
                location = "file root" if q_attribute_path is None else q_attribute_path
                raise ValueError(f"Attribute '{q_attribute}' not found on {location}")
        else:
            raise ValueError("Must specify either q_dataset or q_attribute for HDF5 files")

        # Ensure q_values is 1D
        if q_values.ndim > 1:
            q_values = q_values.flatten()
            print(f"Flattened multi-dimensional q values to 1D array with {len(q_values)} points")

        # Load intensity values
        if I_dataset is None:
            raise ValueError("Must specify I_dataset for HDF5 files")

        if I_dataset in f:
            # Get the raw dataset first
            raw_I_data = f[I_dataset]

            # Determine how to extract intensities based on the data structure
            if hasattr(raw_I_data.dtype, 'names') and raw_I_data.dtype.names is not None:
                # This is a compound dataset with named fields
                if isinstance(intensity_index, str) and intensity_index in raw_I_data.dtype.names:
                    I_values = np.array(raw_I_data[intensity_index])
                    print(f"Extracted intensities from compound field: '{intensity_index}'")
                else:
                    field_names = raw_I_data.dtype.names
                    raise ValueError(
                        f"For compound datasets, intensity_index must be a valid field name. Available fields: {field_names}")
            else:
                # Regular numeric dataset
                raw_data_array = np.array(raw_I_data)

                # Handle different array shapes
                if raw_data_array.ndim == 1:
                    # 1D array - assume it's just intensities
                    I_values = raw_data_array
                    print("Loaded 1D intensity array")
                elif raw_data_array.ndim == 2:
                    # 2D array - could be [frames, q_points] or [q_points, (I, err)]
                    if raw_data_array.shape[1] == 2 and len(q_values) == raw_data_array.shape[0]:
                        # Likely [q_points, (I, err)] format
                        I_values = raw_data_array[:, intensity_index]
                        print(f"Extracted intensities from column {intensity_index} of 2D array format [q_points, 2]")
                    else:
                        # Likely [frames, q_points] format or [q_points, frames]
                        if raw_data_array.shape[0] == len(q_values):
                            # Transpose needed - q values match first dimension
                            I_values = raw_data_array.T
                            print(f"Transposed 2D array from [q_points, frames] to [frames, q_points]")
                        else:
                            # No transpose needed
                            I_values = raw_data_array
                            print(f"Loaded 2D array in format [frames, q_points]")
                elif raw_data_array.ndim == 3:
                    # 3D array - handle multiple possible formats
                    print(f"Found 3D array with shape: {raw_data_array.shape}")

                    # Case 1: [frames, q_points, (I, err)]
                    if raw_data_array.shape[2] == 2:
                        I_values = raw_data_array[:, :, intensity_index]
                        print(
                            f"Extracted intensities from index {intensity_index} of 3D array in format [frames, q_points, 2]")

                    # Case 2: [frames, (I, err), q_points]
                    elif raw_data_array.shape[1] == 2:
                        I_values = raw_data_array[:, intensity_index, :]
                        print(
                            f"Extracted intensities from index {intensity_index} of 3D array in format [frames, 2, q_points]")

                    # Case 3: [(I, err), frames, q_points]
                    elif raw_data_array.shape[0] == 2:
                        I_values = raw_data_array[intensity_index, :, :]
                        print(
                            f"Extracted intensities from index {intensity_index} of 3D array in format [2, frames, q_points]")

                    else:
                        raise ValueError(
                            f"Unexpected 3D array shape: {raw_data_array.shape}. For 3D arrays with intensity/uncertainty, one dimension should be 2.")
                else:
                    raise ValueError(f"Unexpected array dimension: {raw_data_array.ndim}")

        else:
            raise ValueError(f"Intensity dataset not found: {I_dataset}")

        # Handle different data shapes for the extracted intensity values
        if I_values.ndim == 1:  # Single frame
            mixture_stack = I_values.reshape(1, -1)
            print(f"Reshaped 1D intensity array to [1, {len(I_values)}]")
        elif I_values.ndim == 2:  # Multiple frames already in matrix form
            # Check if frames are in first or second dimension by comparing with q_values length
            if I_values.shape[1] == len(q_values):
                # No transpose needed - frames are already in first dimension, q in second
                mixture_stack = I_values
                print(f"Using intensity array as is with shape {I_values.shape}")
            elif I_values.shape[0] == len(q_values):
                # Transpose needed - q values are in first dimension
                mixture_stack = I_values.T
                print(f"Transposed intensity array from {I_values.shape} to {I_values.T.shape}")
            else:
                # If neither dimension matches q_values length exactly, make a best guess
                if abs(I_values.shape[1] - len(q_values)) < abs(I_values.shape[0] - len(q_values)):
                    # Second dimension is closer to q_values length
                    mixture_stack = I_values
                    print(f"Best guess: using array as is with shape {I_values.shape}")
                else:
                    # First dimension is closer to q_values length
                    mixture_stack = I_values.T
                    print(f"Best guess: transposing array from {I_values.shape} to {I_values.T.shape}")
        else:
            raise ValueError(f"Unexpected shape for extracted intensity values: {I_values.shape}")

    # Verify dimensions
    if mixture_stack.shape[1] != len(q_values):
        print(f"Warning: Mismatch between q values ({len(q_values)}) and intensity values ({mixture_stack.shape[1]})")

        # Try to handle common mismatches
        if mixture_stack.shape[1] > len(q_values):
            # Truncate intensity data to match q values
            print(f"Truncating intensity data from {mixture_stack.shape[1]} to {len(q_values)} q-points")
            mixture_stack = mixture_stack[:, :len(q_values)]
        elif abs(mixture_stack.shape[1] - len(q_values)) < 5:
            # Small difference - could be precision issue, reshape intensity
            old_shape = mixture_stack.shape
            uniform_mixture_stack = np.zeros((old_shape[0], len(q_values)))
            min_length = min(old_shape[1], len(q_values))
            uniform_mixture_stack[:, :min_length] = mixture_stack[:, :min_length]
            print(f"Reshaped intensity data to match q values length ({len(q_values)})")
            mixture_stack = uniform_mixture_stack
        else:
            raise ValueError(
                f"Significant mismatch between q values ({len(q_values)}) and intensity values ({mixture_stack.shape[1]}) that cannot be automatically reconciled")

    # Initialize mixture_stack_errors to None
    mixture_stack_errors = None

    # Extract uncertainty/error values if uncertainty_index is provided
    if uncertainty_index is not None:
        try:
            raw_I_data = f[I_dataset]

            # Based on the array dimensionality and shape, extract errors
            if raw_I_data.ndim == 3:
                # 3D array - check shapes to determine format
                if raw_I_data.shape[1] == 2:
                    # Format is [frames, 2, q_points] where index 1 contains errors
                    mixture_stack_errors = np.array(raw_I_data[:, uncertainty_index, :])
                    print(
                        f"Extracted errors from index {uncertainty_index} of 3D array in format [frames, 2, q_points]")
                elif raw_I_data.shape[0] == 2:
                    # Format is [2, frames, q_points] where index 1 contains errors
                    mixture_stack_errors = np.array(raw_I_data[uncertainty_index, :, :])
                    print(
                        f"Extracted errors from index {uncertainty_index} of 3D array in format [2, frames, q_points]")
                elif raw_I_data.shape[2] == 2:
                    # Format is [frames, q_points, 2] where index 1 contains errors
                    mixture_stack_errors = np.array(raw_I_data[:, :, uncertainty_index])
                    print(
                        f"Extracted errors from index {uncertainty_index} of 3D array in format [frames, q_points, 2]")
            elif raw_I_data.ndim == 2:
                # For 2D arrays - check if it's a compound dataset with named fields
                if hasattr(raw_I_data.dtype, 'names') and raw_I_data.dtype.names is not None:
                    # Compound dataset with named fields
                    if isinstance(uncertainty_index, str) and uncertainty_index in raw_I_data.dtype.names:
                        error_values = np.array(raw_I_data[uncertainty_index])
                        # Reshape if needed
                        if error_values.shape == mixture_stack.shape:
                            mixture_stack_errors = error_values
                        else:
                            print(
                                f"Warning: Error shape {error_values.shape} doesn't match intensity shape {mixture_stack.shape}")
                else:
                    # Regular 2D array - assume second column is errors if uncertainty_index=1
                    # This is less common for HDF5 files but included for completeness
                    if uncertainty_index == 1 and raw_I_data.shape[1] > 1:
                        mixture_stack_errors = np.array(raw_I_data[:, 1])
                        print(f"Extracted errors from column {uncertainty_index} of 2D array")

            # If errors were extracted, ensure they match the shape of mixture_stack
            if mixture_stack_errors is not None:
                if mixture_stack_errors.shape != mixture_stack.shape:
                    print(
                        f"Warning: Error shape {mixture_stack_errors.shape} doesn't match intensity shape {mixture_stack.shape}")
                    # Try reshaping if dimensions are compatible
                    if mixture_stack_errors.size == mixture_stack.size:
                        mixture_stack_errors = mixture_stack_errors.reshape(mixture_stack.shape)
                        print(f"Reshaped errors to match intensity shape: {mixture_stack.shape}")
            else:
                print(f"Warning: Could not extract errors using uncertainty_index={uncertainty_index}")
                # Generate default errors as a percentage of intensity
                mixture_stack_errors = np.abs(mixture_stack) * 0.05  # 5% of intensity
                print("Generated default errors as 5% of intensity values")
        except Exception as e:
            print(f"Error extracting uncertainties: {e}")
            # Generate default errors
            mixture_stack_errors = np.abs(mixture_stack) * 0.05  # 5% of intensity
            print("Generated default errors as 5% of intensity values due to extraction error")
    else:
        # No uncertainty index provided, generate default errors
        mixture_stack_errors = np.abs(mixture_stack) * 0.05  # 5% of intensity
        print("No uncertainty_index provided. Generated default errors as 5% of intensity values")

    # Ensure errors are positive
    if mixture_stack_errors is not None:
        # Replace zeros or negative values with small positive values
        min_error = np.abs(mixture_stack).max() * 0.001  # 0.1% of max intensity as minimum error
        mask = mixture_stack_errors <= 0
        if np.any(mask):
            mixture_stack_errors[mask] = min_error
            print(f"Replaced {np.sum(mask)} non-positive error values with minimum value {min_error:.2e}")

    # ------- NaN Handling in q values -------
    # Check for NaN values in q_values
    q_nan_mask = np.isnan(q_values)
    if np.any(q_nan_mask):
        num_q_nans = np.sum(q_nan_mask)
        print(f"  Found {num_q_nans} NaN values in q_values from HDF5 file")

        # NaNs in q_values need to be removed, as q must be well-defined
        valid_q_mask = ~q_nan_mask
        q_values = q_values[valid_q_mask]

        # Adjust mixture_stack to match valid q points
        mixture_stack = mixture_stack[:, valid_q_mask]
        print(f"  Removed {num_q_nans} q points with NaN values")

    # ------- NaN Handling in intensity data -------
    # Check for NaN values in intensity data
    nan_points = np.isnan(mixture_stack)
    if np.any(nan_points):
        num_frames_with_nans = np.sum(np.any(nan_points, axis=1))
        total_nans = np.sum(nan_points)
        print(
            f"  Found {total_nans} NaN values in intensity data across {num_frames_with_nans}/{mixture_stack.shape[0]} frames")

        if handle_nans == 'remove':
            # For each frame, find valid q points (this is more complex since we need consistent q across frames)
            valid_q_mask = ~np.any(nan_points, axis=0)  # Valid if no frame has NaN at this q

            # Check if we're removing too many points
            if np.sum(valid_q_mask) < len(q_values) * 0.5:
                print(f"  Warning: More than 50% of q points have NaNs in some frames")

            # Apply the mask to both q_values and mixture_stack
            q_values = q_values[valid_q_mask]
            mixture_stack = mixture_stack[:, valid_q_mask]
            print(f"  Removed {len(valid_q_mask) - np.sum(valid_q_mask)} q points with NaNs in any frame")

        elif handle_nans == 'interpolate':
            # Interpolate NaNs for each frame independently
            for i in range(mixture_stack.shape[0]):
                frame = mixture_stack[i]
                nan_mask = np.isnan(frame)

                if np.any(nan_mask):
                    valid_mask = ~nan_mask
                    if np.sum(valid_mask) > 3:  # Need enough points for interpolation
                        # Get valid q and I values for this frame
                        valid_q = q_values[valid_mask]
                        valid_I = frame[valid_mask]

                        # Create interpolation function
                        try:
                            interp_func = interp1d(
                                valid_q, valid_I,
                                kind='linear',
                                bounds_error=False,
                                fill_value=(valid_I[0], valid_I[-1])
                            )

                            # Replace NaNs with interpolated values
                            frame[nan_mask] = interp_func(q_values[nan_mask])
                        except Exception as e:
                            # Fallback if interpolation fails (e.g., not enough points)
                            print(f"  Frame {i}: Interpolation failed ({e}), using nearest valid value")
                            # Find nearest valid value for each NaN
                            for j in np.where(nan_mask)[0]:
                                valid_indices = np.where(valid_mask)[0]
                                if len(valid_indices) > 0:
                                    nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - j))]
                                    frame[j] = frame[nearest_idx]
                                else:
                                    frame[j] = 0.0  # If no valid points at all, use zero
                    else:
                        # Not enough points for interpolation, use nearest valid point
                        print(f"  Frame {i}: Not enough valid points for interpolation, using nearest-value fill")
                        # Find nearest valid value for each NaN
                        for j in np.where(nan_mask)[0]:
                            valid_indices = np.where(valid_mask)[0]
                            if len(valid_indices) > 0:
                                nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - j))]
                                frame[j] = frame[nearest_idx]
                            else:
                                frame[j] = 0.0  # If no valid points at all, use zero

            print(f"  Interpolated {total_nans} NaN values across all frames")

        elif handle_nans == 'zero':
            # Replace all NaNs with zeros - simplest approach
            mixture_stack = np.nan_to_num(mixture_stack, nan=0.0)
            print(f"  Replaced {total_nans} NaN values with zeros")

        else:
            # Unknown strategy, default to removal
            print(f"  Unknown NaN handling strategy '{handle_nans}', defaulting to 'remove'")
            valid_q_mask = ~np.any(nan_points, axis=0)
            q_values = q_values[valid_q_mask]
            mixture_stack = mixture_stack[:, valid_q_mask]

    # Final verification - ensure we have enough data points
    if len(q_values) < 10:
        raise ValueError(f"Too few valid q points ({len(q_values)}) remain after NaN handling")

    if mixture_stack.shape[0] == 0:
        raise ValueError("No valid frames remain after NaN handling")

    print(f"Successfully loaded {mixture_stack.shape[0]} frames with {mixture_stack.shape[1]} q-points each")
    return q_values, mixture_stack, mixture_stack_errors


def detect_and_load_mixture_stack(input_path, q_values_file=None,
                              q_column=0, I_column=1, error_column=2,
                              sort_method="natural", range_start=None, range_end=None,
                              range_step=1, range_padding=None, hdf5_q_dataset=None,
                              hdf5_q_attribute=None, hdf5_q_attribute_path=None,
                              hdf5_I_dataset=None, intensity_index=0, uncertainty_index=None,
                              handle_nans='remove'):
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
    hdf5_q_dataset : str or None
        Path to q values dataset in HDF5 file (e.g., '/entry/data/q')
    hdf5_q_attribute : str or None
        Name of attribute containing q values (e.g., 'qgrid')
    hdf5_q_attribute_path : str or None
        Path to object containing the q attribute (default: file root)
    hdf5_I_dataset : str or None
        Path to intensity values dataset in HDF5 file (e.g., '/entry/data/I')

    Returns:
    --------
    tuple
        (q_values, mixture_stack) as numpy arrays
    """

    # First, check if we have a separate q_values file
    q_values = None
    if q_values_file and os.path.isfile(q_values_file):
        q_values = np.genfromtxt(q_values_file, invalid_raise=False, usecols=(0,))
        print(f"Loading q values from: {q_values_file}")

    # Case 0: Check for HDF5 file
    if os.path.isfile(input_path) and (input_path.endswith('.h5') or input_path.endswith('.hdf5')):
        print(f"Detected HDF5 input: {input_path}")
        return load_mixture_stack_from_hdf5(
            file_path=input_path,
            q_dataset=hdf5_q_dataset,
            q_attribute=hdf5_q_attribute,
            q_attribute_path=hdf5_q_attribute_path,
            I_dataset=hdf5_I_dataset,
            intensity_index=intensity_index,
            uncertainty_index=uncertainty_index,
            handle_nans=handle_nans
        )

    # Case 1: Input is a .npy file
    if input_path.endswith('.npy') and os.path.isfile(input_path):
        mixture_data = np.load(input_path)
        print(f"Loading .npy mixture stack from {input_path}")

        # If we already have q values, use those
        if q_values is not None:
            # Ensure the npy file has at least 3 columns (q, intensity, error)
            if mixture_data.shape[1] < 3:
                raise ValueError(f"NPY file must have at least 3 columns (q, I, error) when using with q_values_file")
            # Assume first column after q is I, and second is error
            mixture_stack = mixture_data[:, 1]
            mixture_stack_errors = mixture_data[:, 2]
            return q_values, mixture_stack, mixture_stack_errors
        else:
            # Assume first column is q, second is I, third is error
            if mixture_data.shape[1] < 3:
                raise ValueError(f"NPY file must have at least 3 columns (q, I, error)")
            q = mixture_data[:, 0].copy()
            mixture_stack = mixture_data[:, 1].copy()
            mixture_stack_errors = mixture_data[:, 2].copy()
            return q, mixture_stack, mixture_stack_errors

    # Case 2: Input is a directory
    if os.path.isdir(input_path):
        print(f"Detected directory input. Loading all .dat files from {input_path}")
        files = sorted(glob.glob(os.path.join(input_path, "*.dat")))
        if not files:
            raise ValueError(f"No .dat files found in directory: {input_path}")

        # Process files with error handling
        q, mixture_stack, mixture_stack_errors = process_data_files(
            files,
            q_values,
            q_column,
            I_column,
            error_column,
            sort_method,
            interpolation_mode=parsed_args.get("interpolation_mode", "auto"),
            handle_nans=handle_nans
        )
        return q, mixture_stack, mixture_stack_errors

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

                    # Process files with error handling
                    q, mixture_stack, mixture_stack_errors = process_data_files(
                        files, q_values, q_column, I_column, error_column, sort_method, handle_nans=handle_nans)
                    return q, mixture_stack, mixture_stack_errors

    # Case 4: Input contains wildcard characters
    if '*' in input_path or '?' in input_path:
        matching_files = sorted(glob.glob(input_path))
        if matching_files:
            print(f"Detected wildcard pattern. Loading {len(matching_files)} matching files")
            q, mixture_stack, mixture_stack_errors = process_data_files(
                matching_files, q_values, q_column, I_column, error_column, sort_method, handle_nans=handle_nans)
            return q, mixture_stack, mixture_stack_errors

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

        # Process files with error handling
        q, mixture_stack, mixture_stack_errors = process_data_files(
            files, q_values, q_column, I_column, error_column, sort_method, handle_nans=handle_nans)
        return q, mixture_stack, mixture_stack_errors

    # Case 6: Input is a single data file (fallback)
    if os.path.isfile(input_path):
        print(f"Loading single file mixture stack from {input_path}")

        # Load file with error column
        data = np.genfromtxt(input_path)

        # Check if file has enough columns for error
        if data.shape[1] <= error_column:
            raise ValueError(f"Data file {input_path} must have at least {error_column + 1} columns to include errors")

        # If we already have q values, use those
        if q_values is not None:
            mixture_stack = data[:, I_column]
            mixture_stack_errors = data[:, error_column]
            return q_values, mixture_stack, mixture_stack_errors
        else:
            # Assume columns are q, I, error
            q = data[:, q_column].copy()
            mixture_stack = data[:, I_column].copy()
            mixture_stack_errors = data[:, error_column].copy()

            # Reshape if needed
            if mixture_stack.ndim == 1:
                mixture_stack = mixture_stack.reshape(1, -1)
                mixture_stack_errors = mixture_stack_errors.reshape(1, -1)

            return q, mixture_stack, mixture_stack_errors

    # If we get here, we couldn't determine the input type
    raise ValueError(
        f"Could not determine input type for '{input_path}'. "
        "Please provide a valid file, directory, file list, or pattern."
    )


def load_saxs_file_generic(file_path, q_column=0, I_column=1, error_column=2, handle_nans='remove'):
    """
    A generic SAXS data file loader that extracts q, intensity, and errors.

    Parameters:
    -----------
    file_path : str
        Path to the SAXS data file
    q_column : int
        Index of the q column in the data section (default: 0)
    I_column : int
        Index of the intensity column in the data section (default: 1)
    error_column : int
        Index of the error column in the data section (default: 2)
    handle_nans : str
        Strategy for handling NaN values (default: 'remove')

    Returns:
    --------
    tuple
        (q_values, intensities, errors) as numpy arrays
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

    # Parse the data
    try:
        data = np.genfromtxt(StringIO(data_text), invalid_raise=False)

        # Check if we have enough columns for errors
        if data.shape[1] <= error_column:
            raise ValueError(f"Data file does not contain an error column at index {error_column}. "
                             f"File has {data.shape[1]} columns, but error column {error_column} was requested. "
                             f"Please add error values to your data files.")

        # Extract q, intensity and error values
        q_values = data[:, q_column]
        intensities = data[:, I_column]
        errors = data[:, error_column]

        # Verify errors are positive - raise clear error if not
        if np.any(errors <= 0):
            raise ValueError(f"Error values must be positive. Found {np.sum(errors <= 0)} "
                             f"non-positive error values in file: {file_path}")

        # Handle NaN values
        nan_mask = np.isnan(q_values) | np.isnan(intensities) | np.isnan(errors)
        if np.any(nan_mask):
            if handle_nans == 'remove':
                # Remove all points with NaN values
                valid_mask = ~nan_mask
                q_values = q_values[valid_mask]
                intensities = intensities[valid_mask]
                errors = errors[valid_mask]
            else:
                raise ValueError(f"NaN values found in file: {file_path}. Please clean your data.")

        return q_values, intensities, errors

    except Exception as e:
        # [Handle parsing errors with clear messages]
        raise ValueError(f"Failed to load data from {file_path}: {str(e)}")


def process_data_files(files, q_values=None, q_column=0, I_column=1, error_column=2,
                       sort_method="natural", interpolation_mode="auto", handle_nans='remove'):
    """
    Helper function to process a list of data files into a mixture stack.
    Now with support for error bars.

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
    error_column : int
        Column index for error values
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
        (q_values, mixture_stack, mixture_stack_errors) as numpy arrays
    """
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
            # Use our generic parser with error handling
            file_q, file_I, file_errors = load_saxs_file_generic(file_path, q_column, I_column, error_column,
                                                                 handle_nans=handle_nans)

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

        # Initialize mixture stack and error stack
        n_q_bins = len(q_values)
        n_frames = len(files)
        mixture_stack = np.zeros((n_frames, n_q_bins))
        mixture_stack_errors = np.zeros((n_frames, n_q_bins))

        # Load intensity and error data from each file
        for i, file_path in enumerate(files):
            try:
                file_q, file_I, file_errors = load_saxs_file_generic(file_path, q_column, I_column, error_column,
                                                                     handle_nans=handle_nans)
                mixture_stack[i, :] = file_I
                mixture_stack_errors[i, :] = file_errors
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                # Fill with zeros and large errors
                mixture_stack[i, :] = np.zeros(n_q_bins)
                # Set errors to 10% of maximum intensity across all files
                max_intensity = np.max(mixture_stack) if i > 0 else 1.0
                mixture_stack_errors[i, :] = np.ones(n_q_bins) * max_intensity * 0.1

        return q_values, mixture_stack, mixture_stack_errors

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

    # Initialize mixture stack and error stack
    n_q_bins = len(target_q)
    n_frames = len(files)
    mixture_stack = np.zeros((n_frames, n_q_bins))
    mixture_stack_errors = np.zeros((n_frames, n_q_bins))

    # Load and interpolate intensity and error data from each file
    print(f"Interpolating data from {len(files)} files to a common q-grid with {n_q_bins} points...")

    for i, file_path in enumerate(files):
        try:
            # Load data with errors using our generic parser
            file_q, file_I, file_errors = load_saxs_file_generic(file_path, q_column, I_column, error_column,
                                                                 handle_nans=handle_nans)

            # Create interpolation function for intensities
            interp_func_I = interpolate.interp1d(
                file_q, file_I,
                kind='linear',
                bounds_error=False,
                fill_value=(file_I[0], file_I[-1])  # Extrapolate with end values
            )

            # Create interpolation function for errors
            interp_func_err = interpolate.interp1d(
                file_q, file_errors,
                kind='linear',
                bounds_error=False,
                fill_value=(file_errors[0], file_errors[-1])  # Extrapolate with end values
            )

            # Apply interpolation to get intensity and errors on target q-grid
            interpolated_I = interp_func_I(target_q)
            interpolated_errors = interp_func_err(target_q)

            # Check for NaN or inf values (can happen with extrapolation)
            if np.any(~np.isfinite(interpolated_I)) or np.any(~np.isfinite(interpolated_errors)):
                # Replace any non-finite values with nearest valid values
                mask_I = ~np.isfinite(interpolated_I)
                mask_err = ~np.isfinite(interpolated_errors)

                valid_indices_I = np.where(~mask_I)[0]
                valid_indices_err = np.where(~mask_err)[0]

                if len(valid_indices_I) > 0:
                    for idx in np.where(mask_I)[0]:
                        nearest_valid_idx = valid_indices_I[np.argmin(np.abs(valid_indices_I - idx))]
                        interpolated_I[idx] = interpolated_I[nearest_valid_idx]

                if len(valid_indices_err) > 0:
                    for idx in np.where(mask_err)[0]:
                        nearest_valid_idx = valid_indices_err[np.argmin(np.abs(valid_indices_err - idx))]
                        interpolated_errors[idx] = interpolated_errors[nearest_valid_idx]

                if len(valid_indices_I) == 0 or len(valid_indices_err) == 0:
                    # If no valid values, use zeros for intensity and large errors
                    max_intensity = np.max(mixture_stack) if i > 0 else 1.0
                    interpolated_I = np.zeros_like(interpolated_I)
                    interpolated_errors = np.ones_like(interpolated_errors) * max_intensity * 0.1
                    print(f"Warning: File {file_path} has no valid values after interpolation")

            # Store in mixture stack and error stack
            mixture_stack[i, :] = interpolated_I
            mixture_stack_errors[i, :] = interpolated_errors

            if i % 100 == 0 or i == len(files) - 1:
                print(f"  Processed {i + 1}/{len(files)} files")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            # Fill with zeros for intensity and large errors for this frame
            max_intensity = np.max(mixture_stack) if i > 0 else 1.0
            mixture_stack[i, :] = np.zeros(n_q_bins)
            mixture_stack_errors[i, :] = np.ones(n_q_bins) * max_intensity * 0.1

    return target_q, mixture_stack, mixture_stack_errors


def resample_to_uniform_grid(q_original, I_original):
    n_points = len(q_original)
    q_uniform = np.linspace(q_original.min(), q_original.max(), n_points)
    interp_func = interpolate.interp1d(q_original, I_original, kind='cubic',
                         bounds_error=False, fill_value=(I_original[0], I_original[-1]))
    I_uniform = interp_func(q_uniform)
    return q_uniform, I_uniform


def get_effective_water_peak_range(specified_range, q_values):
    """
    Determine an effective water peak range that works with the available q values.
    Falls back to the highest 20% of q-range if the specified range is invalid.

    Parameters:
    -----------
    specified_range : tuple
        The (min_q, max_q) range specified by the user
    q_values : numpy.ndarray
        The available q values

    Returns:
    --------
    tuple
        Effective (min_q, max_q) range to use
    str
        Description of the range selection (for logging)
    """
    specified_min, specified_max = specified_range
    q_min, q_max = q_values[0], q_values[-1]
    range_valid = (specified_min < q_max) and (specified_max <= q_max)

    if range_valid:
        return specified_range, "specified water peak range"
    else:
        # Calculate a default range in the highest 20% of the q-range
        high_q_start = q_max * 0.8
        high_q_end = q_max * 0.99  # Slightly below the max to ensure multiple points

        message = f"Water peak range ({specified_min:.3f}-{specified_max:.3f}) outside q-range ({q_min:.3f}-{q_max:.3f}). "
        message += f"Using high-q region ({high_q_start:.3f}-{high_q_end:.3f})"

        return (high_q_start, high_q_end), message


def calculate_target_similarity(profile_path, q, water_profile, water_peak_range=None):
    """
    Calculate the cosine similarity between a profile and a water profile.
    The profile can be provided as a .dat file with q and I columns, or as a .pdb file
    from which a scattering profile will be calculated.

    Parameters:
    -----------
    profile_path : str
        Path to the profile .dat file or .pdb file
    q : numpy.ndarray
        Q values for interpolation
    water_profile : numpy.ndarray
        Water profile intensity values
    water_peak_range : tuple or None
        Optional q-range to focus similarity calculation (min_q, max_q)

    Returns:
    --------
    float
        Cosine similarity value
    """
    try:
        # Process based on file type
        if profile_path.lower().endswith('.pdb'):
            print(f"Calculating scattering profile from PDB file: {profile_path}")

            # Calculate profile from PDB file using the provided code
            pdb = denss.PDB(profile_path)
            pdb2mrc = denss.PDB2MRC(pdb, run_all_on_init=True)

            # Calculate profile using the provided q values
            pdb2mrc.save_Iq_calc(qc=q)

            # Extract intensity values (assuming Iq_calc is an Nx3 array with columns [q, I, err])
            interpolated_I = pdb2mrc.Iq_calc[:, 1]

            print(f"Successfully calculated scattering profile from PDB file with {len(interpolated_I)} q-points")

        else:
            # Original functionality for .dat files
            target_data = np.genfromtxt(profile_path, invalid_raise=False)
            if target_data.shape[1] < 2:
                raise ValueError(f"Target profile file must have at least 2 columns (q, I)")

            target_q = target_data[:, 0]
            target_I = target_data[:, 1]

            # Interpolate to same q-grid as water profile
            interp_func = interpolate.interp1d(
                target_q, target_I,
                kind='linear',
                bounds_error=False,
                fill_value=(target_I[0], target_I[-1])
            )

            interpolated_I = interp_func(q)

        # Calculate similarity over full q-range
        similarity, scale_factor = calculate_profile_similarity(
            interpolated_I, water_profile,
            apply_scaling=True,
            scaling_method='lsq'
        )

        source_type = "PDB file" if profile_path.lower().endswith('.pdb') else "profile file"
        print(
            f"Calculated similarity to water: {similarity:.4f} (scale factor: {scale_factor:.4f}) from {source_type}: {profile_path}")
        return similarity

    except Exception as e:
        print(f"Error calculating similarity from file {profile_path}: {e}")
        print(f"Exception details: {str(e)}")
        print("Falling back to default similarity value (0.1)")
        return 0.1  # Default fallback value


def extract_buffer_components(mixture_stack, q, buffer_frames, n_components=2):
    """
    Extract buffer components using SVD and prepare them as known profiles.

    Parameters:
    -----------
    mixture_stack : ndarray
        Full data matrix (frames × q-points)
    q : ndarray
        Q-values corresponding to the data
    buffer_frames : list of tuples
        List of (start, end) frame ranges for buffer-only regions
    n_components : int
        Number of SVD components to extract

    Returns:
    --------
    buffer_profiles_iq : list of arrays
        List of buffer component profiles in (q, I, err) format
    buffer_profile_names : list of str
        Names for the buffer components
    """
    # Extract buffer-only frames
    buffer_data = []
    for start, end in buffer_frames:
        buffer_data.append(mixture_stack[start:end])
    buffer_data = np.vstack(buffer_data)

    # Apply SVD to extract principal components
    U, S, Vt = np.linalg.svd(buffer_data, full_matrices=False)

    # Extract top components
    buffer_profiles = Vt[:n_components]

    # Format as known profiles with errors
    buffer_profiles_iq = []
    for i in range(n_components):
        # Scale component by singular value for physical magnitude
        profile = buffer_profiles[i] * S[i]

        # Create (q, I, err) array
        buffer_iq = np.vstack((
            q,
            profile,
            np.ones_like(profile) * 0.01 * np.max(profile)  # Small errors
        )).T

        buffer_profiles_iq.append(buffer_iq)

    # Create names for the components
    buffer_profile_names = [f"buffer_{i + 1}" for i in range(n_components)]

    return buffer_profiles_iq, buffer_profile_names


def setup_deconvolution(parsed_args):
    """
    Parses command-line arguments, loads data files, preprocesses data,
    and returns a dictionary of parameters for ShannonDeconvolution class.
    """

    # --- Data Loading from Parsed Arguments Dictionary ---
    q, mixture_stack, mixture_stack_errors = detect_and_load_mixture_stack(
        parsed_args["mixture_stack"],
        q_values_file=parsed_args["q_values_file"],
        q_column=parsed_args.get("q_column", 0),
        I_column=parsed_args.get("I_column", 1),
        error_column=parsed_args.get("error_column", 2),
        sort_method=parsed_args.get("sort_method", "natural"),
        range_start=parsed_args.get("range_start"),
        range_end=parsed_args.get("range_end"),
        range_step=parsed_args.get("range_step", 1),
        range_padding=parsed_args.get("range_padding"),
        hdf5_q_dataset=parsed_args.get("hdf5_q_dataset"),
        hdf5_q_attribute=parsed_args.get("hdf5_q_attribute"),
        hdf5_q_attribute_path=parsed_args.get("hdf5_q_attribute_path"),
        hdf5_I_dataset=parsed_args.get("hdf5_I_dataset"),
        intensity_index=parsed_args.get("intensity_index", 0),
        uncertainty_index=parsed_args.get("uncertainty_index"),
        handle_nans=parsed_args.get("nan_handling", "remove")
    )

    n_frames = mixture_stack.shape[0]

    d_values_str = parsed_args["d_values"].split(',')
    unknown_component_Ds = [float(d) for d in d_values_str]

    initial_sasrec_frames = None
    unknown_profiles_init_input = parsed_args["unknown_profiles_init_input"]

    # Initialize as empty list, not None
    unknown_component_profiles_iq_init = []

    if unknown_profiles_init_input.lower() == "auto":
        print("Using automatic frame selection for initial guess profiles.")
        if len(unknown_component_Ds) == 1:
            initial_sasrec_frames = [n_frames // 2]  # Middle frame for 1 component
        elif len(unknown_component_Ds) == 2:
            initial_sasrec_frames = [0, n_frames - 1]  # First and last for 2 components
        elif len(unknown_component_Ds) >= 3:
            # Use integer indices, not numpy arrays
            frames = np.linspace(0, n_frames - 1, len(unknown_component_Ds))
            initial_sasrec_frames = [int(round(f)) for f in frames]  # Ensure integers
        else:
            initial_sasrec_frames = []  # Should not happen, but handle just in case

        print(f"  Automatically selected frames for initial Sasrecs: {initial_sasrec_frames}")

        if initial_sasrec_frames:  # If auto-selected frames are available
            for frame_index in initial_sasrec_frames:
                initialization_profile = np.copy(mixture_stack[frame_index])
                # Use actual error values instead of ones
                initialization_Iq = np.vstack(
                    (q, initialization_profile, mixture_stack_errors[frame_index])).T
                unknown_component_profiles_iq_init.append(initialization_Iq)


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
                    (q, initialization_profile, np.ones_like(initialization_profile))).T  # Create (q, I, errors) array with ones for errors
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

    # Simplified parsing for fraction shape constraints
    fraction_shape_constraints_str = parsed_args.get("fraction_shape_constraints")
    fraction_shape_constraints = None
    if fraction_shape_constraints_str:
        fraction_shape_constraints = [s.strip().lower() for s in fraction_shape_constraints_str.split(",")]
        # Replace "none" with None
        fraction_shape_constraints = [None if s == "none" else s for s in fraction_shape_constraints]

    # --- Process target_similarity argument ---
    target_similarity_str = parsed_args["target_similarity"]
    target_similarity = None

    if target_similarity_str:
        # First try to find a water profile
        water_profile_I = None
        if known_profiles_iq is not None and known_profile_types is not None:
            for i, profile_type in enumerate(known_profile_types):
                if profile_type == 'water':
                    # Get water profile I values (assumes known_profiles_iq is already loaded)
                    water_profile_I = known_profiles_iq[i][:, 1]
                    break

        target_similarity_values = []
        for value in target_similarity_str.split(','):
            value = value.strip()
            if value.lower() in ('none', 'null', 'na'):
                # Special case: None value for this component
                target_similarity_values.append(None)
                print(f"Component {len(target_similarity_values)}: No target similarity constraint")
            else:
                try:
                    # Try to parse as a float
                    target_similarity_values.append(float(value))
                except ValueError:
                    # If not a float, assume it's a file path
                    if water_profile_I is not None:
                        # Calculate similarity to water
                        similarity = calculate_target_similarity(value, q, water_profile_I, water_peak_range)
                        target_similarity_values.append(similarity)
                    else:
                        print(f"Warning: No water profile found for target similarity calculation from file {value}")
                        target_similarity_values.append(None)  # Use None instead of default value

        # Set target_similarity to the processed values
        if target_similarity_values:  # If we have any values
            target_similarity = target_similarity_values
            # Pad with None values if necessary to match component count
            if len(target_similarity) < len(unknown_component_Ds):
                target_similarity.extend([None] * (len(unknown_component_Ds) - len(target_similarity)))

    # If buffer frames are specified
    if "buffer_frames" in parsed_args and parsed_args["buffer_frames"]:
        # Parse buffer frame ranges
        buffer_frames = [tuple(map(int, r.split(',')))
                         for r in parsed_args["buffer_frames"].split(';')]

        # Extract buffer components
        buffer_profiles_iq, buffer_names = extract_buffer_components(
            mixture_stack,
            q,
            buffer_frames,
            n_components=parsed_args.get("n_buffer_components", 2)
        )

        # Add buffer components to known profiles
        known_profiles_iq = (known_profiles_iq or []) + buffer_profiles_iq
        known_profile_types = (known_profile_types or []) + ["buffer"] * len(buffer_profiles_iq)
        known_profile_names = (known_profile_names or []) + buffer_names

    optimization_options = {
        'maxiter': parsed_args["maxiter"],
        'ftol': parsed_args["ftol"],
        'maxfun': parsed_args["maxfun"],
        'maxls': parsed_args["maxls"],
        'eps': parsed_args["eps"]
    }

    params_dict = {
        'mixture_stack': mixture_stack,
        'mixture_stack_errors': mixture_stack_errors,  # Pass errors to the ShannonDeconvolution class
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
        "fraction_shape_constraints": fraction_shape_constraints,
        "fractions_smoothness_weight": parsed_args.get("fractions_smoothness_weight", 0.0),
        "fraction_shape_penalty_weight": parsed_args.get("fraction_shape_penalty_weight", 1.0),
        "global_smoothness": parsed_args.get("global_smoothness", False),
        'profile_similarity_weight': profile_similarity_weight,
        'pr_smoothness_weight': parsed_args.get("pr_smoothness_weight", 0.0),
        'iq_smoothness_weight': parsed_args.get("iq_smoothness_weight", 0.0),
        'iq_negative_weight': parsed_args.get("iq_negative_weight", 0.0),
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
                 q,  # NOW EXPECTS NumPy array
                 mixture_stack,  # NOW EXPECTS NumPy array
                 mixture_stack_errors,  # Now required
                 known_profiles_iq=None,  # NOW EXPECTS list of NumPy arrays (or None)
                 known_profile_names=None,
                 known_profile_types=None,
                 alpha=1e-8,
                 extrapolate_sasrec=False,
                 initial_sasrec_frames=None,
                 water_peak_range=(1.9, 2.0),
                 target_similarity=None,
                 i0_constraint_weight=1.0,
                 fractions_weight=1.0,
                 fraction_shape_constraints=None,
                 fractions_smoothness_weight=0.0,
                 fraction_shape_penalty_weight=0.0,
                 global_smoothness=False,
                 profile_similarity_weight=10.0,
                 pr_smoothness_weight=0.0,
                 iq_smoothness_weight=0.0,
                 iq_negative_weight=0.0,
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
        if mixture_stack_errors is None or mixture_stack_errors.shape != mixture_stack.shape:
            raise ValueError("mixture_stack_errors is required and must match the shape of mixture_stack")
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
        self.mixture_stack_errors = mixture_stack_errors
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
        self.i0_constraint_weight = i0_constraint_weight
        self.fractions_weight = fractions_weight
        self.fraction_shape_constraints = fraction_shape_constraints or []
        self.fractions_smoothness_weight = fractions_smoothness_weight
        self.fraction_shape_penalty_weight = fraction_shape_penalty_weight
        self.global_smoothness = global_smoothness
        self.pr_smoothness_weight = pr_smoothness_weight
        self.iq_smoothness_weight = iq_smoothness_weight
        self.iq_negative_weight = iq_negative_weight
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
        # Initialize optimization_params with initial_params
        self.optimization_params = self.initial_params.copy()


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
        interp_func = interpolate.interp1d(source_q_sorted, source_I_sorted,
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

    def _process_initialization_inputs(self):
        """
        Process initialization inputs after q-grid alignment but before Sasrec creation.
        Handles frame indices and performs scientific preprocessing like water subtraction.
        """
        n_components = len(self.unknown_component_Ds)
        processed_profiles = []

        # Find water profile for subtraction (if available)
        water_profile_I = None
        for i_known_profile, profile_type in enumerate(self.known_profile_types):
            if profile_type == 'water':
                water_profile_I = self.known_profiles_Iq[i_known_profile]
                if self.verbose:
                    print("Found water profile for background subtraction.")
                break

        # Process each initialization input
        for i_component in range(n_components):
            if i_component < len(self.unknown_component_profiles_iq_init_loaded):
                initialization_source = self.unknown_component_profiles_iq_init_loaded[i_component]

                # Handle different input types
                if isinstance(initialization_source, np.ndarray):
                    # It's already a profile array, but we still need to apply water subtraction
                    profile_array = initialization_source.copy()

                    # Extract intensity column for water subtraction
                    profile = profile_array[:, 1]

                    # Apply water background subtraction
                    profile = self._subtract_water_background(profile, water_profile_I)

                    # Update the intensity column with processed values
                    profile_array[:, 1] = profile

                    if self.verbose:
                        print(
                            f"  Using preprocessed profile array for component {i_component + 1} (with water subtraction)")

                elif isinstance(initialization_source, (int, str)) or (
                        isinstance(initialization_source, str) and initialization_source.isdigit()):
                    # It's a frame index
                    try:
                        frame_index = int(initialization_source) if isinstance(initialization_source, int) else int(
                            initialization_source.strip())
                        profile = np.copy(self.mixture_stack[frame_index])
                        if self.verbose:
                            print(f"  Using frame {frame_index} for component {i_component + 1}")

                        # Apply water background subtraction
                        profile = self._subtract_water_background(profile, water_profile_I)

                        # Create properly formatted array
                        profile_array = np.vstack((
                            self.q,
                            profile,
                            self.mixture_stack_errors[frame_index]
                        )).T

                    except (ValueError, IndexError) as e:
                        # Fall back to default
                        if self.verbose:
                            print(f"  Error processing frame index: {e}. Using default frame.")
                        frame_index = i_component if i_component < self.mixture_stack.shape[0] else 0
                        profile = np.copy(self.mixture_stack[frame_index])
                        profile = self._subtract_water_background(profile, water_profile_I)
                        profile_array = np.vstack((self.q, profile, self.mixture_stack_errors[frame_index])).T

                else:
                    # Unexpected type - use default
                    if self.verbose:
                        print(
                            f"  Unexpected initialization input type for component {i_component + 1}. Using default frame.")
                    frame_index = i_component if i_component < self.mixture_stack.shape[0] else 0
                    profile = np.copy(self.mixture_stack[frame_index])
                    profile = self._subtract_water_background(profile, water_profile_I)
                    profile_array = np.vstack((self.q, profile, self.mixture_stack_errors[frame_index])).T

            else:
                # No initialization provided - use default frame
                if self.verbose:
                    print(f"  No initialization provided for component {i_component + 1}. Using default frame.")
                frame_index = i_component if i_component < self.mixture_stack.shape[0] else 0
                profile = np.copy(self.mixture_stack[frame_index])
                profile = self._subtract_water_background(profile, water_profile_I)
                profile_array = np.vstack((self.q, profile, self.mixture_stack_errors[frame_index])).T

            processed_profiles.append(profile_array)

        return processed_profiles

    def _subtract_water_background(self, profile, water_profile_I):
        """
        Subtract water background from a profile if water profile is available.
        Will automatically use the high-q region if the specified water peak range
        is outside the available q-range.
        """
        if water_profile_I is None:
            return profile  # Return original if no water profile available

        effective_range, message = get_effective_water_peak_range(self.water_peak_range, self.q)
        if message and self.verbose:
            print(f"  {message}")

        water_peak_mini = denss.find_nearest_i(self.q, effective_range[0])
        water_peak_maxi = denss.find_nearest_i(self.q, effective_range[1])

        # Ensure we have a valid range with multiple points
        if water_peak_maxi <= water_peak_mini + 1:
            # Need at least 2 points for a meaningful average
            points_needed = max(3, int(len(self.q) * 0.05))  # At least 3 points or 5% of q points

            # If we're near the end of the array, move backward
            if water_peak_maxi >= len(self.q) - 1:
                water_peak_maxi = len(self.q) - 1
                water_peak_mini = max(0, water_peak_maxi - points_needed)
            # Otherwise, expand forward
            else:
                water_peak_maxi = min(len(self.q) - 1, water_peak_mini + points_needed)

        print(f"  Using water peak range indices: {water_peak_mini}-{water_peak_maxi} " +
              f"(q = {self.q[water_peak_mini]:.5f}-{self.q[water_peak_maxi]:.5f})")

        # Calculate mean intensities with safety checks
        try:
            profile_slice = profile[water_peak_mini:water_peak_maxi + 1]  # +1 to include end point
            water_slice = water_profile_I[water_peak_mini:water_peak_maxi + 1]

            if len(profile_slice) == 0:
                print("  Warning: Empty profile slice. Skipping water subtraction.")
                return profile

            profile_water_peak_int = np.nanmean(profile_slice)
            water_peak_int = np.nanmean(water_slice)

            if np.isnan(profile_water_peak_int) or np.isnan(water_peak_int) or water_peak_int <= 0:
                print("  Warning: Invalid water peak intensity calculation. Skipping water subtraction.")
                return profile

            # Calculate scale factor and apply subtraction
            water_background_sf = profile_water_peak_int / water_peak_int

            # For safety, limit extremely large scale factors
            if water_background_sf > 100:
                print(f"  Warning: Very large water scale factor ({water_background_sf:.2f}). Capping at 100.")
                water_background_sf = 100

            # Apply subtraction with slight scaling adjustment to prevent over-subtraction
            subtracted_profile = profile - water_profile_I * (water_background_sf * 0.99)

            # Ensure no negative values after subtraction (optional, depends on your approach)
            # subtracted_profile = np.maximum(subtracted_profile, 0)

            return subtracted_profile

        except Exception as e:
            print(f"  Error in water subtraction: {str(e)}")
            return profile  # Return original profile if calculation fails

    def _initialize_sasrecs_and_params(self):
        """
        Initializes Sasrecs for unknown components and sets initial parameters.
        Now works with processed profile arrays from _process_initialization_inputs.
        """
        # Process initialization inputs into properly formatted arrays
        processed_profiles = self._process_initialization_inputs()
        self.unknown_component_profiles_iq_init_loaded = processed_profiles

        # Create Sasrec objects
        for i_component in range(len(self.unknown_component_Ds)):
            initialization_Iq = self.unknown_component_profiles_iq_init_loaded[i_component]

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

            # Ensure q values are sorted (just in case)
            sort_idx = np.argsort(profile_q)
            profile_q_sorted = profile_q[sort_idx]
            profile_I_sorted = profile_I[sort_idx]

            # Use linear interpolation for values inside the range
            # Use nearest extrapolation for values outside (safer than linear extrapolation)
            interp_func = interpolate.interp1d(profile_q_sorted, profile_I_sorted,
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
                    profile_similarity_to_water, _ = calculate_profile_similarity(initial_profiles[i_component], water_I)
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
        estimated_fractions = np.zeros((len(estimated_profiles), n_frames))

        # Transpose profiles for lstsq
        estimated_profiles_for_lstsq = estimated_profiles.T

        for i_frame in range(n_frames):
            # Get data and errors for this frame
            target_data_frame = self.mixture_stack[i_frame, :]
            frame_errors = self.mixture_stack_errors[i_frame, :]

            # Create weighted matrices for least squares
            weights = 1.0 / frame_errors
            weighted_profiles = estimated_profiles_for_lstsq * weights[:, np.newaxis]
            weighted_target = target_data_frame * weights

            # Solve weighted non-negative least squares problem
            fractions_frame, residuals_frame_lstsq = optimize.nnls(weighted_profiles, weighted_target)
            estimated_fractions[:, i_frame] = fractions_frame

        return estimated_fractions

    def _calculate_residuals(self, calculated_mixture):
        """
        Calculates chi-squared residuals between the mixture stack and the calculated mixture.

        Args:
            calculated_mixture (np.ndarray): Calculated mixture stack (N_frames x N_qbins).

        Returns:
            float: Sum of residuals, normalized by the number of frames.
        """
        n_frames = self.mixture_stack.shape[0]
        n_points = self.mixture_stack.shape[1]

        # Calculate chi-squared: sum((observed - calculated)^2/error^2)
        chi_squared = np.sum(((self.mixture_stack - calculated_mixture) / self.mixture_stack_errors) ** 2)

        # Normalize by number of points for consistency
        normalized_chi_squared = chi_squared / (n_frames * n_points)

        normalized_chi_squared *= normalized_chi_squared # square it for now for testing

        return normalized_chi_squared

    def _calculate_i0_constraint_penalty(self, estimated_profiles):
        """
        Penalize deviations of I(0) from initial values to prevent scale ambiguity.

        This prevents the optimizer from finding mathematically equivalent but physically
        unreasonable solutions where intensity scales down while fractions scale up.
        """
        if not hasattr(self, 'i0_constraint_weight') or self.i0_constraint_weight <= 0:
            return 0.0

        # Store initial I(0) values if not already stored
        if not hasattr(self, 'initial_i0_values'):
            self.initial_i0_values = []
            for i_component in range(len(self.unknown_component_sasrecs)):
                # Get I(0) - using first point
                i0 = estimated_profiles[i_component][0]
                self.initial_i0_values.append(i0)
            return 0.0  # No penalty on first call

        # Calculate penalty based on deviation from initial I(0)
        total_penalty = 0.0
        for i_component in range(len(self.unknown_component_sasrecs)):
            current_i0 = estimated_profiles[i_component][0]
            initial_i0 = self.initial_i0_values[i_component]

            # Squared relative difference
            if initial_i0 > 0:
                relative_deviation = (current_i0 - initial_i0) / initial_i0
                i0_penalty = relative_deviation ** 2
                total_penalty += i0_penalty

        return total_penalty * self.i0_constraint_weight

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

    def _calculate_fraction_smoothness_penalty(self, estimated_fractions):
        """
        Calculates a smoothness penalty for fractions based on the second derivative.
        This penalizes rapid changes in fractions between adjacent frames, encouraging
        smooth evolution of component populations over time.

        Args:
            estimated_fractions (np.ndarray): Estimated fractions matrix (n_components x n_frames).

        Returns:
            float: Smoothness penalty value.
        """
        if not hasattr(self, 'fractions_smoothness_weight') or self.fractions_smoothness_weight <= 0:
            return 0.0  # Skip calculation if weight is zero or negative

        total_smoothness_penalty = 0.0

        # Apply smoothness penalty to components with 'smooth' constraint or all components if global_smoothness=True
        for i_component in range(estimated_fractions.shape[0]):
            # Skip if this component doesn't have a 'smooth' constraint and we're not applying global smoothness
            apply_smoothness = False
            if hasattr(self, 'global_smoothness') and self.global_smoothness:
                apply_smoothness = True
            elif (hasattr(self, 'fraction_shape_constraints') and
                  i_component < len(self.fraction_shape_constraints) and
                  self.fraction_shape_constraints[i_component] == 'smooth'):
                apply_smoothness = True

            if not apply_smoothness:
                continue

            # Get fractions for this component
            component_fractions = estimated_fractions[i_component]

            # Calculate second derivative using finite differences
            # Use central difference formula: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)]/h²
            h = 1.0  # Frame spacing is 1

            # Pad the array to handle boundary conditions (use edge padding)
            fractions_padded = np.pad(component_fractions, 1, mode='edge')

            # Calculate second derivative (excluding boundary points)
            fractions_second_deriv = (fractions_padded[2:] - 2 * fractions_padded[1:-1] + fractions_padded[:-2]) / (
                        h ** 2)

            # Square the second derivative
            fractions_second_deriv_squared = fractions_second_deriv ** 2

            # Sum the squared second derivatives
            smoothness_penalty = np.sum(fractions_second_deriv_squared)

            # Add to total penalty
            total_smoothness_penalty += smoothness_penalty

        # Apply weight and return
        return total_smoothness_penalty * self.fractions_smoothness_weight

    def _calculate_fraction_shape_penalty(self, estimated_fractions):
        """
        Calculates a penalty for deviations from the specified shape types.
        Instead of enforcing specific shape parameters, this fits the best shape
        of the specified type and penalizes deviations from that best-fit shape.

        Args:
            estimated_fractions (np.ndarray): Estimated fractions matrix (n_components x n_frames).

        Returns:
            float: Shape constraint penalty value.
        """
        if not hasattr(self, 'fraction_shape_constraints') or not self.fraction_shape_constraints:
            return 0.0  # Skip calculation if no constraints

        if not hasattr(self, 'fraction_shape_penalty_weight') or self.fraction_shape_penalty_weight <= 0:
            return 0.0  # Skip calculation if weight is zero or negative

        total_shape_penalty = 0.0
        n_frames = estimated_fractions.shape[1]
        frame_indices = np.arange(n_frames)

        for i_component, constraint in enumerate(self.fraction_shape_constraints):
            if constraint is None or constraint == 'none' or constraint == 'smooth' or i_component >= \
                    estimated_fractions.shape[0]:
                continue  # Skip if no constraint, smooth constraint (handled separately), or out of bounds

            # Get fractions for this component
            fractions = estimated_fractions[i_component]

            # Skip if all fractions are zero or very close to zero
            if np.all(np.abs(fractions) < 1e-10):
                continue

            # Apply the appropriate constraint based on the specified shape
            if constraint == 'linear':
                # print(f"Component {i_component} has linear constraint")
                # Fit a line to the fractions
                # y = mx + b
                try:
                    params, _ = optimize.curve_fit(
                        linear_fraction_model,
                        frame_indices,
                        fractions,
                        p0=[0, 0.5],  # Initial guess: flat line at 0.5
                        bounds=([-1, 0], [1, 1])  # Reasonable bounds for m and b
                    )
                    # Calculate best-fit line
                    best_fit = linear_fraction_model(frame_indices, *params)
                    # Calculate R-squared: 1 - SS_residual/SS_total
                    ss_total = np.sum((fractions - np.mean(fractions)) ** 2)
                    ss_residual = np.sum((fractions - best_fit) ** 2)
                    r_squared = 1 - (ss_residual / (ss_total + 1e-10))
                    # Penalize deviation from perfect linearity
                    linearity_penalty = 1.0 - r_squared  # 0 means perfect linearity, 1 means no linearity
                    component_penalty = linearity_penalty
                    total_shape_penalty += linearity_penalty
                except:
                    # If curve_fit fails, apply a higher penalty
                    total_shape_penalty += 1.0

            elif constraint == 'exponential':
                # Fit an exponential decay to the fractions
                # y = A * exp(-rate * x)
                try:
                    # Handle zero/negative values by adding a small offset
                    min_val = np.min(fractions)
                    if min_val <= 0:
                        fitting_fractions = fractions - min_val + 1e-6
                    else:
                        fitting_fractions = fractions

                    # Try log-linear fit first (more stable)
                    log_fractions = np.log(fitting_fractions)
                    # For log_fractions = log(A) - rate*frame_indices
                    # we can use linear regression
                    slope, intercept = np.polyfit(frame_indices, log_fractions, 1)
                    A = np.exp(intercept)
                    rate = -slope

                    # Calculate best-fit exponential
                    best_fit = exponential_fraction_model(frame_indices, A, rate)

                    # Calculate R-squared
                    ss_total = np.sum((fractions - np.mean(fractions)) ** 2)
                    ss_residual = np.sum((fractions - best_fit) ** 2)
                    r_squared = 1 - (ss_residual / (ss_total + 1e-10))

                    # Penalize deviation from perfect exponential shape
                    exp_penalty = 1.0 - r_squared
                    component_penalty = exp_penalty
                    total_shape_penalty += exp_penalty
                except:
                    # If fitting fails, apply a higher penalty
                    total_shape_penalty += 1.0

            elif constraint == 'sigmoid':
                # Fit a sigmoid to the fractions
                # y = L / (1 + exp(-k * (x - x0)))
                try:
                    # Initial guess for sigmoid parameters
                    p0 = [max(fractions), 0.2, n_frames / 2]  # [L, k, x0]

                    params, _ = optimize.curve_fit(
                        sigmoid_fraction_model,
                        frame_indices,
                        fractions,
                        p0=p0,
                        bounds=([0, 0.01, 0], [1.5, 1.0, n_frames])
                    )

                    # Calculate best-fit sigmoid
                    best_fit = sigmoid_fraction_model(frame_indices, *params)

                    # Calculate R-squared
                    ss_total = np.sum((fractions - np.mean(fractions)) ** 2)
                    ss_residual = np.sum((fractions - best_fit) ** 2)
                    r_squared = 1 - (ss_residual / (ss_total + 1e-10))

                    # Penalize deviation from perfect sigmoid shape
                    sigmoid_penalty = 1.0 - r_squared
                    component_penalty = sigmoid_penalty
                    total_shape_penalty += sigmoid_penalty
                except:
                    # If fitting fails, apply a higher penalty
                    total_shape_penalty += 1.0

            # print(f"Component {i_component} shape weight is: {component_penalty}")

        # Apply weight and return
        return total_shape_penalty * self.fraction_shape_penalty_weight

    def _calculate_profile_similarity_penalty(self, estimated_profiles):
        """
        Calculates the profile similarity penalty (similarity to water).
        Uses the full q-range for consistent comparison with target_similarity initialization.
        """
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

        if self.target_similarity is not None:
            n_unknown_components = len(self.unknown_component_nsh)
            profile_similarity_error = 0.0  # Initialize error accumulator

            # Find water profile
            water_I = None
            for i_known_profile, profile_type in enumerate(self.known_profile_types):
                if profile_type == 'water':
                    water_I = self.known_profiles_Iq[i_known_profile]
                    break

            if water_I is not None:
                # Now calculate similarities for each component over the FULL q-range
                for i_component in range(n_unknown_components):
                    # Skip components with None target similarity
                    if (i_component < len(self.target_similarity) and
                            self.target_similarity[i_component] is not None):
                        estimated_profile = estimated_profiles[i_component]

                        # Check for NaN values in the profile
                        if np.any(np.isnan(estimated_profile)) or np.any(np.isnan(water_I)):
                            if self.verbose:
                                print(
                                    f"  Warning: NaN detected in profile or water profile, skipping similarity calculation")
                            continue

                        try:
                            # Use enhanced similarity calculation with scaling over FULL q-range
                            similarity_to_water, scale_factor = calculate_profile_similarity(
                                estimated_profile, water_I,
                                apply_scaling=True,
                                scaling_method='lsq'
                                # No q_range parameter = use full range
                            )

                            # Check if similarity is valid
                            if np.isnan(similarity_to_water):
                                if self.verbose:
                                    print(f"  Warning: NaN similarity value for component {i_component}")
                                continue

                            # Add to error
                            component_error = np.sqrt(
                                (similarity_to_water - self.target_similarity[i_component]) ** 2
                            )
                            profile_similarity_error += component_error

                        except Exception as e:
                            if self.verbose:
                                print(f"  Warning: Error calculating profile similarity: {e}")
                            continue

            profile_similarity_penalty = profile_similarity_error * self.profile_similarity_weight
            # Ensure penalty is finite
            if np.isnan(profile_similarity_penalty) or np.isinf(profile_similarity_penalty):
                if self.verbose:
                    print("  Warning: Invalid profile similarity penalty value, using 0.0")
                return 0.0

            return profile_similarity_penalty

        return 0.0  # Default return if no conditions are met

    def _calculate_iq_smoothness_penalty(self, params):
        """
        Calculates a smoothness penalty based on the integral of the squared second derivative of I(q)
        for high q values (highest 20% where the most noise typically is).
        This directly penalizes curvature in the high q I(q) function.

        Args:
            params (np.ndarray): Parameter vector containing Shannon coefficients

        Returns:
            float: Smoothness penalty value
        """
        if self.iq_smoothness_weight <= 0 and self.iq_negative_weight <= 0:
            return 0.0  # Skip calculation if weight is zero or negative

        params_dict = self._params2dict_In_only(params)
        total_smoothness_penalty = 0.0
        total_negative_penalty = 0.0

        for i_component in range(len(self.unknown_component_sasrecs)):
            # Get Shannon coefficients and sasrec instance
            Ish = params_dict['component_In'][i_component]
            sasrec = self.unknown_component_sasrecs[i_component]

            # Calculate P(r) on sasrec.r grid
            iq = self.Ish2Iq(Ish, sasrec.Bc)

            # Calculate second derivative using finite differences
            # Use central difference formula: P''(r) ≈ [P(r+h) - 2P(r) + P(r-h)]/h²
            h = sasrec.qc[1] - sasrec.qc[0]  # Grid spacing (assumed uniform)

            # Pad the array to handle boundary conditions (use zero padding)
            iq_padded = np.pad(iq, 1, mode='edge')  # Edge padding is better than zero padding for derivatives

            # Calculate second derivative (excluding boundary points)
            iq_second_deriv = (iq_padded[2:] - 2 * iq_padded[1:-1] + iq_padded[:-2]) / (h ** 2)

            # Square the second derivative
            iq_second_deriv_squared = iq_second_deriv ** 2

            # Integrate using trapezoidal rule
            # use only high q values, highest 20%
            idx_min = int(0.6 * len(sasrec.qc))
            smoothness_penalty = np.trapezoid(iq_second_deriv_squared[idx_min:], sasrec.qc[idx_min:])

            negative_penalty = np.sum(iq[iq<0])**2

            # Add to total penalty
            total_smoothness_penalty += smoothness_penalty
            total_negative_penalty += negative_penalty

        # Apply weight and return
        return total_smoothness_penalty * self.iq_smoothness_weight + total_negative_penalty * self.iq_negative_weight

    def _calculate_pr_smoothness_penalty(self, params):
        """
        Calculates a smoothness penalty based on the integral of the squared second derivative of P(r).
        This directly penalizes curvature in the P(r) function.

        Args:
            params (np.ndarray): Parameter vector containing Shannon coefficients

        Returns:
            float: Smoothness penalty value
        """
        if self.pr_smoothness_weight <= 0:
            return 0.0  # Skip calculation if weight is zero or negative

        params_dict = self._params2dict_In_only(params)
        total_smoothness_penalty = 0.0

        for i_component in range(len(self.unknown_component_sasrecs)):
            # Get Shannon coefficients and sasrec instance
            Ish = params_dict['component_In'][i_component]
            sasrec = self.unknown_component_sasrecs[i_component]

            # Calculate P(r) on sasrec.r grid
            pr = self.Ish2P(Ish, sasrec.S)

            # Calculate second derivative using finite differences
            # Use central difference formula: P''(r) ≈ [P(r+h) - 2P(r) + P(r-h)]/h²
            h = sasrec.r[1] - sasrec.r[0]  # Grid spacing (assumed uniform)

            # Pad the array to handle boundary conditions (use zero padding)
            pr_padded = np.pad(pr, 1, mode='edge')  # Edge padding is better than zero padding for derivatives

            # Calculate second derivative (excluding boundary points)
            pr_second_deriv = (pr_padded[2:] - 2 * pr_padded[1:-1] + pr_padded[:-2]) / (h ** 2)

            # Square the second derivative
            pr_second_deriv_squared = pr_second_deriv ** 2

            # Integrate using trapezoidal rule
            smoothness_penalty = np.trapezoid(pr_second_deriv_squared, sasrec.r)

            # Add to total penalty
            total_smoothness_penalty += smoothness_penalty

        # Apply weight and return
        return total_smoothness_penalty * self.pr_smoothness_weight

    def calc_score_from_In(self, params, update_viz=False):
        """Calculates the score function from In parameters."""
        n_frames = self.mixture_stack.shape[0]

        # Store current parameters for plotting
        self.optimization_params = params.copy()

        estimated_profiles = self._calculate_profiles_from_params(params)
        estimated_fractions = self._calculate_fractions(estimated_profiles)

        # Calculate Mixture
        calculated_mixture = generate_evolving_mixture_stack(profiles=estimated_profiles, fractions=estimated_fractions)

        # Calculate Residuals
        residuals = self._calculate_residuals(calculated_mixture)

        # --- Penalties ---
        # a. Sum-to-One Fraction Constraint Penalty
        fraction_sum_penalty = self._calculate_fraction_sum_penalty(estimated_fractions)

        # b. Fraction Shape Constraint Penalty
        fraction_shape_penalty = self._calculate_fraction_shape_penalty(estimated_fractions)

        # c. Fraction Smoothness Penalty
        fraction_smoothness_penalty = self._calculate_fraction_smoothness_penalty(estimated_fractions)

        # d. Profile Similarity Penalty
        profile_similarity_penalty = self._calculate_profile_similarity_penalty(estimated_profiles)

        # e. P(r) Smoothness Penalty
        pr_smoothness_penalty = self._calculate_pr_smoothness_penalty(params)

        # f. I(q) Smoothness Penalty
        iq_smoothness_penalty = self._calculate_iq_smoothness_penalty(params)

        # g. I(0) Constraint Penalty
        i0_constraint_penalty = self._calculate_i0_constraint_penalty(estimated_profiles)

        # Total Score
        score = (residuals + fraction_sum_penalty + fraction_shape_penalty + fraction_smoothness_penalty +
                 profile_similarity_penalty + pr_smoothness_penalty + iq_smoothness_penalty +
                 i0_constraint_penalty)

        if self.verbose:
            sys.stdout.write(f"\r {self.plot_counter:06d} | {residuals:.6e} | {fraction_sum_penalty:.6e} "
                             f"| {fraction_shape_penalty:.6e} | {fraction_smoothness_penalty:.6e} "
                             f"| {profile_similarity_penalty:.6e} | {pr_smoothness_penalty:.6e} "
                             f"| {iq_smoothness_penalty:.6e} | {i0_constraint_penalty:.6e} | {score:.6e} ")
            sys.stdout.flush()

        # Visualization Update
        self.plot_counter += 1
        if update_viz and self.update_visualization and (self.plot_counter % self.plot_update_frequency == 0):
            self.update_plot_simple(estimated_profiles, estimated_fractions)

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
        fig, ax = plt.subplots(2, 3, figsize=(16, 8))

        # global line_true_profiles_noisy, line_true_profiles, line_fitted_profiles
        global line_true_profiles_noisy, line_fitted_profiles
        line_true_profiles_noisy = []
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

        # --- NEW: Third column: P(r) Profiles ---
        ax[0, 2].set_title("P(r) Profiles")
        ax[0, 2].set_ylabel("P(r)")
        ax[0, 2].set_xlabel("r (Å)")
        ax[0, 2].grid(True)

        global line_pr_profiles
        line_pr_profiles = []

        # Add P(r) profiles for each unknown component
        pr_handles = []
        pr_labels = []

        # Use existing r-values and Sn matrices from sasrec instances
        for i_component, sasrec in enumerate(self.unknown_component_sasrecs):
            component_name = self.unknown_component_names[i_component]

            # Use the r values already calculated in the sasrec instance
            r = sasrec.r

            # Initialize empty P(r) profile line
            pr_line, = ax[0, 2].plot(r, np.zeros_like(r), color=colors[i_component],
                                     label=f'{component_name} (D={sasrec.D}Å)')
            line_pr_profiles.append(pr_line)
            pr_handles.append(pr_line)
            pr_labels.append(f'{component_name} (D={sasrec.D}Å)')

        ax[0, 2].legend(pr_handles, pr_labels, loc='upper right')


        # Fit subplot - Example fit to data
        global line_example_fit_data, line_example_fit_total, line_example_fit_components  # Renamed
        line_example_fit_components = []  # List for example fit component lines

        ax[1, 0].set_title(f"Example Fit - Frame {self.fit_frame_number}")  # Generic title
        ax[1, 0].set_ylabel("I(q)")
        ax[1, 0].set_xlabel("q")
        ax[1, 0].semilogy()
        ax[1, 0].grid(True)

        example_fit_handles = []  # Legend handles for example fit
        example_fit_labels = []  # Legend labels for example fit
        # Add error bars to plots
        line_example_fit_data = ax[1, 0].errorbar(
            self.q,
            self.mixture_stack[self.fit_frame_number],
            yerr=self.mixture_stack_errors[self.fit_frame_number],
            fmt='o', color='gray', alpha=0.3,
            ecolor='lightgray', elinewidth=0.5, capsize=0,
            label=f'Frame {self.fit_frame_number} Data'
        )
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

        # --- Bottom row, second column: Chi² per Frame ---
        ax[1, 1].set_title("Chi² per Frame")
        ax[1, 1].set_ylabel("Chi²")
        ax[1, 1].set_xlabel("Frame")
        ax[1, 1].grid(True)

        global line_chi2_per_frame
        # Initialize with zeros
        line_chi2_per_frame, = ax[1, 1].plot(np.arange(self.n_frames), np.zeros(self.n_frames),
                                             color='black', marker='o', ms=3, linestyle='-', alpha=0.7)

        # Residuals Subplot - Same as before
        global residuals_img
        initial_residuals = np.zeros((self.n_frames, len(self.q)))
        residuals_img = ax[1, 2].imshow(initial_residuals,
                                        extent=[self.q.min(), self.q.max(), 0, self.n_frames],
                                        aspect='auto', cmap='viridis',
                                        origin='lower', interpolation='nearest')
        ax[1, 2].set_title("Residuals")
        ax[1, 2].set_xlabel("q")
        ax[1, 2].set_ylabel("Frame")
        fig.colorbar(residuals_img, ax=ax[1, 2], label="Residual Magnitude")

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

        global plot_counter
        plot_counter = 0
        global last_update_time
        last_update_time = time.time()

    def update_plot_simple(self, estimated_profiles, estimated_fractions):
        """Updates the plot with current profiles, fractions, P(r) functions, and chi² values."""
        global line_fitted_profiles, line_fractions_fitted, line_example_fit_components, line_pr_profiles, line_chi2_per_frame

        # Update Fitted Profiles Data
        for i_component in range(len(self.unknown_component_names)):
            line_fitted_profiles[i_component].set_ydata(estimated_profiles[i_component])

        # Update Fitted Fractions Data
        for i_component in range(len(self.unknown_component_names)):
            line_fractions_fitted[i_component].set_ydata(estimated_fractions[i_component])

        # Update Fitted Fractions Data for known profiles (if any)
        if self.known_profile_names:
            for i_known_profile in range(len(self.known_profile_names)):
                line_fractions_fitted[len(self.unknown_component_names) + i_known_profile].set_ydata(
                    estimated_fractions[len(self.unknown_component_names) + i_known_profile])

        # Calculate mixture for residuals and example fit
        calculated_mixture = generate_evolving_mixture_stack(
            profiles=estimated_profiles,
            fractions=estimated_fractions
        )

        # Update P(r) profiles using the current parameters
        params_dict = self._params2dict_In_only(self.optimization_params)

        for i_component in range(len(self.unknown_component_names)):
            # Get the Shannon coefficients for this component
            Ish = params_dict['component_In'][i_component]

            # Get the sasrec instance for this component
            sasrec = self.unknown_component_sasrecs[i_component]

            # Calculate P(r) using the existing method
            pr_values = self.Ish2P(Ish, sasrec.S)

            # Update the P(r) line
            line_pr_profiles[i_component].set_ydata(pr_values)

        # Adjust ylim for P(r) plot if needed
        all_pr_values = [line.get_ydata() for line in line_pr_profiles]
        if all_pr_values and len(all_pr_values[0]) > 0:
            max_pr = max([np.max(pr) for pr in all_pr_values if len(pr) > 0 and not np.all(np.isnan(pr))])
            min_pr = min([np.min(pr) for pr in all_pr_values if len(pr) > 0 and not np.all(np.isnan(pr))])
            ax[0, 2].set_ylim([min_pr * 1.1 if min_pr < 0 else 0, max_pr * 1.1])

        # Calculate chi^2 for each frame
        chi2_per_frame = np.zeros(self.n_frames)
        for i_frame in range(self.n_frames):
            if self.mixture_stack_errors is not None:
                # Error-weighted chi-squared
                safe_errors = np.maximum(self.mixture_stack_errors[i_frame], 1e-10)  # Prevent division by zero
                chi2_per_frame[i_frame] = np.sum(
                    ((self.mixture_stack[i_frame] - calculated_mixture[i_frame]) / safe_errors) ** 2
                ) / len(self.q)  # Normalize by number of q points
            else:
                # Unweighted chi-squared
                chi2_per_frame[i_frame] = np.sum(
                    (self.mixture_stack[i_frame] - calculated_mixture[i_frame]) ** 2
                ) / len(self.q)

        # Update chi^2 plot
        line_chi2_per_frame.set_ydata(chi2_per_frame)

        # Adjust ylim for chi^2 plot
        if not np.all(np.isnan(chi2_per_frame)):
            max_chi2 = np.nanmax(chi2_per_frame)
            if max_chi2 > 0:
                ax[1, 1].set_ylim([0, max_chi2 * 1.1])

        # Update Example Fit Data
        if hasattr(line_example_fit_data, 'set_ydata'):
            # It's a regular line
            line_example_fit_data.set_ydata(self.mixture_stack[self.fit_frame_number])
        elif hasattr(line_example_fit_data, 'errorbar'):
            # It's an errorbar plot
            line_example_fit_data.errorbar[0].set_ydata(self.mixture_stack[self.fit_frame_number])
            # Update error bars if they exist
            if len(line_example_fit_data.errorbar) > 1 and self.mixture_stack_errors is not None:
                ebars = line_example_fit_data.errorbar[1]
                for i, bar in enumerate(ebars):
                    if hasattr(bar, 'set_ydata'):
                        new_y = self.mixture_stack[self.fit_frame_number][i]
                        err = self.mixture_stack_errors[self.fit_frame_number][i]
                        bar.set_ydata([new_y - err, new_y + err])

        line_example_fit_total.set_ydata(calculated_mixture[self.fit_frame_number])

        for i_component in range(len(self.unknown_component_names)):
            line_example_fit_components[i_component].set_ydata(
                estimated_profiles[i_component] * estimated_fractions[i_component][self.fit_frame_number])

        if self.known_profile_names:
            for i_known_profile in range(len(self.known_profile_names)):
                line_example_fit_components[len(self.unknown_component_names) + i_known_profile].set_ydata(
                    estimated_profiles[len(self.unknown_component_names) + i_known_profile] *
                    estimated_fractions[len(self.unknown_component_names) + i_known_profile][self.fit_frame_number])

        # Update the residuals image (now in ax[1, 2])
        frame_residuals = self.mixture_stack - calculated_mixture
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

        print(f"\r {'Step':^6s} | {'Residuals':^12s} | {'Fract Sum':^12s} | "
              f"{'Frac Shape':^12s} | {'Frac Smooth':^12s} | "
              f"{'Similarity':^12s} | {'Pr Smooth':^12s} | {'Iq SmoothPos':^12s} | {'I0 Drift':^12s} | {'Score':^12s}")

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

