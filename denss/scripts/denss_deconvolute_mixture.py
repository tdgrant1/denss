import os.path
import sys, time
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import denss  # Assuming denss is installed or in your PYTHONPATH
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
    parser.add_argument("-m", "--mixture_stack_file", required=True,
                        help="Path to mixture stack data file (N_frames x N_qbins).")
    parser.add_argument("-q", "--q_values_file", required=True,
                        help="Path to file containing q values (must correspond to mixture_stack rows).")
    parser.add_argument("-d", "--d_values", required=True, help="Comma-separated D values for unknown components.")

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
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Enable verbose output. Default: True")  # Default verbose to True
    parser.add_argument("--no-verbose", action="store_false", dest='verbose',
                        help="Disable verbose output.")  # Option to disable verbose output

    args = parser.parse_args()

    # --- Convert argparse.Namespace to dictionary ---
    parsed_args_dict = vars(args) # Use vars() to convert Namespace to dict

    return parsed_args_dict # Return the dictionary


def setup_deconvolution(parsed_args):
    """
    Parses command-line arguments, loads data files, preprocesses data,
    and returns a dictionary of parameters for ShannonDeconvolution class.
    Now removes hardcoded water.dat loading and relies on known profiles input for water profile.
    """

    # --- Data Loading from Parsed Arguments Dictionary ---
    basename, ext = os.path.splitext(parsed_args["mixture_stack_file"])
    if ext == ".npy":
        mixture_stack = np.load(parsed_args["mixture_stack_file"])
    else:
        mixture_stack = np.genfromtxt(parsed_args["mixture_stack_file"])

    q_values_file = parsed_args["q_values_file"]  # Get q_values_file path from args
    if q_values_file:  # If q_values_file is provided
        q = np.genfromtxt(q_values_file, usecols=(0,))  # Load q-values from file
        mixture_stack_I = mixture_stack  # Assume mixture_stack file contains only intensity data
        print(f"Loading q-values from: {q_values_file}")
    else:  # If q_values_file is NOT provided, load q from mixture_stack_file
        q = mixture_stack[:, 0].copy()  # Load q-values from mixture_stack (1st column)
        mixture_stack_I = mixture_stack[:, 1:]  # Mixture stack is now only intensities (from 2nd column onwards)
        print(f"Loading q-values from first column of: {parsed_args['mixture_stack_file']}")

    mixture_stack = mixture_stack_I  # Assign intensity data back to mixture_stack
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
                initialization_profile_iq = np.genfromtxt(file_path)  # Load profile from file
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
        known_profiles_iq = [np.genfromtxt(f) for f in known_profiles_files_list]
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
        self.plot_counter = 0
        self.fit_frame_number = mixture_stack.shape[0] // 2

        # --- Initialization Steps (Data is assumed to be loaded and preprocessed already) ---
        self._load_profiles()
        self._initialize_known_profiles()  # Process known profiles (no file loading here anymore)
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
                        loaded_profile = np.genfromtxt(profile_file_path)  # Load from file path
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

    def _initialize_sasrecs_and_params(self):
        """Initializes Sasrecs for unknown components and sets initial parameters."""
        n_components = len(self.unknown_component_Ds)

        if self.initial_sasrec_frames is None:
            # Default frames: evenly spaced
            initial_sasrec_frames = np.linspace(0, self.mixture_stack.shape[0] - 1, n_components)
            self.initial_sasrec_frames = [int(round(frame)) for frame in initial_sasrec_frames]
        elif len(self.initial_sasrec_frames) != n_components:
            raise ValueError("Length of initial_sasrec_frames must match the number of unknown components.")

        water_profile_Iq = None  # ... (water profile loading logic - same as before) ...
        for i_known_profile, profile_type in enumerate(self.known_profile_types):
            if profile_type == 'water':
                water_profile_Iq = self.known_profiles_iq_loaded[i_known_profile]  # Found water profile

        self.unknown_component_profiles_iq_init_loaded = self.unknown_component_profiles_iq_init  # <--- ASSUME ALREADY NUMPY ARRAYS, just assign

        for i_component in range(n_components):  # Loop through components
            initialization_Iq = self.unknown_component_profiles_iq_init_loaded[
                i_component]  # Get profile from preloaded list
            initialization_profile = initialization_Iq[:, 1]  # Extract I(q) - assuming (q, I, errors) format

            # Subtract water background roughly
            if water_profile_Iq is not None: # Use the water_profile_Iq we found
                water_peak_mini = denss.find_nearest_i(self.q, self.water_peak_range[0])
                water_peak_maxi = denss.find_nearest_i(self.q, self.water_peak_range[1])
                initialization_profile_water_peak_int = np.mean(initialization_profile[water_peak_mini:water_peak_maxi])
                water_peak_int = np.mean(water_profile_Iq[:,1][water_peak_mini:water_peak_maxi])
                print(f"  _initialize_sasrecs_and_params: component {i_component}, water_peak_int = {water_peak_int:.6e}") # DEBUG PRINT
                if water_peak_int > 0:
                    water_background_sf = initialization_profile_water_peak_int / water_peak_int
                    print(f"  _initialize_sasrecs_and_params: component {i_component}, water_background_sf = {water_background_sf:.6e}") # DEBUG PRINT
                    initialization_profile -= water_profile_Iq[:,1] * (water_background_sf * 0.99)
                else:
                    print("Warning: Water profile has zero intensity in the water peak region. Water background subtraction skipped.")

            initialization_Iq = np.vstack((self.q, initialization_profile, np.ones_like(initialization_profile))).T

            # Now append the initialization_Iq to self.unknown_component_profiles_iq_init_loaded
            self.unknown_component_profiles_iq_init_loaded.append(initialization_Iq)

            sasrec = denss.Sasrec(initialization_Iq, D=self.unknown_component_Ds[i_component], qc=self.q,
                                  alpha=0, extrapolate=False)
            self.unknown_component_sasrecs.append(sasrec)
            print(f"  Component {self.unknown_component_names[i_component]}: Nsh = {sasrec.n}")

        self.unknown_component_Bns = [sasrec.B for sasrec in self.unknown_component_sasrecs]
        self.unknown_component_nsh = [sasrec.n for sasrec in self.unknown_component_sasrecs]
        self.initial_params = self._dict2params_In_only(self._create_params_dict_In_only())

    def _initialize_known_profiles(self):
        """Initializes known profiles, extracting I(q) values."""
        self.known_profiles_Iq = []
        for profile_iq in self.known_profiles_iq_loaded:
            self.known_profiles_Iq.append(profile_iq[:, 1]) # Extract only I(q) values

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
        """
        Calculates the water background intensity penalty, now finding water profile
        from known profiles based on type.

        Args:
            estimated_fractions (np.ndarray): Estimated fractions matrix (n_total_profiles x n_frames).

        Returns:
            float: Water penalty value.
        """
        water_penalty = 0.0
        if self.water_peak_range is not None: # water_profile_iq_loaded check removed, now checking just water_peak_range
            n_frames = self.mixture_stack.shape[0]
            n_unknown_components = len(self.unknown_component_nsh)
            water_peak_mini_index = denss.find_nearest_i(self.q, self.water_peak_range[0])
            water_peak_maxi_index = denss.find_nearest_i(self.q, self.water_peak_range[1])

            target_water_int = np.mean(self.mixture_stack[:, water_peak_mini_index:water_peak_maxi_index], axis=1)

            calculated_water_int = np.zeros(n_frames) # Initialize to zeros
            water_profile_I = None # Initialize water_profile_I to None

            for i_known_profile, profile_type in enumerate(self.known_profile_types): # Iterate to find water profile
                if profile_type == 'water':
                    water_profile_I = self.known_profiles_Iq[i_known_profile] # Found water profile

            if water_profile_I is not None: # Proceed with water penalty calculation only if water profile is found
                water_stack_component = water_profile_I[None, :] * estimated_fractions[n_unknown_components + 0][:, None] # Assuming water is the *first* known profile of type 'water' - might need adjustment if you have multiple water profiles
                calculated_water_int += np.mean(water_stack_component[:, water_peak_mini_index:water_peak_maxi_index], axis=1) # Accumulate water intensity

                water_error = np.sum(np.sqrt((target_water_int - calculated_water_int)**2))
                water_penalty = water_error * self.water_penalty_weight

        return water_penalty

    def _calculate_profile_similarity_penalty(self, estimated_profiles):
        """Calculates the profile similarity penalty (similarity to water)."""
        profile_similarity_error = 0.0  # Initialize to zero
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
        if self.unknown_component_profiles_iq_init is not None:  # Check if true profiles are available
            for i_component in range(len(self.unknown_component_names)):
                component_name = self.unknown_component_names[i_component]
                if i_component == 0:
                    Iq = np.genfromtxt('SASDCK8_fit1_model1_FH_pdb.dat')
                    sf = Iq[0,1]
                    Iq[:,1] /= sf
                    Iq[:, 2] /= sf
                elif i_component == 1:
                    Iq = np.genfromtxt('4fe9_FH_pdb.dat')
                    sf = Iq[0,1] / 2
                    Iq[:,1] /= sf
                    Iq[:, 2] /= sf
                noise = np.random.normal(loc=0, scale=Iq[:,2]*5e-2, size=Iq.shape[0])
                # temporary
                noisy_line, = ax[0, 0].plot(self.q, Iq[:, 1]+noise,
                                            marker=markers[i_component], linestyle='None', color='gray', alpha=0.3,
                                            mec='none', ms=4, label=f'Initial {component_name} I(q)')
                # noisy_line, = ax[0, 0].plot(self.q, self.unknown_component_profiles_iq_init[i_component][:, 1],
                #                             marker=markers[i_component], linestyle='None', color='gray', alpha=0.3,
                #                             mec='none', ms=4, label=f'Initial {component_name} I(q)')

                # true_line, = ax[0, 0].plot(self.q, self.unknown_component_profiles_iq_init[i_component][:, 1], color='k',
                #                            alpha=0.8, label=f'True {component_name} I(q)')  # Black line for true profile
                line_true_profiles_noisy.append(noisy_line)  # Store line handles
                # line_true_profiles.append(true_line)
                # profile_handles.extend([noisy_line, true_line])  # Add handles for legend
                profile_handles.extend([noisy_line])  # Add handles for legend
                # profile_labels.extend([f'True {component_name} I(q) (Noisy)', f'True {component_name} I(q)'])
                # profile_labels.extend([f'Initial {component_name} I(q)'])
                profile_labels.extend([f'True {component_name} I(q)'])

        # Dynamic data - Fitted Profiles (using dynamic component names)
        line_fitted_profiles = []  # List to hold lines for fitted profiles
        for i_component in range(len(self.unknown_component_names)):
            component_name = self.unknown_component_names[i_component]
            fitted_line, = ax[0, 0].plot(self.q, np.zeros_like(self.q), color=colors[i_component],
                                        label=f'Fitted {component_name} I(q)')
            line_fitted_profiles.append(fitted_line)  # Store fitted profile line handles
            profile_handles.append(fitted_line)  # Add fitted profile handle to legend
            profile_labels.append(f'Fitted {component_name} I(q)')  # Add fitted profile label to legend

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
        residuals_img.set_clim(vmin=-0.01, vmax=0.01)

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
            # bounds=self.parameter_bounds,    # Parameter bounds
            callback=callback,          # Callback function
            options=optimization_options   # Optimization options
        )

        self.optimization_results = results # Store optimization results in the class instance

        if self.verbose:
            print("\nOptimization finished.")
            print("Optimization Results:")
            print(results) # Print full results for now

        return results # Or return specific parts of results if you prefer

# if __name__ == '__main__':
#     # --- Load Data (replace with your actual data loading) ---
#     lys_Iq = np.genfromtxt("SASDCK8_fit1_model1_FH_pdb.dat")
#     cbp_Iq = np.genfromtxt("4fe9_FH_pdb.dat")
#     # rescale to have same I(0) for now
#     lys_Iq[:, 1] /= lys_Iq[0, 1]
#     lys_Iq[:, 2] /= lys_Iq[0, 1]  # scale errors too
#     cbp_Iq[:, 1] /= cbp_Iq[0, 1]
#     cbp_Iq[:, 2] /= cbp_Iq[0, 1]  # scale errors too
#     # make cbp 2x more scattering due to MW
#     sf = 2.0
#     cbp_Iq[:, 1] *= sf
#     cbp_Iq[:, 2] *= sf  # scale errors too
#     water = np.genfromtxt("water.dat")
#     q = lys_Iq[:,0]
#     water_I = np.interp(q, water[:,0], water[:,1])
#     # should be on the same q bins, but just in case interpolate
#     water_I = np.interp(q, water[:, 0], water[:, 1])
#     water_I *= lys_Iq[:,1].sum() / water_I.sum()  # just put on roughly the same scale
#
#     # --- Synthetic Mixture Stack (replace with your actual mixture data) ---
#     component_profiles = np.vstack((lys_Iq[:,1], cbp_Iq[:,1]))
#     n_frames = 50
#     n_components = component_profiles.shape[0]
#     frames = np.arange(n_frames)
#     frac_lys = 1-sigmoid_fraction_model(frames)
#     frac_cbp = 1-frac_lys
#     frac_water = np.random.normal(loc=0.4, scale=0.1, size=frac_cbp.shape)
#     component_fractions = np.vstack((frac_lys, frac_cbp))
#     background_fractions = [frac_water]
#     background_stack = np.zeros((1, len(q)))
#     background_stack[0] = water_I
#     profiles = np.vstack((component_profiles, background_stack))
#     fractions = np.vstack((component_fractions, background_fractions))
#     mixture_stack = generate_evolving_mixture_stack(profiles, fractions)
#     np.save('mixture_stack_no_noise.npy', mixture_stack)
#
#     noise = np.random.normal(0, scale=lys_Iq[:,2]*5e-8, size=mixture_stack.shape)
#     mixture_stack += noise
#     np.save('mixture_stack_noise.npy', mixture_stack)
#     exit()



#
#     # --- Prepare Inputs for ShannonDeconvolution ---
#     unknown_component_profiles_iq_init = [lys_Iq, cbp_Iq]
#     unknown_component_Ds = [50.0, 100.0]
#     known_profiles_iq = [np.vstack((q, water_I)).T]
#     unknown_component_names = ["lys", "cbp"]
#     known_profile_names = ["water"]
#     known_profile_types = ['water']  # Specify the type of the known profile
#
#     # --- Calculate target_similarity ---
#     # We'll need to come up with a generic solution for this. For example, alphafold prediction or something
#     # The nice thing is the similarity measure is similar for different protein conformations
#     target_similarity = np.zeros(n_components) # Initialize for 2 components (lys, cbp)
#     profiles_for_similarity = np.vstack((lys_Iq[:,1], cbp_Iq[:,1]))
#     for i_component in range(n_components):  # Loop through each component profile
#         # Cosine similarity between component profile and water profile
#         profile_similarity_to_water = calculate_cosine_similarity(profiles_for_similarity[i_component], water_I)
#         target_similarity[i_component] = profile_similarity_to_water
#         # add a small amount of variability to the target to account for conformational diversity,
#         # it seems the same to about std=0.0012 in my tests with different conformations of the same protein
#         # so to avoid total bias, try and add some random error to simulate a different conformation
#         target_similarity[i_component] += np.random.normal(loc=0, scale=0.0012)
#
#     # --- Instantiate ShannonDeconvolution ---
#     deconvolver = ShannonDeconvolution(
#         unknown_component_profiles_iq_init=unknown_component_profiles_iq_init,
#         unknown_component_Ds=unknown_component_Ds,
#         mixture_stack=mixture_stack,
#         q=q,
#         known_profiles_iq=known_profiles_iq,
#         unknown_component_names=unknown_component_names,
#         known_profile_names=known_profile_names,
#         known_profile_types=known_profile_types,  # Pass the known_profile_types
#         fractions_weight=1.0,
#         water_penalty_weight=0.0,
#         profile_similarity_weight=10.0,
#         target_similarity=target_similarity,
#         update_visualization=True,
#         verbose=True
#     )

# if __name__ == '__main__':
#     import argparse
#     import os  # Import os module which is used in your code
#
#     # --- Argument Parser Setup (Comprehensive Command Line Options) ---
#     parser = argparse.ArgumentParser(description="Deconvolute evolving mixture SAXS data using Shannon expansion.")
#
#     # Essential Required Input Arguments
#     parser.add_argument("-m", "--mixture_stack_file", required=True,
#                         help="Path to mixture stack data file (N_frames x N_qbins).")
#     parser.add_argument("-q", "--q_values_file", required=True, default=None,
#                         help="Path to file containing q-values.")
#     parser.add_argument("-d", "--d_values", required=True, help="Comma-separated D values for unknown components.")
#
#     # Flexible Initial Guess Profiles Input
#     parser.add_argument("-u", "--unknown_profiles_init_input", default="auto",
#                         help="Comma-separated frame indices (integers), file paths (strings), or 'auto' for automatic initial guess profiles. Defaults to 'auto'.")
#
#     # Flexible Known Profiles Input
#     parser.add_argument("-k", "--known_profiles_files", default=None,
#                         help="Comma-separated paths to known profiles (q, I). Optional.")
#     parser.add_argument("--known_profile_types", default=None,
#                         help="Comma-separated types of known profiles. Defaults to 'generic'.")
#     parser.add_argument("--known_profile_names", default=None,
#                         help="Comma-separated names for known profiles (for plotting). Optional.")
#
#     # Optimization Settings (Optional, with defaults)
#     parser.add_argument("--optimization_method", default='L-BFGS-B',
#                         help="Optimization method (scipy.optimize.minimize). Default: L-BFGS-B")
#     parser.add_argument("--maxiter", type=int, default=100, help="Maximum iterations for optimization. Default: 100")
#     parser.add_argument("--ftol", type=float, default=1e-8,
#                         help="Function tolerance for optimization convergence. Default: 1e-8")
#     parser.add_argument("--maxfun", type=int, default=100000, help="Maximum function evaluations. Default: 100000")
#     parser.add_argument("--maxls", type=int, default=50, help="Maximum line search iterations. Default: 50")
#     parser.add_argument("--eps", type=float, default=1e-8, help="Step size for numerical derivatives. Default: 1e-8")
#
#     # Penalty Settings (Optional, with defaults)
#     parser.add_argument("--fractions_weight", type=float, default=1.0,
#                         help="Weight for fraction sum penalty. Default: 1.0")
#     parser.add_argument("--profile_similarity_weight", type=float, default=10.0,
#                         help="Weight for profile similarity penalty. Default: 10.0")
#     parser.add_argument("--water_penalty_weight", type=float, default=0.0,
#                         help="Weight for water penalty. Default: 0.0")
#     parser.add_argument("--water_peak_range", default="1.9,2.0",
#                         help="q-range for water peak penalty (e.g., '1.9,2.0'). Default: 1.9,2.0")
#     parser.add_argument("--target_similarity", type=str, default=None,
#                         help="Comma-separated target profile similarity to water for each unknown component. Optional.")
#
#     # Output and Visualization Options (Optional)
#     parser.add_argument("-o", "--output_params_file", default='optimization_params_flexible_fractions.npy',
#                         help="Path to save optimized parameters (.npy file). Default: optimization_params_flexible_fractions.npy")
#     parser.add_argument("-v", "--visualize", action="store_true",
#                         help="Enable interactive visualization during optimization.")
#     parser.add_argument("--plot_update_frequency", type=int, default=10,
#                         help="Plot update frequency (iterations). Default: 10")
#     parser.add_argument("--verbose", action="store_true", default=True,
#                         help="Enable verbose output. Default: True")  # Default verbose to True
#     parser.add_argument("--no-verbose", action="store_false", dest='verbose',
#                         help="Disable verbose output.")  # Option to disable verbose output
#
#     args = parser.parse_args()
#
#
#     # --- Data Loading from Command Line Arguments (Handling --q_values_file) ---
#     basename, ext = os.path.splitext(args.mixture_stack_file)
#     if ext == ".npy":
#         mixture_stack = np.load(args.mixture_stack_file)
#     else:
#         mixture_stack = np.genfromtxt(args.mixture_stack_file)
#
#     q_values_file = args.q_values_file # Get q_values_file path from args
#
#     if q_values_file: # If q_values_file is provided
#         q = np.genfromtxt(q_values_file, usecols=(1,)) # Load q-values from file
#         mixture_stack_I = mixture_stack # Assume mixture_stack file contains only intensity data
#         print(f"Loading q-values from: {q_values_file}") # Informative message
#     else: # If q_values_file is NOT provided, load q from mixture_stack_file
#         q = mixture_stack[:, 0].copy() # Load q-values from mixture_stack (1st column)
#         mixture_stack_I = mixture_stack[:, 1:] # Mixture stack is now only intensities (from 2nd column onwards)
#         print(f"Loading q-values from first column of: {args.mixture_stack_file}") # Informative message
#
#     mixture_stack = mixture_stack_I # Assign intensity data back to mixture_stack
#
#     d_values_str = args.d_values.split(',')
#     unknown_component_Ds = [float(d) for d in d_values_str]
#
#     initial_sasrec_frames = None
#     unknown_profiles_init_input = args.unknown_profiles_init_input
#
#     if unknown_profiles_init_input.lower() == "auto":
#         initial_sasrec_frames = None
#         print("Using automatic frame selection for initial guess profiles.")
#     elif unknown_profiles_init_input:
#         initial_sasrec_frames = []
#         unknown_profiles_init_input_list = [x.strip() for x in unknown_profiles_init_input.split(',')]
#         for input_item in unknown_profiles_init_input_list:
#             try:
#                 frame_index = int(input_item)
#                 initial_sasrec_frames.append(frame_index)
#             except ValueError:
#                 file_path = input_item
#                 initial_sasrec_frames.append(file_path)
#         print(f"Using user-provided initial Sasrec inputs: {initial_sasrec_frames}")
#     else:
#         print("Using automatic frame selection for initial guess profiles (default).")
#
#     known_profiles_files = args.known_profiles_files
#     known_profiles_iq = None
#     known_profile_types_str = args.known_profile_types
#     known_profile_types = None
#     known_profile_names_str = args.known_profile_names
#     known_profile_names = None
#
#     if known_profiles_files:
#         known_profiles_files_list = known_profiles_files.split(',')
#         known_profiles_iq = [np.genfromtxt(f) for f in known_profiles_files_list]
#         if known_profile_types_str:
#             known_profile_types = known_profile_types_str.split(',')
#             if len(known_profile_types) != len(known_profiles_files_list):
#                 parser.error(
#                     "--known_profile_types must have the same number of entries as --known_profiles_files if both are provided.")
#         else:
#             known_profile_types = ['generic'] * len(known_profiles_files_list)
#         if known_profile_names_str:  # Parse known profile names if provided
#             known_profile_names = known_profile_names_str.split(',')
#             if len(known_profile_names) != len(known_profiles_files_list):
#                 parser.error(
#                     "--known_profile_names must have the same number of entries as --known_profiles_files if both are provided.")
#         else:
#             known_profile_names = [f"known_{i + 1}" for i in
#                                    range(len(known_profiles_files_list))]  # Default known profile names
#
#     # --- Parse penalty related arguments (now from command line) ---
#     water_peak_range_str = args.water_peak_range.split(',')
#     water_peak_range = tuple(float(qr) for qr in water_peak_range_str)
#     fractions_weight = args.fractions_weight
#     profile_similarity_weight = args.profile_similarity_weight
#     water_penalty_weight = args.water_penalty_weight
#
#
#     optimization_options = {  # Collect optimization options dictionary - ***MOVED UP HERE***
#         'maxiter': args.maxiter,
#         'ftol': args.ftol,
#         'maxfun': args.maxfun,
#         'maxls': args.maxls,
#         'eps': args.eps
#     }
#
#     # --- Prepare Inputs for ShannonDeconvolution (Corrected Order) ---
#     unknown_component_profiles_iq_init = None
#     n_components = len(unknown_component_Ds)
#     unknown_component_names = [f"component_{i + 1}" for i in range(n_components)]
#
#     # --- Instantiate ShannonDeconvolution ---
#     deconvolver = ShannonDeconvolution(
#         unknown_component_profiles_iq_init=unknown_component_profiles_iq_init,
#         unknown_component_Ds=unknown_component_Ds,
#         mixture_stack=mixture_stack,
#         q=q,
#         known_profiles_iq=known_profiles_iq,
#         known_profile_types=known_profile_types,
#         known_profile_names=known_profile_names,
#         fractions_weight=args.fractions_weight,
#         water_penalty_weight=args.water_penalty_weight,
#         profile_similarity_weight=args.profile_similarity_weight,
#         target_similarity=args.target_similarity,  # Initialize target_similarity to None initially
#         update_visualization=args.visualize,
#         verbose=args.verbose,
#         plot_update_frequency=args.plot_update_frequency,
#         optimization_method=args.optimization_method,
#         optimization_options=optimization_options
#     )
#
#     # --- Run Optimization ---
#     optimization_results = deconvolver.run_optimization()
#
#     print("\n--- Optimization Results ---")
#     print(optimization_results)
#
#     # --- Keep plot window open after optimization if visualization was enabled ---
#     if deconvolver.update_visualization:
#         print("\nPlotting is persistent. Close the plot window to exit.")
#         plt.show(block=True)
#     else:
#         print("\nPlotting was not enabled. Script finished.")


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


# if __name__ == '__main__':
#     import numpy as np
#     from scipy import optimize
#     from functools import partial
#
#     # --- Minimal Example Data and Initial Guess ---
#     mixture_stack = np.array([[1.0, 0.5, 0.2], [0.8, 0.4, 0.15]]) # Dummy 2x3 mixture stack
#     q = np.array([1.0, 2.0, 3.0]) # Dummy q-values
#     initial_params = np.array([1.0, 1.0]) # Dummy initial parameters
#     component_nsh = [2, 2] # Dummy component_nsh
#     component_Bns = [np.random.rand(2, len(q)), np.random.rand(2, len(q))] # Dummy Bns
#     background_stack = np.zeros((1, len(q))) # Dummy background_stack
#
#     # --- Define a Simple Objective Function (for testing) ---
#     def simple_score_function(params):
#         print("test")
#         residuals = np.sum(params**2) # Simple score: sum of squares of parameters
#         gradient = 2*params
#         return residuals, gradient
#
#     # --- Optimization Options with maxiter=2 ---
#     optimization_options = {
#         'maxiter': 3,
#         'ftol': 1e-8,
#         'maxfun': 10,
#         'maxls': 2,
#         'eps': 1e-8
#     }
#
#     # --- Run Optimization ---
#     results = optimize.minimize(
#         partial(simple_score_function), # Objective function
#         initial_params, # Initial guess
#         method='L-BFGS-B',
#         bounds=None, # No bounds for simple test
#         callback=None,
#         options=optimization_options,
#         jac=True,
#     )
#
#     print("\n--- Minimal Example Optimization Results ---")
#     print(results) # Print optimization results
