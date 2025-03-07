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
    """
    Calculates the cosine dissimilarity between two scattering profiles.

    Cosine similarity = 1 - cosine similarity.

    Args:
        profile1 (np.ndarray): First scattering profile (I(q) values).
        profile2 (np.ndarray): Second scattering profile (I(q) values).

    Returns:
        float: Cosine dissimilarity value (between 0 and 2, where 0 is identical, and 2 is maximally dissimilar).
               Returns NaN if either profile has zero norm.
    """
    norm_profile1 = np.linalg.norm(profile1)
    norm_profile2 = np.linalg.norm(profile2)

    if norm_profile1 == 0 or norm_profile2 == 0:
        return np.nan  # Return NaN if either profile has zero norm to avoid division by zero

    cosine_similarity = np.dot(profile1, profile2) / (norm_profile1 * norm_profile2)
    return cosine_similarity

class ShannonDeconvolution:
    def __init__(self,
                 unknown_component_profiles_iq_init,
                 unknown_component_Ds,
                 mixture_stack,
                 q,
                 known_profiles_iq=None,
                 unknown_component_names=None,
                 known_profile_names=None,
                 known_profile_types=None,
                 alpha=1e-8,
                 extrapolate_sasrec=False,
                 initial_sasrec_frames=None,
                 water_peak_range=(1.9, 2.0),
                 target_similarity=None,
                 fractions_weight=1.0,
                 water_penalty_weight=30.0,
                 profile_similarity_weight=10.0,
                 optimization_method='L-BFGS-B',
                 optimization_options=None,
                 use_basin_hopping=False,
                 basin_hopping_options=None,
                 parameter_bounds=None,
                 callback=None,
                 update_visualization=False,
                 plot_update_frequency=10,
                 verbose=True,):
        """
        Initializes the ShannonDeconvolution class.

        Args:
            unknown_component_profiles_iq_init (list of np.ndarray or list of str):
                Initial guess I(q) profiles for unknown components (q, I) or file paths.
            unknown_component_Ds (list or np.ndarray): D values for unknown components.
            mixture_stack (np.ndarray): 2D mixture stack data (N_frames x N_qbins).
            q (np.ndarray): q-values.
            known_profiles_iq (list of np.ndarray or list of str, optional):
                Known I(q) profiles (q, I) or file paths. Defaults to None.
            unknown_component_names (list of str, optional): Names for unknown components. Defaults to None.
            known_profile_names (list of str, optional): Names for known profiles. Defaults to None.
            alpha (float, optional): Sasrec alpha parameter. Defaults to 1e-8.
            extrapolate_sasrec (bool, optional): Whether to extrapolate Sasrec profiles. Defaults to True.
            initial_sasrec_frames (list of int, optional): Frames for initial Sasrec fits. Defaults to None.
            water_peak_range (tuple, optional): q-range for water peak penalty. Defaults to (1.9, 2.0).
            target_similarity (np.ndarray, optional): Target profile similarity to water. Defaults to None.
            fractions_weight (float, optional): Weight for sum-to-one fraction constraint. Defaults to 1.0.
            water_penalty_weight (float, optional): Weight for water background penalty. Defaults to 30.0.
            profile_similarity_weight (float, optional): Weight for profile dissimilarity penalty. Defaults to 10.0.
            optimization_method (str, optional): Optimization method. Defaults to 'L-BFGS-B'.
            optimization_options (dict, optional): Optimization options. Defaults to None.
            use_basin_hopping (bool, optional): Use Basin Hopping optimization. Defaults to False.
            basin_hopping_options (dict, optional): Basin Hopping options. Defaults to None.
            parameter_bounds (list of tuples, optional): Parameter bounds. Defaults to None.
            callback (callable, optional): Callback function for optimization. Defaults to None.
            update_visualization (bool, optional): Update visualization during optimization. Defaults to False.
            verbose (bool, optional): Verbose output. Defaults to True.
        """
        # --- Store parameters as attributes ---
        self.unknown_component_profiles_iq_init = unknown_component_profiles_iq_init
        self.unknown_component_Ds = unknown_component_Ds
        self.mixture_stack = mixture_stack
        self.q = q
        self.n_q = len(q)
        self.known_profiles_iq = known_profiles_iq
        self.unknown_component_names = unknown_component_names
        self.known_profile_names = known_profile_names
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

        # --- Initialize internal attributes (will be populated later) ---
        self.unknown_component_sasrecs = []
        self.unknown_component_Bns = []
        self.unknown_component_nsh = []
        self.initial_params = None
        self.known_profiles_Iq = []
        self.plot_counter = 0 # For visualization updates

        # --- Data Loading and Initialization (to be implemented next) ---
        self._load_profiles()

        self.known_profile_types = known_profile_types
        if self.known_profile_types is None: # Default to 'generic' if not provided
            if self.known_profiles_iq is not None:
                self.known_profile_types = ['generic'] * len(self.known_profiles_iq)
            else:
                self.known_profile_types = []
        else:
            if len(self.known_profile_types) != len(self.known_profiles_iq_loaded):
                raise ValueError("Length of known_profile_types must match the number of known profiles.")

        self._initialize_sasrecs_and_params()
        self._initialize_known_profiles()

        # --- Initialize parameter bounds ---
        if parameter_bounds is None: # Only create default bounds if not provided in __init__
            self.parameter_bounds = self._create_parameter_bounds_In_only()
        else:
            self.parameter_bounds = parameter_bounds # Use user-provided bounds if given

        if self.verbose:
            print("ShannonDeconvolution object initialized.")
            print(f"  Unknown Components: {self.unknown_component_names}")
            print(f"  Known Profiles:     {self.known_profile_names}")

        self.update_visualization = update_visualization

        if self.update_visualization: # Initialize plots only if update_visualization is True
            # --- Plotting Initialization ---
            global fig, ax # Use global if you want to keep it simple for now, or make fig and ax class attributes if preferred
            fig, ax = plt.subplots(2, 2, figsize=(12, 8))

            global line_true_lys, line_true_lys_noisy, line_true_cbp, line_true_cbp_noisy, line_fitted_lys, line_fitted_cbp # globals for lines - consider making class attributes
            # Static data (reference data that won't change)
            line_true_lys_noisy, = ax[0, 0].plot(q, lys_Iq[:,1]+noise[0], 'o', color='gray', alpha=0.3, mec='none', ms=4, label='True lys I(q)') # Assuming lys_Iq, noise are accessible here, if not, you might need to pass them to __init__ or load them here again.
            line_true_lys, = ax[0, 0].plot(q, lys_Iq[:,1], 'k-', alpha=0.8)
            line_true_cbp_noisy, = ax[0, 0].plot(q, cbp_Iq[:,1]+noise[0], '^', color='gray', alpha=0.3, mec='none', ms=4, label='True cbp I(q)') # Same for cbp_Iq
            line_true_cbp, = ax[0, 0].plot(q, cbp_Iq[:,1], 'k-', alpha=0.8)

            # Dynamic data (will be updated during optimization)
            line_fitted_lys, = ax[0, 0].plot(q, np.zeros_like(q), 'blue', label='Fitted lys I(q)')
            line_fitted_cbp, = ax[0, 0].plot(q, np.zeros_like(q), 'red', label='Fitted cbp I(q)')
            ax[0, 0].semilogy()
            ax[0, 0].set_ylabel("I(q)")
            ax[0, 0].set_xlabel("q")
            ax[0, 0].legend(loc='upper right')

            global line_fractions_lys_target, line_fractions_cbp_target, line_fractions_bkgd_target # globals for lines - consider making class attributes
            global line_fractions_lys, line_fractions_cbp, line_fractions_bkgd
            line_fractions_lys_target, = ax[0, 1].plot(np.arange(n_frames), frac_lys, 'k--', alpha=0.2) # Assuming frac_lys, frac_cbp, frac_water, n_frames are accessible or passed to __init__
            line_fractions_cbp_target, = ax[0, 1].plot(np.arange(n_frames), frac_cbp, 'k--', alpha=0.2)
            line_fractions_bkgd_target, = ax[0, 1].plot(np.arange(n_frames), frac_water, 'k--', alpha=0.2)
            line_fractions_lys, = ax[0, 1].plot(np.arange(n_frames), np.zeros(n_frames), label='lys fractions', color='blue')
            line_fractions_cbp, = ax[0, 1].plot(np.arange(n_frames), np.zeros(n_frames), label='cbp fractions', color='red')
            line_fractions_bkgd, = ax[0, 1].plot(np.arange(n_frames), np.zeros(n_frames), label='bkgd fractions', color='orange')
            ax[0, 1].set_ylabel("Fractions")
            ax[0, 1].set_xlabel("Frames")
            ax[0, 1].set_ylim([-0.1,1.2])
            ax[0, 1].legend(loc='upper center')

            # Third subplot - Example fit to data
            global fit_frame_number, line_example_fit_data, line_example_fit_total, line_example_fit_lys, line_example_fit_cbp, line_example_fit_bkgd # globals for lines - consider making class attributes
            fit_frame_number = n_frames//2 # Assuming n_frames is accessible
            line_example_fit_data, = ax[1, 0].plot(q, mixture_stack[fit_frame_number], 'o', color='gray', alpha=0.3, mec='none', ms=4, label=f'Frame {fit_frame_number}') # Assuming mixture_stack, q are accessible
            line_example_fit_total, = ax[1, 0].plot(q, np.zeros(self.n_q), color='green', label=f'total') # Assuming n_q is accessible
            line_example_fit_lys, = ax[1, 0].plot(q, np.zeros(self.n_q), color='blue', label=f'lys')
            line_example_fit_cbp, = ax[1, 0].plot(q, np.zeros(self.n_q), color='red', label=f'cbp')
            line_example_fit_bkgd, = ax[1, 0].plot(q, np.zeros(self.n_q), color='orange', label=f'bkgd')
            ax[1, 0].legend(loc='upper center')
            # ax[1, 0].semilogy() # Keep linear scale for now

            # Fourth subplot - Residuals as 2D image
            global residuals_img # global for image - consider making class attribute
            # Initialize with zeros - dimensions will be n_frames x n_qbins
            initial_residuals = np.zeros((n_frames, len(q))) # Assuming n_frames, q are accessible
            residuals_img = ax[1, 1].imshow(initial_residuals, aspect='auto', cmap='viridis',
                                        origin='lower', interpolation='nearest')
            ax[1, 1].set_title("Residuals")
            ax[1, 1].set_xlabel("q bin")
            ax[1, 1].set_ylabel("Frame")
            fig.colorbar(residuals_img, ax=ax[1,1], label="Residual Magnitude")

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.01) # Small pause to make sure window is shown

            # Initialize counter for plot updates
            global plot_counter # global for counter - consider making class attribute
            plot_counter = 0
            global last_update_time # global for time - consider making class attribute
            last_update_time = time.time()


    def _load_profiles(self):
        """Loads initial guess profiles and known profiles from arrays."""
        self.unknown_component_profiles_iq_init_loaded = []
        for profile_iq in self.unknown_component_profiles_iq_init:
            if isinstance(profile_iq, str): # Check if it's a file path (not implemented yet, for future)
                raise NotImplementedError("Loading profiles from file paths is not yet implemented.")
            elif isinstance(profile_iq, np.ndarray):
                if profile_iq.shape[1] < 2:
                    raise ValueError("Initial guess profiles must be NumPy arrays with shape (N, 2) [q, I].")
                self.unknown_component_profiles_iq_init_loaded.append(profile_iq)
            else:
                raise TypeError("Initial guess profiles must be NumPy arrays or file paths (string).")

        self.known_profiles_iq_loaded = []
        if self.known_profiles_iq is not None: # Known profiles are optional
            for profile_iq in self.known_profiles_iq:
                if isinstance(profile_iq, str): # Check if it's a file path (not implemented yet, for future)
                    raise NotImplementedError("Loading known profiles from file paths is not yet implemented.")
                elif isinstance(profile_iq, np.ndarray):
                    if profile_iq.shape[1] != 2:
                        raise ValueError("Known profiles must be NumPy arrays with shape (N, 2) [q, I].")
                    self.known_profiles_iq_loaded.append(profile_iq)
                else:
                    raise TypeError("Known profiles must be NumPy arrays or file paths (string).")
        else:
            self.known_profiles_iq_loaded = [] # Initialize as empty list if None is provided

    def _initialize_sasrecs_and_params(self):
        """Initializes Sasrecs for unknown components and sets initial parameters."""
        if self.initial_sasrec_frames is None:
            # Default frames: evenly spaced
            n_frames_init = len(self.unknown_component_profiles_iq_init_loaded)
            initial_sasrec_frames = np.linspace(0, self.mixture_stack.shape[0] - 1, n_frames_init)
            self.initial_sasrec_frames = [int(round(frame)) for frame in initial_sasrec_frames]
        elif len(self.initial_sasrec_frames) != len(self.unknown_component_profiles_iq_init_loaded):
            raise ValueError("Length of initial_sasrec_frames must match the number of unknown components.")

        water_profile_Iq = None # Initialize water profile to None
        print(self.known_profile_types)
        for i_known_profile, profile_type in enumerate(self.known_profile_types):
            if profile_type == 'water':
                water_profile_Iq = self.known_profiles_iq_loaded[i_known_profile] # Found water profile

        for i_component, profile_iq in enumerate(self.unknown_component_profiles_iq_init_loaded):
            initialization_frame = self.initial_sasrec_frames[i_component]
            print(f"Initializing Sasrec for component {self.unknown_component_names[i_component]} using frame {initialization_frame}")
            initialization_profile = np.copy(self.mixture_stack[initialization_frame])

            # Subtract water background roughly (same as in original script)
            if water_profile_Iq is not None: # Use the water_profile_Iq we found
                water_peak_mini = denss.find_nearest_i(self.q, self.water_peak_range[0])
                water_peak_maxi = denss.find_nearest_i(self.q, self.water_peak_range[1])
                initialization_profile_water_peak_int = np.mean(initialization_profile[water_peak_mini:water_peak_maxi])
                water_peak_int = np.mean(water_profile_Iq[:,1][water_peak_mini:water_peak_maxi])
                if water_peak_int > 0:
                    water_background_sf = initialization_profile_water_peak_int / water_peak_int
                    initialization_profile -= water_profile_Iq[:,1] * (water_background_sf * 0.99)
                else:
                    print("Warning: Water profile has zero intensity in the water peak region. Water background subtraction skipped.")

            initialization_Iq = np.vstack((self.q, initialization_profile, profile_iq[:, 2])).T
            sasrec = denss.Sasrec(initialization_Iq, D=self.unknown_component_Ds[i_component], qc=self.q, alpha=self.alpha, extrapolate=self.extrapolate_sasrec)
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

    def _calculate_profile_dissimilarity_penalty(self, estimated_profiles):
        """
        Calculates the profile dissimilarity penalty (similarity to water),
        finding water profile from known profiles based on type.

        Args:
            estimated_profiles (np.ndarray): Estimated profiles matrix (n_total_profiles x n_qbins).

        Returns:
            float: Profile dissimilarity penalty value.
        """
        profile_similarity_error = 0.0  # Initialize to zero
        if self.target_similarity is not None:
            n_unknown_components = len(self.unknown_component_nsh)
            profile_similarity_to_water = np.zeros(n_unknown_components)
            water_I = None # Initialize water_I to None

            for i_known_profile, profile_type in enumerate(self.known_profile_types): # Find water profile
                if profile_type == 'water':
                    water_I = self.known_profiles_Iq[i_known_profile] # Found water profile, extract I(q) values
                    break # Assuming only one water profile, exit loop once found

            if water_I is not None: # Proceed only if water profile is found
                for i_component in range(n_unknown_components):
                    estimated_profile = estimated_profiles[i_component]
                    profile_similarity_to_water[i_component] = calculate_cosine_similarity(estimated_profile, water_I)

                profile_dissimilarity_error = np.sum(np.sqrt((profile_similarity_to_water - self.target_similarity)**2)) # Calculate error

        profile_similarity_penalty = profile_similarity_error * self.profile_similarity_weight # Apply weight
        return profile_similarity_penalty

    def calc_score_from_In(self, params, update_viz=False):
        """Calculates the score function from In parameters."""
        n_frames = self.mixture_stack.shape[0]
        estimated_profiles = self._calculate_profiles_from_params(params) # Get profiles
        estimated_fractions = self._calculate_fractions(estimated_profiles) # Get fractions

        n_qbins = self.mixture_stack.shape[1] # Still need n_qbins (not directly used in penalties, but might be later)
        # n_unknown_components = len(self.unknown_component_nsh) # Not directly used in profile dissimilarity penalty anymore

        # 4. Calculate Mixture
        calculated_mixture = generate_evolving_mixture_stack(profiles=estimated_profiles, fractions=estimated_fractions)

        # 5. Calculate Residuals
        residuals = self._calculate_residuals(calculated_mixture)

        # --- 6. Penalties ---
        # a. Sum-to-One Fraction Constraint Penalty
        fraction_sum_penalty = self._calculate_fraction_sum_penalty(estimated_fractions)

        # b. Water Penalty
        water_penalty = self._calculate_water_penalty(estimated_fractions)

        # c. Profile Dissimilarity Penalty (using the new method)
        profile_dissimilarity_penalty = self._calculate_profile_dissimilarity_penalty(estimated_profiles)

        # 7. Total Score (rest of the original score aggregation and output)
        score = residuals + fraction_sum_penalty + water_penalty + profile_dissimilarity_penalty

        if self.verbose:
            sys.stdout.write(f"\rStep {self.plot_counter:06d} {residuals:.6e} {fraction_sum_penalty:.6e} {water_penalty:.6e} {profile_dissimilarity_penalty:.6e} {score:.6e} ") # Updated verbose output for profile_dissimilarity_penalty
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

    def update_plot_simple(self, estimated_profiles, estimated_fractions):
        """Updates the plot with current profiles and fractions."""
        global line_fitted_lys, line_fitted_cbp, line_fractions_lys, line_fractions_cbp, line_fractions_bkgd # Globals for lines
        global line_example_fit_total, line_example_fit_lys, line_example_fit_cbp, line_example_fit_bkgd # Globals for example fit lines
        global residuals_img # Global for residuals image
        global fig, ax # Globals for figure and axes
        global fit_frame_number # Global for frame number

        # Update the line data
        line_fitted_lys.set_ydata(estimated_profiles[0])
        line_fitted_cbp.set_ydata(estimated_profiles[1])
        line_fractions_lys.set_ydata(estimated_fractions[0])
        line_fractions_cbp.set_ydata(estimated_fractions[1])
        line_fractions_bkgd.set_ydata(estimated_fractions[2])

        # Calculate and update the residuals image
        calculated_mixture = generate_evolving_mixture_stack(
            profiles=estimated_profiles,
            fractions=estimated_fractions
        )
        frame_residuals = self.mixture_stack - calculated_mixture # Use self.mixture_stack

        line_example_fit_total.set_ydata(calculated_mixture[fit_frame_number])
        line_example_fit_lys.set_ydata(estimated_profiles[0]*estimated_fractions[0][fit_frame_number])
        line_example_fit_cbp.set_ydata(estimated_profiles[1]*estimated_fractions[1][fit_frame_number])
        line_example_fit_bkgd.set_ydata(estimated_profiles[2]*estimated_fractions[2][fit_frame_number])

        # Update the image data
        residuals_img.set_array(frame_residuals)
        residuals_img.set_clim(vmin=-0.01, vmax=0.01) # Keep fixed color limits for now

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
        default_options = {
            'maxiter': 100,     # Default max iterations (adjust as needed for testing)
            'maxfun': 1e5,
            'ftol': 1e-8,
            'maxls': 50,
            'eps': 1e-8
        }
        optimization_options = {**default_options, **optimization_options} # Merge default and user options, user options take precedence

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
    # --- Load Data (replace with your actual data loading) ---
    lys_Iq = np.genfromtxt("SASDCK8_fit1_model1_FH_pdb.dat")
    cbp_Iq = np.genfromtxt("4fe9_FH_pdb.dat")
    # rescale to have same I(0) for now
    lys_Iq[:, 1] /= lys_Iq[0, 1]
    lys_Iq[:, 2] /= lys_Iq[0, 1]  # scale errors too
    cbp_Iq[:, 1] /= cbp_Iq[0, 1]
    cbp_Iq[:, 2] /= cbp_Iq[0, 1]  # scale errors too
    # make cbp 2x more scattering due to MW
    sf = 2.0
    cbp_Iq[:, 1] *= sf
    cbp_Iq[:, 2] *= sf  # scale errors too
    water = np.genfromtxt("water.dat")
    q = lys_Iq[:,0]
    water_I = np.interp(q, water[:,0], water[:,1])
    # should be on the same q bins, but just in case interpolate
    water_I = np.interp(q, water[:, 0], water[:, 1])
    water_I *= lys_Iq[:,1].sum() / water_I.sum()  # just put on roughly the same scale

    # --- Synthetic Mixture Stack (replace with your actual mixture data) ---
    component_profiles = np.vstack((lys_Iq[:,1], cbp_Iq[:,1]))
    n_frames = 50
    n_components = component_profiles.shape[0]
    frames = np.arange(n_frames)
    frac_lys = 1-sigmoid_fraction_model(frames)
    frac_cbp = 1-frac_lys
    frac_water = np.random.normal(loc=0.4, scale=0.1, size=frac_cbp.shape)
    component_fractions = np.vstack((frac_lys, frac_cbp))
    background_fractions = [frac_water]
    background_stack = np.zeros((1, len(q)))
    background_stack[0] = water_I
    profiles = np.vstack((component_profiles, background_stack))
    fractions = np.vstack((component_fractions, background_fractions))
    mixture_stack = generate_evolving_mixture_stack(profiles, fractions)
    noise = np.random.normal(0, scale=lys_Iq[:,2]*5e-8, size=mixture_stack.shape)
    mixture_stack += noise

    # --- Prepare Inputs for ShannonDeconvolution ---
    unknown_component_profiles_iq_init = [lys_Iq, cbp_Iq]
    unknown_component_Ds = [50.0, 100.0]
    known_profiles_iq = [np.vstack((q, water_I)).T]
    unknown_component_names = ["lys", "cbp"]
    known_profile_names = ["water"]
    known_profile_types = ['water']  # Specify the type of the known profile

    # --- Calculate target_similarity ---
    # We'll need to come up with a generic solution for this. For example, alphafold prediction or something
    # The nice thing is the similarity measure is similar for different protein conformations
    target_similarity = np.zeros(n_components) # Initialize for 2 components (lys, cbp)
    profiles_for_similarity = np.vstack((lys_Iq[:,1], cbp_Iq[:,1]))
    for i_component in range(n_components):  # Loop through each component profile
        # Cosine similarity between component profile and water profile
        profile_similarity_to_water = calculate_cosine_similarity(profiles_for_similarity[i_component], water_I)
        target_similarity[i_component] = profile_similarity_to_water
        # add a small amount of variability to the target to account for conformational diversity,
        # it seems the same to about std=0.0012 in my tests with different conformations of the same protein
        # so to avoid total bias, try and add some random error to simulate a different conformation
        target_similarity[i_component] += np.random.normal(loc=0, scale=0.0012)

    # --- Instantiate ShannonDeconvolution ---
    deconvolver = ShannonDeconvolution(
        unknown_component_profiles_iq_init=unknown_component_profiles_iq_init,
        unknown_component_Ds=unknown_component_Ds,
        mixture_stack=mixture_stack,
        q=q,
        known_profiles_iq=known_profiles_iq,
        unknown_component_names=unknown_component_names,
        known_profile_names=known_profile_names,
        known_profile_types=known_profile_types,  # Pass the known_profile_types
        fractions_weight=1.0,
        water_penalty_weight=0.0,
        profile_dissimilarity_weight=10.0,
        target_similarity=target_similarity,
        update_visualization=True,
        verbose=True
    )

    print("\nShannonDeconvolution object successfully created.")
    print(f"Number of unknown components: {len(deconvolver.unknown_component_names)}")
    print(f"Number of known profiles: {len(deconvolver.known_profile_names)}")

    # --- Run Optimization ---
    optimization_results = deconvolver.run_optimization()

    print("\n--- Optimization Results ---")  # Separator for clarity
    print(optimization_results)  # Print the full results object

    # --- Keep plot window open after optimization if visualization was enabled ---
    if deconvolver.update_visualization:
        print("\nPlotting is persistent. Close the plot window to exit.")
        plt.show(block=True)  # Keep plot window open until closed manually
    else:
        print("\nPlotting was not enabled. Script finished.")
