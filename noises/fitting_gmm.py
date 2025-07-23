import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import stats
import json
from config import Flow_info

# TFP distribution and bijector shortcuts
tfd = tfp.distributions
tfb = tfp.bijectors

# -----------------------------------------------------------------------------
# 1. Configuration and Hyperparameters
# -----------------------------------------------------------------------------
class TrainingConfig:
    """Configuration class for storing all hyperparameters and target values"""
    
    def __init__(self, target_moments=None):
        """
        Initialize configuration
        
        Args:
            target_moments (dict): Target moments dictionary containing mean, variance, skewness, kurtosis
        """
        # Use default values if no target moments provided
        if target_moments is None:
            target_moments = {
                'mean': 0.0,
                'variance': 4.45186e14,
                'skewness': 23.062561,
                'kurtosis': 2601.00122
            }
        
        self.TARGET_MOMENTS = target_moments
        
        # Model configuration - reduce components for faster training
        self.NUM_COMPONENTS = 3 
        
        # Optimization configuration - cosine annealing learning rate
        self.MAX_LEARNING_RATE = 0.01  # Maximum learning rate
        self.MIN_LEARNING_RATE = 0.001  # Minimum learning rate
        self.TRAINING_STEPS = 5000  
        
        # Random seed for reproducibility
        self.RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# 2. GMM Model Wrapper Class
# -----------------------------------------------------------------------------
class MomentMatchingGMM:
    """
    A class that encapsulates the GMM and its moment matching logic.
    """
    def __init__(self, num_components, config):
        """
        Initialize the model's trainable parameters and TFP distribution objects.
        Intelligently set initial values based on target moment characteristics.
        
        Args:
            num_components (int): Number of Gaussian components in the mixture model.
            config (TrainingConfig): Training configuration object
        """
        self.num_components = num_components
        self.config = config
        
        # Calculate reference values for initialization based on target moments
        target_mean = config.TARGET_MOMENTS['mean']
        target_variance = config.TARGET_MOMENTS['variance']
        target_skewness = config.TARGET_MOMENTS['skewness']
        target_stddev = np.sqrt(target_variance)
        
        # weights are parameterized by logits, means and stddevs are parameterized directly
        
        # 1. Means: unconstrained, random initialization, make initial means more dispersed
        self.locs = tf.Variable(
            np.random.normal(0, 3, num_components).astype(np.float32),
            name='locs'
        )
        
        # 2. Stddev: use positive values directly (ensure positive during training with abs)
        # Use a fixed small initial value to avoid training difficulties
        initial_scale = 10.0
        self.scales = tf.Variable(
            np.ones(num_components, dtype=np.float32) * initial_scale,
            name='scales'
        )
        
        # 3. Weights: parameterized by logits, converted to weights by softmax
        self.logits = tf.Variable(
            np.zeros(num_components, dtype=np.float32),
            name='logits'
        )
        
        # Collect trainable variables
        self.trainable_variables = [self.locs, self.scales, self.logits]
        
        # Print initialization information
        print(f"Model initialization completed (simplified version following gmm.txt)")

    @tf.function
    def get_gmm_distribution(self):
        """
        Build and return TFP's GMM distribution based on current trainable variables.
        weights use logits+softmax, stddevs use abs to ensure positivity.
        """
        # Weights: logits converted to weights by softmax
        weights = tf.nn.softmax(self.logits)
        # Stddev: ensure positivity
        positive_scales = tf.abs(self.scales)
        
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=weights),
            components_distribution=tfd.Normal(loc=self.locs, scale=positive_scales)
        )
    
# -----------------------------------------------------------------------------
# 3. Moment Calculation Functions
# -----------------------------------------------------------------------------
@tf.function
def get_analytical_moments(gmm_dist):
    """Calculate mean and variance using TFP's built-in methods."""
    mean = gmm_dist.mean()
    variance = gmm_dist.variance()
    return mean, variance



@tf.function
def get_closed_form_moments(gmm_dist):
    """
    Calculate GMM's skewness and kurtosis using closed-form solutions.
    Based on mathematical formulas in gaussian mixture.md.
    
    Args:
        gmm_dist: TFP's MixtureSameFamily distribution object
        
    Returns:
        tuple: (skewness, kurtosis) - closed-form solutions for skewness and kurtosis
    """
    # Extract parameters from GMM distribution
    # Weights π_k (get from softmax probabilities)
    weights = gmm_dist.mixture_distribution.probs_parameter()  # π_k
    
    # Means μ_k and variances σ_k^2
    component_means = gmm_dist.components_distribution.mean()  # μ_k
    component_variances = gmm_dist.components_distribution.variance()  # σ_k^2
    component_stddevs = tf.sqrt(component_variances)  # σ_k
    
    # Calculate GMM's overall mean μ = Σ π_k μ_k
    gmm_mean = tf.reduce_sum(weights * component_means)
    
    # Calculate GMM's overall variance
    # Var(X) = Σ π_k (σ_k^2 + μ_k^2) - (Σ π_k μ_k)^2
    gmm_variance = tf.reduce_sum(weights * (component_variances + tf.square(component_means))) - tf.square(gmm_mean)
    gmm_stddev = tf.sqrt(gmm_variance + 1e-10)  # Add small constant to prevent numerical instability
    
    # Calculate skewness numerator: third central moment
    # E[(X - μ)^3] = Σ π_k [(μ_k - μ)^3 + 3 σ_k^2 (μ_k - μ)]
    mean_diff = component_means - gmm_mean  # (μ_k - μ)
    third_central_moment = tf.reduce_sum(
        weights * (tf.pow(mean_diff, 3) + 3 * component_variances * mean_diff)
    )
    
    # Skewness = E[(X - μ)^3] / σ^3
    skewness = third_central_moment / tf.pow(gmm_stddev, 3)
    
    # Calculate kurtosis numerator: fourth central moment
    # E[(X - μ)^4] = Σ π_k [3 σ_k^4 + 6 σ_k^2 (μ_k - μ)^2 + (μ_k - μ)^4]
    fourth_central_moment = tf.reduce_sum(
        weights * (
            3 * tf.pow(component_variances, 2) +  # 3 σ_k^4
            6 * component_variances * tf.square(mean_diff) +  # 6 σ_k^2 (μ_k - μ)^2
            tf.pow(mean_diff, 4)  # (μ_k - μ)^4
        )
    )
    
    # Kurtosis = E[(X - μ)^4] / σ^4
    # Note: This calculates ordinary kurtosis, not excess kurtosis (doesn't subtract 3)
    # Because kurtosis in TARGET_MOMENTS is ordinary kurtosis
    kurtosis = fourth_central_moment / tf.pow(gmm_variance, 2)
    
    return skewness, kurtosis


@tf.function
def compute_empirical_moments(distribution, num_samples=200000):
    """
    Compute empirical statistics of the distribution (by sampling)
    
    Args:
        distribution: TFP distribution object
        num_samples: Number of samples
        
    Returns:
        dict: Dictionary containing mean, variance, skewness, kurtosis
    """
    # Sampling
    samples = distribution.sample(num_samples)
    
    # Compute empirical statistics
    mean = tf.reduce_mean(samples)
    variance = tf.reduce_mean(tf.square(samples - mean))
    
    # Standardize samples
    std = tf.sqrt(variance)
    standardized = (samples - mean) / std
    
    # Compute skewness and kurtosis
    skewness = tf.reduce_mean(tf.pow(standardized, 3))
    kurtosis = tf.reduce_mean(tf.pow(standardized, 4))
    
    return {
        'mean': mean,
        'variance': variance,
        'skewness': skewness,
        'kurtosis': kurtosis
    }

# -----------------------------------------------------------------------------
# 4. Tensor Utility Functions
# -----------------------------------------------------------------------------
def create_tensor_dict(tensor_dict, convert_to_float=False):
    """
    Create a dictionary with tensor values, optionally converting to float
    
    Args:
        tensor_dict (dict): Dictionary with tensor values
        convert_to_float (bool): Whether to convert tensors to float
        
    Returns:
        dict: Dictionary with processed values
    """
    if convert_to_float:
        return {key: float(value.numpy()) for key, value in tensor_dict.items()}
    else:
        return tensor_dict

def safe_tensor_comparison(tensor_val, scalar_val):
    """
    Safely compare tensor value with scalar, converting only once
    
    Args:
        tensor_val: Tensor value
        scalar_val: Scalar value to compare
        
    Returns:
        tuple: (tensor_as_numpy, comparison_result)
    """
    numpy_val = tensor_val.numpy()
    return numpy_val, numpy_val < scalar_val

# -----------------------------------------------------------------------------
# 5. Loss Function and Training Step
# -----------------------------------------------------------------------------
@tf.function
def compute_loss(model_moments, target_moments):
    """
    Calculate loss between model moments and target moments.
    Only fit skewness and kurtosis as per gmm.txt specification.
    Mean and variance will be adjusted through linear transformation later.
    """
    # Skewness loss: relative error on skewness
    skewness_diff = (model_moments['skewness'] 
                     - target_moments['skewness']) / target_moments['skewness']
    skewness_loss = tf.square(skewness_diff)
    
    # Kurtosis loss: relative error on kurtosis
    kurtosis_diff = (model_moments['kurtosis'] 
                     - target_moments['kurtosis']) / target_moments['kurtosis']
    kurtosis_loss = tf.square(kurtosis_diff)
    
    # Only optimize skewness and kurtosis
    total_loss = (skewness_loss + kurtosis_loss) / 2.0
    
    return total_loss

@tf.function
def train_step_core(model, target_moments):
    """
    Core computation part of training step (inside tf.function)
    """
    with tf.GradientTape() as tape:
        # 1. Get current GMM distribution
        gmm = model.get_gmm_distribution()
        
        # 2. Calculate required moments (using closed-form solutions)
        mean, variance = get_analytical_moments(gmm)
        skewness, kurtosis = get_closed_form_moments(gmm)
        
        model_moments = {
            'mean': mean,
            'variance': variance,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
        
        # 3. Calculate loss
        loss = compute_loss(model_moments, target_moments)
        
    # 4. Calculate gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    return loss, model_moments, gradients

def train_step(model, optimizer, target_moments):
    """
    Execute a single training iteration.
    
    Args:
        model (MomentMatchingGMM): Our GMM model instance.
        optimizer (tf.keras.optimizers.Optimizer): TF optimizer.
        target_moments (dict): Dictionary containing target moments.
        
    Returns:
        tuple: Tuple containing loss and computed model moments.
    """
    # Call tf.function optimized core computation
    loss, model_moments, gradients = train_step_core(model, target_moments)
    
    # Apply gradients outside tf.function
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss, model_moments

# -----------------------------------------------------------------------------
# 5. Training Related Functions
# -----------------------------------------------------------------------------

def create_optimizer_and_scheduler(config):
    """
    Create optimizer and learning rate scheduler with cosine annealing
    
    Args:
        config (TrainingConfig): Training configuration object
        
    Returns:
        tf.keras.optimizers.Optimizer: Configured Adam optimizer with cosine annealing
    """
    # Use cosine annealing learning rate scheduler
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=config.MAX_LEARNING_RATE,
        first_decay_steps=config.TRAINING_STEPS,
        t_mul=1.0,  # Multiplier for the decay period
        m_mul=1.0,  # Multiplier for the initial learning rate
        alpha=config.MIN_LEARNING_RATE / config.MAX_LEARNING_RATE  # Minimum learning rate ratio
    )
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    return adam_optimizer

def train_model(model, optimizer, target_moments, config):
    """
    Execute complete training loop
    
    Args:
        model (MomentMatchingGMM): GMM model instance
        optimizer (tf.keras.optimizers.Optimizer): Optimizer
        target_moments (dict): Target moments dictionary
        config (TrainingConfig): Training configuration object
        
    Returns:
        tuple: (loss_history, best_loss, best_step) Training history and best results
    """
    print("Starting training...")
    print("Following gmm.txt specification: fitting only skewness and kurtosis")
    print(f"Target moments to fit: Skew={target_moments['skewness']:.4f}, "
          f"Kurt={target_moments['kurtosis']:.4f}")
    print(f"Target mean and variance will be matched via linear transformation later:")
    print(f"  Mean={target_moments['mean']:.4f}, Var={target_moments['variance']:.4e}")
    print("-" * 100)

    # Record training history
    loss_history = []
    best_loss = float('inf')
    best_step = 0
    
    for step in range(config.TRAINING_STEPS):
        loss_val, moments_val = train_step(
            model,
            optimizer,
            target_moments
        )
        
        # Convert to numpy only once for loss tracking using utility function
        loss_numpy, is_better = safe_tensor_comparison(loss_val, best_loss)
        loss_history.append(loss_numpy)
        
        # Record best results
        if is_better:
            best_loss = loss_numpy
            best_step = step
        
        # Detailed training progress output
        if step % 200 == 0 or step == config.TRAINING_STEPS - 1:
            current_lr = optimizer.learning_rate.numpy()
            print(f"Step {step:5d} | Loss: {loss_numpy:.6f} | LR: {current_lr:.6f}")
            print(f"         | Current fitted moments (before linear transformation):")
            print(f"         | Mean: {moments_val['mean'].numpy():.4f}, Var: {moments_val['variance'].numpy():.4e}")
            print(f"         | Skew: {moments_val['skewness'].numpy():.4f} (Target: {target_moments['skewness']:.4f})")
            print(f"         | Kurt: {moments_val['kurtosis'].numpy():.4f} (Target: {target_moments['kurtosis']:.4f})")
            print("-" * 100)

    print("Training completed.")
    print(f"Best loss: {best_loss:.6f} (achieved at step {best_step})")
    print("-" * 100)
    
    return loss_history, best_loss, best_step

def postprocess_model(model, target_moments):
    """
    Post-training processing: apply linear transformation y = ax + b to match target mean and variance
    Following gmm.txt specification: only fit skewness and kurtosis, then scale to match mean and variance
    
    Args:
        model (MomentMatchingGMM): Trained GMM model
        target_moments (dict): Target moments dictionary
        
    Returns:
        tuple: (final_gmm, sorted_weights, sorted_means, sorted_stddevs) 
               Processed distribution and sorted parameters
    """
    # Get current distribution fitted on skewness and kurtosis
    fitted_gmm = model.get_gmm_distribution()
    
    # Calculate current moments as tensors
    current_mean_tensor = fitted_gmm.mean()
    current_variance_tensor = fitted_gmm.variance()
    current_std_tensor = tf.sqrt(current_variance_tensor)
    
    # Target values as tensors
    target_mean_tensor = tf.constant(target_moments['mean'], dtype=tf.float32)
    target_std_tensor = tf.sqrt(tf.constant(target_moments['variance'], dtype=tf.float32))
    
    # Calculate linear transformation parameters y = ax + b using tensors
    # a = σ_target / σ_x,  b = μ_target - a * μ_x
    a_tensor = target_std_tensor / current_std_tensor
    b_tensor = target_mean_tensor - a_tensor * current_mean_tensor
    
    # Convert to numpy only for printing
    current_mean_np = current_mean_tensor.numpy()
    current_std_np = current_std_tensor.numpy()
    target_mean_np = target_mean_tensor.numpy()
    target_std_np = target_std_tensor.numpy()
    a_np = a_tensor.numpy()
    b_np = b_tensor.numpy()
    
    print(f"Applied linear transformation: y = {a_np:.6f} * x + {b_np:.4f}")
    print(f"  Current: mean={current_mean_np:.4f}, std={current_std_np:.4e}")
    print(f"  Target:  mean={target_mean_np:.4f}, std={target_std_np:.4e}")
    
    # Use TFP's TransformedDistribution for linear transformation
    # linear transformation y = ax + b is directly applied to the distribution
    # Create bijector for linear transformation: y = ax + b, use tensor directly
    # Note: Chain executes bijectors in REVERSE order, so we put Shift first, then Scale
    linear_transform = tfb.Chain([
        tfb.Shift(shift=b_tensor),     # Second: shift by b
        tfb.Scale(scale=a_tensor)      # First: scale by a  
    ])  # Result: y = (ax) + b = ax + b
    
    # Apply linear transformation to get the final distribution
    final_gmm = tfd.TransformedDistribution(
        distribution=fitted_gmm,
        bijector=linear_transform
    )
    
    print("-" * 100)
    
    # For parameter output, manually transform original parameters (for display and JSON output only)
    # Note: These parameters are for compatibility with existing output format, actual distribution uses final_gmm
    # Keep as tensor for computation
    original_weights_tensor = tf.nn.softmax(model.logits)  # weights unchanged
    original_means_tensor = model.locs
    original_stddevs_tensor = tf.abs(model.scales)
    
    # Manually apply linear transformation to parameters (for display only), use tensor computation
    transformed_means_tensor = a_tensor * original_means_tensor + b_tensor
    transformed_stddevs_tensor = tf.abs(a_tensor) * original_stddevs_tensor
    
    # Only convert to numpy for sorting
    transformed_means_np = transformed_means_tensor.numpy()
    original_weights_np = original_weights_tensor.numpy()
    transformed_stddevs_np = transformed_stddevs_tensor.numpy()
    
    # Sort results to get deterministic output (solve label switching problem)
    sort_indices = np.argsort(transformed_means_np)
    sorted_weights = original_weights_np[sort_indices]
    sorted_means = transformed_means_np[sort_indices]
    sorted_stddevs = transformed_stddevs_np[sort_indices]

    print("Final GMM parameters after linear transformation (sorted by mean):")
    print("(Note: These parameters are for display only. Actual distribution uses TransformedDistribution)")
    for i in range(model.num_components):
        print(f"  Component {i+1}: Weight={sorted_weights[i]:.4f}, "
              f"Mean={sorted_means[i]:.4f}, "
              f"StdDev={sorted_stddevs[i]:.4e}")
    
    print("-" * 100)
    
    return final_gmm, sorted_weights, sorted_means, sorted_stddevs, a_np, b_np



def print_training_tips():
    """
    Print training tips
    """
    print("Training tips:")
    print("1. Following gmm.txt specification:")
    print("   - Only fit skewness and kurtosis during training")
    print("   - Mean and variance are matched via linear transformation y = ax + b")
    print("   - Skewness and kurtosis remain unchanged during linear transformation")
    print("2. If loss doesn't decrease significantly, try:")
    print("   - Increase number of components (NUM_COMPONENTS)")
    print("   - Adjust learning rate (MAX_LEARNING_RATE)")
    print("   - Increase training steps (TRAINING_STEPS)")
    print("3. Focus on skewness and kurtosis fitting - mean and variance will be automatically corrected")

def fit_single_variable(var_name, var_data, suffix=""):
    """
    Fit GMM model for a single variable
    
    Args:
        var_name (str): Variable name
        var_data (dict): Variable data dictionary
        suffix (str): Suffix (for pos/neg distinction)
        
    Returns:
        dict: Dictionary containing fitting results, including weights, means, stds
    """
    print(f"\n{'='*80}")
    print(f"Starting to fit variable: {var_name}{suffix}")
    print(f"{'='*80}")
    
    # Build target moments dictionary
    target_moments = {
        'mean': var_data[f'mean{suffix}'],
        'variance': var_data[f'variance{suffix}'],
        'skewness': var_data[f'skewness{suffix}'],
        'kurtosis': var_data[f'kurtosis{suffix}']
    }
    
    # Create configuration object
    config = TrainingConfig(target_moments=target_moments)
    
    # Clear previous computation graph state
    tf.keras.backend.clear_session()
    
    # Set random seed
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Initialize model
    gmm_model = MomentMatchingGMM(num_components=config.NUM_COMPONENTS, config=config)
    
    # Create optimizer
    optimizer = create_optimizer_and_scheduler(config)
    
    # Train model
    loss_history, best_loss, best_step = train_model(
        gmm_model, 
        optimizer, 
        target_moments,
        config
    )
    
    # Post-process model
    final_gmm, sorted_weights, sorted_means, sorted_stddevs, a, b = postprocess_model(
        gmm_model, 
        target_moments
    )
    
    # Compute empirical statistics of the final distribution after linear transformation
    empirical_stats = compute_empirical_moments(final_gmm, num_samples=200000)
    
    # Get fitted skewness and kurtosis (from training process)
    fitted_gmm_before_transform = gmm_model.get_gmm_distribution()
    fitted_skewness, fitted_kurtosis = get_closed_form_moments(fitted_gmm_before_transform)
    
    # Use utility function to delay numpy conversion, only convert when outputting
    empirical_stats_dict = create_tensor_dict(empirical_stats, convert_to_float=True)
    
    fitted_stats = create_tensor_dict({
        'skewness': fitted_skewness,
        'kurtosis': fitted_kurtosis
    }, convert_to_float=True)
    
    print(f"Final distribution statistics (empirical, after linear transformation):")
    print(f"  Mean: {empirical_stats_dict['mean']:.4f} (Target: {target_moments['mean']:.4f})")
    print(f"  Var:  {empirical_stats_dict['variance']:.4e} (Target: {target_moments['variance']:.4e})")
    print(f"  Skew: {empirical_stats_dict['skewness']:.4f} (Target: {target_moments['skewness']:.4f})")
    print(f"  Kurt: {empirical_stats_dict['kurtosis']:.4f} (Target: {target_moments['kurtosis']:.4f})")
    print(f"Fitted skewness and kurtosis (before linear transformation):")
    print(f"  Fitted Skew: {fitted_stats['skewness']:.4f} (Target: {target_moments['skewness']:.4f})")
    print(f"  Fitted Kurt: {fitted_stats['kurtosis']:.4f} (Target: {target_moments['kurtosis']:.4f})")
    print(f"Linear transformation applied: y = {a:.6f} * x + {b:.4f}")
    
    # Return fitting results
    result = {
        'weights': [float(w) for w in sorted_weights.tolist()],
        'means': [float(m) for m in sorted_means.tolist()],
        'stds': [float(s) for s in sorted_stddevs.tolist()],
        'loss': float(best_loss),
        'empirical_stats': {
            'mean': empirical_stats_dict['mean'],
            'variance': empirical_stats_dict['variance'],
            'skewness': empirical_stats_dict['skewness'],
            'kurtosis': empirical_stats_dict['kurtosis']
        },
        'fitted_stats': {
            'skewness': fitted_stats['skewness'],
            'kurtosis': fitted_stats['kurtosis']
        },
        'target_stats': {
            'mean': float(target_moments['mean']),
            'variance': float(target_moments['variance']),
            'skewness': float(target_moments['skewness']),
            'kurtosis': float(target_moments['kurtosis'])
        }
    }
    
    print(f"Variable {var_name}{suffix} fitting completed, final loss: {best_loss:.6f}")
    print(f"{'='*80}")
    
    return result

def process_all_variables():
    """
    Process fitting for all variables
    
    Returns:
        dict: Dictionary containing fitting results for all variables
    """
    results = {}
    
    print("Starting to process all variables...")
    print(f"Total {len(Flow_info)} variables to process")
    
    for var_name, var_data in Flow_info.items():
        method = var_data['method']
        
        print(f"\nProcessing variable: {var_name}, Method: {method}")
        
        # Initialize result dictionary
        results[var_name] = {
            'method': method
        }
        
        if method == 'Tobit':
            # Tobit method doesn't need GMM fitting
            print(f"Variable {var_name} uses Tobit method, skip GMM fitting")
            scale_value = var_data.get('scale', None)
            results[var_name]['scale'] = float(scale_value) if scale_value is not None else None
            continue
        
        elif method in ['HS', 'LLG', 'LLN']:
            # These methods fit single mean, variance, skewness, kurtosis
            if all(key in var_data for key in ['mean', 'variance', 'skewness', 'kurtosis']):
                fit_result = fit_single_variable(var_name, var_data)
                results[var_name].update(fit_result)
            else:
                print(f"Warning: Variable {var_name} missing necessary moment information")
                
        elif method in ['LSG', 'MUNO']:
            # These methods need to fit both pos and neg distributions
            # Fit positive distribution
            if all(key in var_data for key in ['mean_pos', 'variance_pos', 'skewness_pos', 'kurtosis_pos']):
                fit_result_pos = fit_single_variable(var_name, var_data, suffix="_pos")
                results[var_name]['weights_pos'] = fit_result_pos['weights']
                results[var_name]['means_pos'] = fit_result_pos['means']
                results[var_name]['stds_pos'] = fit_result_pos['stds']
                results[var_name]['loss_pos'] = fit_result_pos['loss']
                results[var_name]['empirical_stats_pos'] = fit_result_pos['empirical_stats']
                results[var_name]['fitted_stats_pos'] = fit_result_pos['fitted_stats']
                results[var_name]['target_stats_pos'] = fit_result_pos['target_stats']
            else:
                print(f"Warning: Variable {var_name} missing pos moment information")
            
            # Fit negative distribution
            if all(key in var_data for key in ['mean_neg', 'variance_neg', 'skewness_neg', 'kurtosis_neg']):
                fit_result_neg = fit_single_variable(var_name, var_data, suffix="_neg")
                results[var_name]['weights_neg'] = fit_result_neg['weights']
                results[var_name]['means_neg'] = fit_result_neg['means']
                results[var_name]['stds_neg'] = fit_result_neg['stds']
                results[var_name]['loss_neg'] = fit_result_neg['loss']
                results[var_name]['empirical_stats_neg'] = fit_result_neg['empirical_stats']
                results[var_name]['fitted_stats_neg'] = fit_result_neg['fitted_stats']
                results[var_name]['target_stats_neg'] = fit_result_neg['target_stats']
            else:
                print(f"Warning: Variable {var_name} missing neg moment information")
                
        else:
            print(f"Unknown method: {method}")
    
    return results

def save_results_to_json(results, parameters_filename='gmm_parameters.json', statistics_filename='gmm_statistics.json'):
    """
    Save results to two separate JSON files: one for parameters and one for statistics
    
    Args:
        results (dict): Fitting results dictionary
        parameters_filename (str): Output filename for parameters
        statistics_filename (str): Output filename for statistics
    """
    print(f"\nSeparating results into parameters and statistics...")
    
    # Separate results into parameters and statistics
    parameters_data = {}
    statistics_data = {}
    
    for var_name, var_data in results.items():
        method = var_data['method']
        
        # Initialize entries
        parameters_data[var_name] = {'method': method}
        statistics_data[var_name] = {'method': method}
        
        if method == 'Tobit':
            # Tobit method only has scale parameter
            if 'scale' in var_data:
                parameters_data[var_name]['scale'] = var_data['scale']
        
        elif method in ['HS', 'LLG', 'LLN']:
            # Single distribution methods
            if 'weights' in var_data:
                parameters_data[var_name]['weights'] = var_data['weights']
            if 'means' in var_data:
                parameters_data[var_name]['means'] = var_data['means']
            if 'stds' in var_data:
                parameters_data[var_name]['stds'] = var_data['stds']
            
            # Statistics
            for stat_key in ['loss', 'empirical_stats', 'fitted_stats', 'target_stats']:
                if stat_key in var_data:
                    statistics_data[var_name][stat_key] = var_data[stat_key]
        
        elif method in ['LSG', 'MUNO']:
            # Dual distribution methods (pos and neg)
            # Parameters for positive distribution
            for param_key in ['weights_pos', 'means_pos', 'stds_pos']:
                if param_key in var_data:
                    parameters_data[var_name][param_key] = var_data[param_key]
            
            # Parameters for negative distribution
            for param_key in ['weights_neg', 'means_neg', 'stds_neg']:
                if param_key in var_data:
                    parameters_data[var_name][param_key] = var_data[param_key]
            
            # Statistics for both distributions
            for stat_key in ['loss_pos', 'empirical_stats_pos', 'fitted_stats_pos', 'target_stats_pos',
                           'loss_neg', 'empirical_stats_neg', 'fitted_stats_neg', 'target_stats_neg']:
                if stat_key in var_data:
                    statistics_data[var_name][stat_key] = var_data[stat_key]
    
    # Save parameters file
    print(f"Saving parameters to {parameters_filename}...")
    with open(parameters_filename, 'w', encoding='utf-8') as f:
        json.dump(parameters_data, f, indent=2, ensure_ascii=False)
    
    # Save statistics file
    print(f"Saving statistics to {statistics_filename}...")
    with open(statistics_filename, 'w', encoding='utf-8') as f:
        json.dump(statistics_data, f, indent=2, ensure_ascii=False)
    
    print(f"Parameters saved to {parameters_filename}")
    print(f"Statistics saved to {statistics_filename}")
    print(f"Total {len(results)} variables processed")
    
    # Print statistics
    method_counts = {}
    for var_name, var_data in results.items():
        method = var_data['method']
        method_counts[method] = method_counts.get(method, 0) + 1
    
    print("\nVariable counts by method:")
    for method, count in method_counts.items():
        print(f"  {method}: {count}")

def run_batch_fitting():
    """
    Run batch fitting task
    """
    print("Starting batch fitting task...")
    print("="*100)
    
    # Process all variables
    results = process_all_variables()
    
    # Save results
    save_results_to_json(results)
    
    print("\nBatch fitting task completed!")
    print("="*100)


if __name__ == '__main__':

    run_batch_fitting()