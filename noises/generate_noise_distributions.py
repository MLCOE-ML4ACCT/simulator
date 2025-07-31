import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import json
from config import Flow_info

# TFP shortcuts
tfd = tfp.distributions
tfb = tfp.bijectors

def load_noise_parameters(filename='noises/combined_noise_parameters.json'):
    """
    Load noise parameters from JSON file
    
    Args:
        filename (str): Path to JSON file
        
    Returns:
        dict: Dictionary containing parameters for all variables
    """
    with open(filename, 'r', encoding='utf-8') as f:
        parameters = json.load(f)
    return parameters

def create_johnson_su_distribution(gamma, delta, xi, lambda_param):
    """
    Create Johnson SU distribution
    
    Args:
        gamma (float): Skewness parameter
        delta (float): Tailweight parameter
        xi (float): Location parameter
        lambda_param (float): Scale parameter
        
    Returns:
        tfp.distributions.JohnsonSU: Johnson SU distribution object
    """
    return tfd.JohnsonSU(
        skewness=gamma,
        tailweight=delta,
        loc=xi,
        scale=lambda_param,
        validate_args=False,
        allow_nan_stats=True
    )

def create_gaussian_mixture_distribution(weights, means, stds, target_mean=None, target_variance=None):
    """
    Create Gaussian mixture distribution, and apply linear transformation to match target mean and variance if needed
    
    Args:
        weights (list): List of weights
        means (list): List of means  
        stds (list): List of standard deviations
        target_mean (float, optional): Target mean, if provided, apply linear transformation
        target_variance (float, optional): Target variance, if provided, apply linear transformation
        
    Returns:
        tfp.distributions.Distribution: GMM distribution object (possibly after linear transformation)
    """
    # Convert to tensorflow tensor
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)
    means = tf.convert_to_tensor(means, dtype=tf.float32)
    stds = tf.convert_to_tensor(stds, dtype=tf.float32)
    
    # Create original Gaussian mixture distribution
    gmm = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=weights),
        components_distribution=tfd.Normal(loc=means, scale=stds)
    )
    
    # If target mean and variance are provided, apply linear transformation y = ax + b
    if target_mean is not None and target_variance is not None:
        # Calculate current mean and variance
        current_mean = gmm.mean()
        current_variance = gmm.variance()
        current_std = tf.sqrt(current_variance)
        
        # Calculate linear transformation parameters y = ax + b
        # a = σ_target / σ_current,  b = μ_target - a * μ_current
        target_std = tf.sqrt(tf.convert_to_tensor(target_variance, dtype=tf.float32))
        target_mean_tensor = tf.convert_to_tensor(target_mean, dtype=tf.float32)
        
        a = target_std / current_std
        b = target_mean_tensor - a * current_mean
        
        # Create bijector for linear transformation: y = ax + b
        # Note: Chain executes bijectors in REVERSE order, so we put Shift first, then Scale
        linear_transform = tfb.Chain([
            tfb.Shift(shift=b),        # Second: shift by b
            tfb.Scale(scale=a)         # First: scale by a
        ])  # Result: y = (ax) + b = ax + b
        
        # Apply linear transformation to get the final distribution
        transformed_gmm = tfd.TransformedDistribution(
            distribution=gmm,
            bijector=linear_transform
        )
        
        return transformed_gmm
    
    return gmm

def create_distribution_for_variable(var_name, var_params):
    """
    Create distribution for a single variable. For GMM, apply linear transformation to match target statistics
    
    Args:
        var_name (str): Variable name
        var_params (dict): Variable parameter dictionary
        
    Returns:
        dict or tfp.distributions.Distribution or None: 
        - For LSG/MUNO returns dict with pos/neg distributions
        - For HS/LLG/LLN returns a single distribution object
        - For Tobit returns None
    """
    method = var_params['method']
    
    # Tobit method does not require noise generation
    if method == 'Tobit':
        return None
    
    # Handle single distribution methods: HS, LLG, LLN
    elif method in ['HS', 'LLG', 'LLN']:
        if var_params['johnsonSU_has_result']:
            # Use Johnson SU distribution
            return create_johnson_su_distribution(
                gamma=var_params['gamma'],
                delta=var_params['delta'],
                xi=var_params['xi'],
                lambda_param=var_params['lambda']
            )
        else:
            # Use Gaussian mixture distribution, apply linear transformation to match target statistics
            target_mean = None
            target_variance = None
            
            # Get target statistics from Flow_info
            if var_name in Flow_info:
                target_mean = Flow_info[var_name].get('mean', None)
                target_variance = Flow_info[var_name].get('variance', None)
            
            return create_gaussian_mixture_distribution(
                weights=var_params['weights'],
                means=var_params['means'],
                stds=var_params['stds'],
                target_mean=target_mean,
                target_variance=target_variance
            )
    
    # Handle dual distribution methods: LSG, MUNO  
    elif method in ['LSG', 'MUNO']:
        distributions = {}
        
        # Handle positive distribution
        if 'pos' in var_params:
            pos_params = var_params['pos']
            if pos_params['johnsonSU_has_result']:
                distributions['pos'] = create_johnson_su_distribution(
                    gamma=pos_params['gamma'],
                    delta=pos_params['delta'],
                    xi=pos_params['xi'],
                    lambda_param=pos_params['lambda']
                )
            else:
                # Use Gaussian mixture distribution, apply linear transformation to match target statistics
                target_mean_pos = None
                target_variance_pos = None
                
                # Get target statistics for positive distribution from Flow_info
                if var_name in Flow_info:
                    target_mean_pos = Flow_info[var_name].get('mean_pos', None)
                    target_variance_pos = Flow_info[var_name].get('variance_pos', None)
                
                distributions['pos'] = create_gaussian_mixture_distribution(
                    weights=pos_params['weights'],
                    means=pos_params['means'],
                    stds=pos_params['stds'],
                    target_mean=target_mean_pos,
                    target_variance=target_variance_pos
                )
        
        # Handle negative distribution
        if 'neg' in var_params:
            neg_params = var_params['neg']
            if neg_params['johnsonSU_has_result']:
                distributions['neg'] = create_johnson_su_distribution(
                    gamma=neg_params['gamma'],
                    delta=neg_params['delta'],
                    xi=neg_params['xi'],
                    lambda_param=neg_params['lambda']
                )
            else:
                # Use Gaussian mixture distribution, apply linear transformation to match target statistics
                target_mean_neg = None
                target_variance_neg = None
                
                # Get target statistics for negative distribution from Flow_info
                if var_name in Flow_info:
                    target_mean_neg = Flow_info[var_name].get('mean_neg', None)
                    target_variance_neg = Flow_info[var_name].get('variance_neg', None)
                
                distributions['neg'] = create_gaussian_mixture_distribution(
                    weights=neg_params['weights'],
                    means=neg_params['means'],
                    stds=neg_params['stds'],
                    target_mean=target_mean_neg,
                    target_variance=target_variance_neg
                )
        
        return distributions
    
    else:
        raise ValueError(f"Unknown method: {method}")

def generate_all_distributions(parameters_file='noises/combined_noise_parameters.json'):
    """
    Generate distributions for all variables
    
    Args:
        parameters_file (str): Path to parameter file
        
    Returns:
        dict: Dictionary containing distributions for all variables
        Dictionary structure:
        - key: variable name
        - value: 
          - For HS/LLG/LLN: TFP distribution object
          - For LSG/MUNO: dict {'pos': distribution object, 'neg': distribution object}
          - For Tobit: None
    """
    # Load parameters
    parameters = load_noise_parameters(parameters_file)
    
    # Create distribution dictionary
    distributions = {}
    
    # Generate distribution for each variable
    for var_name, var_params in parameters.items():
        distributions[var_name] = create_distribution_for_variable(var_name, var_params)
    
    return distributions

def sample_from_distributions(distributions, num_samples=1000):
    """
    Sample from distribution dictionary, return results as tensors with shape [batch_size, 1]
    
    Args:
        distributions (dict): Distribution dictionary generated by generate_all_distributions
        num_samples (int): Number of samples
        
    Returns:
        dict: Dictionary containing samples for each variable
        Dictionary structure:
        - key: variable name
        - value: 
          - For HS/LLG/LLN: tensorflow tensor, shape [num_samples, 1]
          - For LSG/MUNO: dict {'pos': tensor [num_samples, 1], 'neg': tensor [num_samples, 1]}
          - For Tobit: not included in results
    """
    samples = {}
    
    for var_name, distribution in distributions.items():
        if distribution is None:
            # Tobit method does not sample, skip
            continue
            
        if isinstance(distribution, dict):
            # LSG/MUNO methods: contain pos and neg distributions
            samples[var_name] = {}
            for dist_type, dist in distribution.items():
                # Sample and reshape to [num_samples, 1]
                sample_tensor = dist.sample(num_samples)
                samples[var_name][dist_type] = tf.expand_dims(sample_tensor, axis=-1)
        else:
            # HS/LLG/LLN methods: single distribution
            # Sample and reshape to [num_samples, 1]
            sample_tensor = distribution.sample(num_samples)
            samples[var_name] = tf.expand_dims(sample_tensor, axis=-1)
    
    return samples




if __name__ == '__main__':
    # Example usage
    print("Generating noise distributions...")
    
    # Generate all distributions
    distributions = generate_all_distributions()
    
    print(f"Successfully generated distributions for {len(distributions)} variables")
    
    # Show distribution type statistics
    distribution_types = {}
    for var_name, dist in distributions.items():
        if dist is None:
            dist_type = "None (Tobit)"
        elif isinstance(dist, dict):
            dist_type = f"Dict ({', '.join([f'{k}: {type(v).__name__}' for k, v in dist.items()])})"
        else:
            dist_type = type(dist).__name__
        distribution_types[var_name] = dist_type
    
    print("\nDistribution type statistics:")
    johnson_su_count = sum(1 for dt in distribution_types.values() if 'JohnsonSU' in dt)
    transformed_count = sum(1 for dt in distribution_types.values() if 'TransformedDistribution' in dt)
    mixture_count = sum(1 for dt in distribution_types.values() if 'MixtureSameFamily' in dt)
    tobit_count = sum(1 for dt in distribution_types.values() if 'None' in dt)
    
    print(f"  Johnson SU distributions: {johnson_su_count}")
    print(f"  Linear transformed GMM distributions: {transformed_count}")
    print(f"  Original GMM distributions: {mixture_count}")
    print(f"  Tobit method: {tobit_count}")
    
    # Sampling example
    print("\nSampling example (1000 samples)...")
    samples = sample_from_distributions(distributions, num_samples=1000)
    
    print(f"Sampling completed, contains samples for {len(samples)} variables")
    
    # Show shape information of samples
    print("\nSample shape information:")
    for var_name, sample in samples.items():
        if isinstance(sample, dict):
            for dist_type, tensor in sample.items():
                print(f"  {var_name}_{dist_type}: {tensor.shape}")
        else:
            print(f"  {var_name}: {sample.shape}")
    
    print("\nDistribution dictionary is ready for noise generation")
    print("Note: GMM distributions have been automatically linearly transformed to match target mean and variance")
    print("All sample shapes are [batch_size, 1]") 
