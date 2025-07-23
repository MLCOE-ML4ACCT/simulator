import tensorflow as tf
import tensorflow_probability as tfp

def step1_initialize(mean, std, skew, kurt):
    """
    Step 1: Initialize parameters
    Input: mean, std, skew, kurt - each is a tensor of shape [batch_size]
    Output: beta1, beta2 - corresponding to Skew² and Kurt
    """
    beta1 = tf.square(skew)  # β₁ = Skew², shape: [batch_size]
    beta2 = kurt             # β₂ = Kurt, shape: [batch_size]
    
    return beta1, beta2

def step2_find_omega1(beta2):
    """
    Step 2: Find lower bound ω₁
    Solve for the positive root of ω⁴+2ω³+3ω²-3=β₂
    
    Input: beta2 - tensor of shape [batch_size]
    Output: omega1 - lower bound
    """
    # According to the formula
    # D = (3+β₂)(16β₂²+87β₂+171)/27
    D = (3.0 + beta2) * (16.0 * tf.square(beta2) + 87.0 * beta2 + 171.0) / 27.0  # shape: [batch_size]
    
    # d = -1 + ∛(7+2β₂+2√D) - ∛(2√D-7-2β₂)
    sqrt_D = tf.sqrt(D)  # shape: [batch_size]
    term1 = 7.0 + 2.0 * beta2 + 2.0 * sqrt_D  # shape: [batch_size]
    term2 = 2.0 * sqrt_D - 7.0 - 2.0 * beta2  # shape: [batch_size]
    
    # Cube root calculation, handle negative numbers
    cbrt_term1 = tf.sign(term1) * tf.pow(tf.abs(term1), 1.0/3.0)  # shape: [batch_size]
    cbrt_term2 = tf.sign(term2) * tf.pow(tf.abs(term2), 1.0/3.0)  # shape: [batch_size]
    d = -1.0 + cbrt_term1 - cbrt_term2  # shape: [batch_size]
    
    # ω₁ = (1/2) × (-1 + √d + √(4/√d - d - 3))
    sqrt_d = tf.sqrt(d)  # shape: [batch_size]
    inner_sqrt = 4.0 / sqrt_d - d - 3.0  # shape: [batch_size]
    omega1 = 0.5 * (-1.0 + sqrt_d + tf.sqrt(inner_sqrt))  # shape: [batch_size]
    
    return omega1

def step3_define_functions(omega, beta2):
    """
    Step 3: Define functions m(ω) and f(ω)
    
    Input: 
        omega - tensor of shape [batch_size]
        beta2 - tensor of shape [batch_size]
    Output: 
        m_omega - value of m(ω), shape: [batch_size]
        f_omega - value of f(ω), shape: [batch_size]
    """
    # m(ω) = -2 + √(4 + 2[ω² - (β₂+3)/(ω²+2ω+3)])
    omega_sq = tf.square(omega)  # shape: [batch_size]
    denominator = omega_sq + 2.0 * omega + 3.0  # shape: [batch_size]
    fraction = (beta2 + 3.0) / denominator  # shape: [batch_size]
    inner_term = 4.0 + 2.0 * (omega_sq - fraction)  # shape: [batch_size]
    m_omega = -2.0 + tf.sqrt(inner_term)  # shape: [batch_size]
    
    # f(ω) = (ω-1-m(ω))(ω+2+m(ω)/2)²
    term1 = omega - 1.0 - m_omega  # shape: [batch_size]
    term2 = omega + 2.0 + 0.5 * m_omega  # shape: [batch_size]
    f_omega = term1 * tf.square(term2)  # shape: [batch_size]
    
    return m_omega, f_omega

def step4_check_solution(omega1, beta1):
    """
    Step 4: Check existence of solution
    If (ω₁-1)(ω₁+2)² ≤ β₁, then no solution
    
    Input:
        omega1 - tensor of shape [batch_size]
        beta1 - tensor of shape [batch_size]
    Output:
        has_solution - bool tensor, True means solution exists, shape: [batch_size]
    """
    # Compute (ω₁-1)(ω₁+2)²
    omega1_minus_1 = omega1 - 1.0  # shape: [batch_size]
    omega1_plus_2 = omega1 + 2.0  # shape: [batch_size]
    left_side = omega1_minus_1 * tf.square(omega1_plus_2)  # shape: [batch_size]
    
    # Check if left_side > β₁
    has_solution = left_side > beta1  # shape: [batch_size]
    
    return has_solution

def step5_calculate_omega2(beta2):
    """
    Step 5: Calculate upper bound ω₂
    ω₂ = √(-1 + √(2(β₂-1)))
    
    Input: beta2 - tensor of shape [batch_size]
    Output: omega2 - upper bound, shape: [batch_size]
    """
    inner_sqrt = tf.sqrt(2.0 * (beta2 - 1.0))  # shape: [batch_size]
    omega2 = tf.sqrt(-1.0 + inner_sqrt)  # shape: [batch_size]
    
    return omega2

def create_objective_function(beta1, beta2):
    """
    Create objective function for root finding
    f(ω) = (ω-1-m(ω))(ω+2+m(ω)/2)² - β₁
    
    Input:
        beta1 - tensor of shape [batch_size]
        beta2 - tensor of shape [batch_size]
    Output:
        objective_fn - callable function
    """
    def objective_fn(omega):
        """
        Objective function: f(ω) - β₁ = 0
        
        Input:
            omega - tensor of shape [batch_size]
        Output:
            result - tensor of shape [batch_size]
        """
        # m(ω) = -2 + √(4 + 2[ω² - (β₂+3)/(ω²+2ω+3)])
        omega_sq = tf.square(omega)  # shape: [batch_size]
        denominator = omega_sq + 2.0 * omega + 3.0  # shape: [batch_size]
        fraction = (beta2 + 3.0) / denominator  # shape: [batch_size]
        inner_term = 4.0 + 2.0 * (omega_sq - fraction)  # shape: [batch_size]
        m_omega = -2.0 + tf.sqrt(inner_term)  # shape: [batch_size]
        
        # f(ω) = (ω-1-m(ω))(ω+2+m(ω)/2)²
        term1 = omega - 1.0 - m_omega  # shape: [batch_size]
        term2 = omega + 2.0 + 0.5 * m_omega  # shape: [batch_size]
        f_omega = term1 * tf.square(term2)  # shape: [batch_size]
        
        # Return f(ω) - β₁
        return f_omega - beta1
    
    return objective_fn

def step6_find_omega_star(omega1, omega2, beta1, beta2):
    """
    Step 6: Use root finding algorithm to find ω*
    Find ω* in the interval (ω₁, ω₂] such that f(ω*) = β₁
    
    Input:
        omega1 - lower bound, shape: [batch_size]
        omega2 - upper bound, shape: [batch_size]
        beta1 - tensor of shape [batch_size]
        beta2 - tensor of shape [batch_size]
    Output:
        omega_star - found root, shape: [batch_size]
    """
    # Ensure data type is float32
    omega1 = tf.cast(omega1, tf.float32)
    omega2 = tf.cast(omega2, tf.float32)
    beta1 = tf.cast(beta1, tf.float32)
    beta2 = tf.cast(beta2, tf.float32)
    
    # Create objective function
    objective_fn = create_objective_function(beta1, beta2)
    
    # Use TensorFlow Probability's root finding algorithm
    root_results = tfp.math.find_root_chandrupatla(
        objective_fn=objective_fn,
        low=omega1,
        high=omega2,
        position_tolerance=1e-8,
        value_tolerance=1e-8,
        max_iterations=50,
        stopping_policy_fn=tf.reduce_all,
        validate_args=False,
        name='find_omega_star'
    )
    
    # Extract found root
    omega_star = root_results.estimated_root
    
    return omega_star

def step7_recover_parameters(omega_star, mean, std, skew, kurt, beta2):
    """
    Step 7: Recover JohnsonSU distribution parameters from ω*
    
    Input:
        omega_star - found root, shape: [batch_size]
        mean - mean, shape: [batch_size]
        std - stddev, shape: [batch_size]
        skew - skewness, shape: [batch_size]
        kurt - kurtosis, shape: [batch_size]
        beta2 - tensor of shape [batch_size]
    Output:
        delta - shape: [batch_size]
        lambda_param - shape: [batch_size]
        gamma - shape: [batch_size]
        xi - shape: [batch_size]
    """
    # Ensure data type is float32
    omega_star = tf.cast(omega_star, tf.float32)
    mean = tf.cast(mean, tf.float32)
    std = tf.cast(std, tf.float32)
    skew = tf.cast(skew, tf.float32)
    kurt = tf.cast(kurt, tf.float32)
    beta2 = tf.cast(beta2, tf.float32)
    
    # Directly call step3_define_functions to get m(omega*)
    m, _ = step3_define_functions(omega_star, beta2)  # shape: [batch_size]
    
    # Calculate Ω
    # Ω = -sgn(Skew) * sinh⁻¹(√((ω*-1)/(2ω*) * ((ω*-1)/m - 1)))
    omega_minus_1 = omega_star - 1.0  # shape: [batch_size]
    term1 = omega_minus_1 / (2.0 * omega_star)  # shape: [batch_size]
    term2 = omega_minus_1 / m - 1.0  # shape: [batch_size]
    inner_sqrt = term1 * term2  # shape: [batch_size]
    sinh_inv_arg = tf.sqrt(inner_sqrt)  # shape: [batch_size]
    omega_calc = tf.asinh(sinh_inv_arg)  # shape: [batch_size]
    
    # sgn function
    sgn_skew = tf.sign(skew)  # shape: [batch_size]
    omega_final = -sgn_skew * omega_calc  # shape: [batch_size]
    
    # Calculate δ
    # δ = 1/√(ln(ω*))
    delta = 1.0 / tf.sqrt(tf.math.log(omega_star))  # shape: [batch_size]
    
    # Calculate γ
    # γ = Ω/√(ln(ω*))
    gamma = omega_final / tf.sqrt(tf.math.log(omega_star))  # shape: [batch_size]
    
    # Calculate λ
    # λ = σ(x)/(ω*-1) * √(2m/(ω*+1))
    lambda_numerator = std / omega_minus_1  # shape: [batch_size]
    lambda_denominator = 2.0 * m / (omega_star + 1.0)  # shape: [batch_size]
    lambda_param = lambda_numerator * tf.sqrt(lambda_denominator)  # shape: [batch_size]
    
    # Calculate ξ
    # ξ = μ(x) - sgn(Skew) * σ(x)/(ω*-1) * √(ω*-1-m)
    xi_term1 = mean  # shape: [batch_size]
    xi_term2 = sgn_skew * std / omega_minus_1  # shape: [batch_size]
    xi_term3 = tf.sqrt(omega_minus_1 - m)  # shape: [batch_size]
    xi = xi_term1 - xi_term2 * xi_term3  # shape: [batch_size]
    
    return delta, lambda_param, gamma, xi

def johnson_su_fit_steps1to7(mean, std, skew, kurt):
    """
    Main function: perform steps 1-7 of JohnsonSU fitting
    
    Input:
        mean - mean, shape: [batch_size]
        std - stddev, shape: [batch_size]  
        skew - skewness, shape: [batch_size]
        kurt - kurtosis, shape: [batch_size]
    
    Output:
        results - dict, contains all intermediate results and final parameters
    """
    print("Executing Step 1: Initialize parameters")
    beta1, beta2 = step1_initialize(mean, std, skew, kurt)  # shape: [batch_size]
    
    print("Executing Step 2: Calculate lower bound ω₁")
    omega1 = step2_find_omega1(beta2)  # shape: [batch_size]
    
    print("Executing Step 3: Define functions m(ω) and f(ω)")
    # Here, test function definition with omega1
    m_omega1, f_omega1 = step3_define_functions(omega1, beta2)  # shape: [batch_size]
    
    print("Executing Step 4: Check existence of solution")
    has_solution = step4_check_solution(omega1, beta1)  # shape: [batch_size]
    
    print("Executing Step 5: Calculate upper bound ω₂")
    omega2 = step5_calculate_omega2(beta2)  # shape: [batch_size]
    
    print("Executing Step 6: Root finding to get ω*")
    omega_star = step6_find_omega_star(omega1, omega2, beta1, beta2)  # shape: [batch_size]
    
    print("Executing Step 7: Recover JohnsonSU parameters")
    delta, lambda_param, gamma, xi = step7_recover_parameters(omega_star, mean, std, skew, kurt, beta2)  # shape: [batch_size]
    
    results = {
        'beta1': beta1,  # shape: [batch_size]
        'beta2': beta2,  # shape: [batch_size]
        'omega1': omega1,  # shape: [batch_size]
        'omega2': omega2,  # shape: [batch_size]
        'has_solution': has_solution,  # shape: [batch_size]
        'm_omega1': m_omega1,  # shape: [batch_size]
        'f_omega1': f_omega1,  # shape: [batch_size]
        'omega_star': omega_star,  # shape: [batch_size]
        'delta': delta,  # shape: [batch_size]
        'lambda': lambda_param,  # shape: [batch_size]
        'gamma': gamma,  # shape: [batch_size]
        'xi': xi  # shape: [batch_size]
    }
    
    return results

def verify_johnson_su_fit(gamma, delta, xi, lambda_param, n_samples=10000, seed=42):
    """
    Verify JohnsonSU fitting results using tfp's JohnsonSU distribution
    
    Input:
        gamma - shape parameter (corresponds to tfp's skewness), shape: [batch_size]
        delta - shape parameter (corresponds to tfp's tailweight), shape: [batch_size]  
        xi - location parameter (corresponds to tfp's loc), shape: [batch_size]
        lambda_param - scale parameter (corresponds to tfp's scale), shape: [batch_size]
        n_samples - number of samples per distribution, default 10000
        seed - random seed
    Output:
        verification_results - contains original parameters, empirical statistics, and comparison results
    """
    print(f"Start verifying JohnsonSU fitting results, {n_samples} samples per distribution")
    
    # Ensure input is float32
    gamma = tf.cast(gamma, tf.float32)
    delta = tf.cast(delta, tf.float32)
    xi = tf.cast(xi, tf.float32)
    lambda_param = tf.cast(lambda_param, tf.float32)
    
    # Create JohnsonSU distribution
    # Parameter mapping: gamma->skewness, delta->tailweight, xi->loc, lambda->scale
    johnson_su_dist = tfp.distributions.JohnsonSU(
        skewness=gamma,      # shape: [batch_size]
        tailweight=delta,    # shape: [batch_size]
        loc=xi,              # shape: [batch_size]
        scale=lambda_param,  # shape: [batch_size]
        validate_args=False,
        allow_nan_stats=True
    )
    
    print(f"JohnsonSU distribution created, batch_size: {gamma.shape[0]}")
    
    # Generate samples
    tf.random.set_seed(seed)
    samples = johnson_su_dist.sample(n_samples)  # shape: [n_samples, batch_size]
    
    print(f"Samples generated, samples shape: {samples.shape}")
    
    # Compute empirical statistics
    emp_mean, emp_var, emp_skew, emp_kurt = calculate_empirical_moments(samples)
    
    print("Empirical statistics computed")
    
    # Compute theoretical statistics (if available)
    try:
        theo_mean = johnson_su_dist.mean()
        theo_var = johnson_su_dist.variance()
        print("Theoretical mean and variance computed successfully")
    except:
        theo_mean = tf.fill(gamma.shape, float('nan'))
        theo_var = tf.fill(gamma.shape, float('nan'))
        print("Warning: Unable to compute theoretical mean and variance")
    
    # Organize results
    verification_results = {
        # Input parameters
        'fitted_gamma': gamma,          # shape: [batch_size]
        'fitted_delta': delta,          # shape: [batch_size]
        'fitted_xi': xi,                # shape: [batch_size]
        'fitted_lambda': lambda_param,  # shape: [batch_size]
        
        # Sample info
        'n_samples': n_samples,
        'samples_shape': samples.shape,
        
        # Empirical statistics
        'empirical_mean': emp_mean,     # shape: [batch_size]
        'empirical_var': emp_var,       # shape: [batch_size]
        'empirical_skew': emp_skew,     # shape: [batch_size]
        'empirical_kurt': emp_kurt,     # shape: [batch_size]
        
        # Theoretical statistics (if available)
        'theoretical_mean': theo_mean,  # shape: [batch_size]
        'theoretical_var': theo_var,    # shape: [batch_size]
        
        # Original samples (optional, large memory usage)
        'samples': samples              # shape: [n_samples, batch_size]
    }
    
    return verification_results

def calculate_empirical_moments(samples):
    """
    Compute empirical moments: mean, variance, skewness, kurtosis
    
    Input:
        samples - tensor of shape [n_samples, batch_size]
    Output:
        emp_mean - empirical mean, shape: [batch_size]
        emp_var - empirical variance, shape: [batch_size]
        emp_skew - empirical skewness, shape: [batch_size]
        emp_kurt - empirical kurtosis, shape: [batch_size]
    """
    # Compute empirical mean
    emp_mean = tf.reduce_mean(samples, axis=0)  # shape: [batch_size]
    
    # Compute empirical variance
    emp_var = tf.reduce_mean(tf.square(samples - emp_mean), axis=0)  # shape: [batch_size]
    
    # Compute stddev
    emp_std = tf.sqrt(emp_var)  # shape: [batch_size]
    
    # Standardize samples
    standardized = (samples - emp_mean) / emp_std  # shape: [n_samples, batch_size]
    
    # Compute empirical skewness (third central moment divided by stddev cubed)
    emp_skew = tf.reduce_mean(tf.pow(standardized, 3), axis=0)  # shape: [batch_size]
    
    # Compute empirical kurtosis (fourth central moment divided by variance squared)
    emp_kurt = tf.reduce_mean(tf.pow(standardized, 4), axis=0)  # shape: [batch_size]
    
    return emp_mean, emp_var, emp_skew, emp_kurt

def extract_moments_from_config():
    """
    Extract all variables' four moments from config.py, generate batch tensor and corresponding label info
    
    Returns:
        tuple: (means_tensor, variances_tensor, skews_tensor, kurts_tensor, labels_info)
            - means_tensor: tensor of shape [batch_size]
            - variances_tensor: tensor of shape [batch_size]  
            - skews_tensor: tensor of shape [batch_size]
            - kurts_tensor: tensor of shape [batch_size]
            - labels_info: list, each element is a dict containing variable info for the corresponding tensor position
    """
    # Import config data
    from config import Flow_info
    
    # Store extracted data
    means_list = []
    variances_list = []
    skews_list = []
    kurts_list = []
    labels_info = []
    
    print("Start extracting four moments information from config.py...")
    print(f"Total {len(Flow_info)} variables to process")
    
    for var_name, var_data in Flow_info.items():
        method = var_data.get('method', 'Unknown')
        
        print(f"Processing variable: {var_name}, method: {method}")
        
        if method == 'Tobit':
            # Tobit method has no mean, variance, skewness, kurtosis, skip
            print(f"  Skip {var_name} (Tobit method has no four moments)")
            continue
            
        elif method in ['HS', 'LLG', 'LLN']:
            # These methods have a single set of four moments
            if all(key in var_data for key in ['mean', 'variance', 'skewness', 'kurtosis']):
                means_list.append(var_data['mean'])
                variances_list.append(var_data['variance'])
                skews_list.append(var_data['skewness'])
                kurts_list.append(var_data['kurtosis'])
                
                # Record label info
                labels_info.append({
                    'variable_name': var_name,
                    'method': method,
                    'type': 'single',  # single distribution
                    'subtype': None
                })
                
                print(f"  Extracted {var_name}: mean={var_data['mean']:.6f}, var={var_data['variance']:.6e}, skew={var_data['skewness']:.6f}, kurt={var_data['kurtosis']:.6f}")
            else:
                print(f"  Warning: {var_name} missing required four moments information")
                
        elif method in ['LSG', 'MUNO']:
            # These methods have pos and neg distributions
            
            # Process positive distribution
            if all(key in var_data for key in ['mean_pos', 'variance_pos', 'skewness_pos', 'kurtosis_pos']):
                means_list.append(var_data['mean_pos'])
                variances_list.append(var_data['variance_pos'])
                skews_list.append(var_data['skewness_pos'])
                kurts_list.append(var_data['kurtosis_pos'])
                
                labels_info.append({
                    'variable_name': var_name,
                    'method': method,
                    'type': 'dual',  # dual distribution
                    'subtype': 'pos'
                })
                
                print(f"  Extracted {var_name}_pos: mean={var_data['mean_pos']:.6f}, var={var_data['variance_pos']:.6e}, skew={var_data['skewness_pos']:.6f}, kurt={var_data['kurtosis_pos']:.6f}")
            else:
                print(f"  Warning: {var_name} missing positive distribution's four moments information")
            
            # Process negative distribution
            if all(key in var_data for key in ['mean_neg', 'variance_neg', 'skewness_neg', 'kurtosis_neg']):
                means_list.append(var_data['mean_neg'])
                variances_list.append(var_data['variance_neg'])
                skews_list.append(var_data['skewness_neg'])
                kurts_list.append(var_data['kurtosis_neg'])
                
                labels_info.append({
                    'variable_name': var_name,
                    'method': method,
                    'type': 'dual',  # dual distribution
                    'subtype': 'neg'
                })
                
                print(f"  Extracted {var_name}_neg: mean={var_data['mean_neg']:.6f}, var={var_data['variance_neg']:.6e}, skew={var_data['skewness_neg']:.6f}, kurt={var_data['kurtosis_neg']:.6f}")
            else:
                print(f"  Warning: {var_name} missing negative distribution's four moments information")
                
        else:
            print(f"  Unknown method: {method}, skip variable {var_name}")
    
    # Convert to tensorflow tensor
    if len(means_list) == 0:
        print("Warning: No valid four moments data extracted!")
        return None, None, None, None, []
    
    means_tensor = tf.constant(means_list, dtype=tf.float32)
    variances_tensor = tf.constant(variances_list, dtype=tf.float32)
    skews_tensor = tf.constant(skews_list, dtype=tf.float32)
    kurts_tensor = tf.constant(kurts_list, dtype=tf.float32)
    
    print(f"\nSuccessfully extracted data:")
    print(f"  Total batch size: {len(means_list)}")
    print(f"  means_tensor shape: {means_tensor.shape}")
    print(f"  variances_tensor shape: {variances_tensor.shape}")
    print(f"  skews_tensor shape: {skews_tensor.shape}")
    print(f"  kurts_tensor shape: {kurts_tensor.shape}")
    print(f"  labels_info length: {len(labels_info)}")
    
    return means_tensor, variances_tensor, skews_tensor, kurts_tensor, labels_info 