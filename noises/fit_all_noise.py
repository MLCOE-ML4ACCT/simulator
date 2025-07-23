import json
import tensorflow as tf
from fitting_johnsonSU import extract_moments_from_config, johnson_su_fit_steps1to7
from fitting_gmm import fit_single_variable
from config import Flow_info

def fit_remaining_variables_with_gmm():
    """
    Use GMM fitting for variables where JohnsonSU fitting fails
    """
    print("=" * 80)
    print("Start GMM fitting for remaining variables")
    print("=" * 80)
    
    # 1. Get all variables and JohnsonSU fitting results
    means, variances, skews, kurts, labels_info = extract_moments_from_config()
    stds = tf.sqrt(variances)
    fit_results = johnson_su_fit_steps1to7(means, stds, skews, kurts)
    has_solution = fit_results['has_solution'].numpy()  # Need numpy for Python control flow
    
    print(f"Total {len(labels_info)} distributions")
    print(f"JohnsonSU successfully fitted: {sum(has_solution)}")
    print(f"Need GMM fitting: {sum(~has_solution)}")
    print()
    
    # 2. Organize variables that need GMM fitting
    gmm_variables = {}  # Store variables that need GMM fitting
    
    for i, label in enumerate(labels_info):
        if has_solution[i]:
            continue  # Skip those already fitted by JohnsonSU
        
        var_name = label['variable_name']
        method = label['method']
        direction = label['subtype'] if label['type'] == 'dual' else None
        
        # Build variable data dict, keep as tensor
        var_data = {
            'mean': means[i],
            'variance': variances[i], 
            'skewness': skews[i],
            'kurtosis': kurts[i]
        }
        
        if method in ["MUNO", "LSG"]:
            # Dual distribution variables
            if var_name not in gmm_variables:
                gmm_variables[var_name] = {
                    'method': method,
                    'type': 'dual',
                    'directions': {}
                }
            gmm_variables[var_name]['directions'][direction] = var_data
        else:
            # Single distribution variable
            gmm_variables[var_name] = {
                'method': method,
                'type': 'single',
                'data': var_data
            }
    
    # 2.1 Add variables with Tobit method
    for var_name, var_data in Flow_info.items():
        method = var_data.get('method', '')
        if method == 'Tobit':
            # Tobit variables get scale parameter directly from config
            scale_value = var_data.get('scale', None)
            if scale_value is not None:
                gmm_variables[var_name] = {
                    'method': method,
                    'type': 'tobit',
                    'scale': scale_value
                }
                print(f"Add Tobit variable: {var_name}, scale: {scale_value}")
    
    print(f"Variables to be fitted by GMM: {list(gmm_variables.keys())}")
    print()
    
    # 3. Fit each variable with GMM
    gmm_parameters = {}
    gmm_statistics = {}
    
    for var_name, var_info in gmm_variables.items():
        print(f"Fitting variable: {var_name} (method: {var_info['method']})")
        
        method = var_info['method']
        
        if method == 'Tobit':
            # Tobit method, set parameters directly
            gmm_parameters[var_name] = {
                "method": method,
                "johnsonSU_has_result": False,
                "scale": var_info['scale']
            }
            
            gmm_statistics[var_name] = {
                "method": method
            }
            
            print(f"  {var_name} Tobit method, scale: {var_info['scale']}")
            
        elif var_info['type'] == 'single':
            # Single distribution variable
            try:
                result = fit_single_variable(var_name, var_info['data'])
                
                gmm_parameters[var_name] = {
                    "method": method,
                    "johnsonSU_has_result": False,
                    "weights": result['weights'],
                    "means": result['means'],
                    "stds": result['stds']
                }
                
                gmm_statistics[var_name] = {
                    "method": method,
                    "loss": result['loss'],
                    "empirical_stats": result['empirical_stats'],
                    "fitted_stats": result['fitted_stats'],
                    "target_stats": result['target_stats']
                }
                
                print(f"  {var_name} fitted successfully, loss: {result['loss']:.6f}")
                
            except Exception as e:
                print(f"  {var_name} fitting failed: {str(e)}")
                gmm_parameters[var_name] = {
                    "method": method,
                    "johnsonSU_has_result": False,
                    "error": str(e)
                }
                
        else:
            # Dual distribution variable
            gmm_parameters[var_name] = {"method": method}
            gmm_statistics[var_name] = {"method": method}
            
            for direction, var_data in var_info['directions'].items():
                print(f"  Fitting {var_name}_{direction}")
                
                try:
                    # Build data format suitable for fit_single_variable
                    suffix = f"_{direction}"
                    formatted_data = {
                        f'mean{suffix}': var_data['mean'],
                        f'variance{suffix}': var_data['variance'],
                        f'skewness{suffix}': var_data['skewness'],
                        f'kurtosis{suffix}': var_data['kurtosis']
                    }
                    
                    result = fit_single_variable(var_name, formatted_data, suffix=suffix)
                    
                    gmm_parameters[var_name][direction] = {
                        "johnsonSU_has_result": False,
                        "weights": result['weights'],
                        "means": result['means'],
                        "stds": result['stds']
                    }
                    
                    gmm_statistics[var_name][f"loss_{direction}"] = result['loss']
                    gmm_statistics[var_name][f"empirical_stats_{direction}"] = result['empirical_stats']
                    gmm_statistics[var_name][f"fitted_stats_{direction}"] = result['fitted_stats']
                    gmm_statistics[var_name][f"target_stats_{direction}"] = result['target_stats']
                    
                    print(f"    {var_name}_{direction} fitted successfully, loss: {result['loss']:.6f}")
                    
                except Exception as e:
                    print(f"    {var_name}_{direction} fitting failed: {str(e)}")
                    gmm_parameters[var_name][direction] = {
                        "johnsonSU_has_result": False,
                        "error": str(e)
                    }
    
    return gmm_parameters, gmm_statistics

def merge_with_johnsonsu_results():
    """
    Merge JohnsonSU and GMM results
    """
    print("=" * 80)
    print("Merging JohnsonSU and GMM results")
    print("=" * 80)
    
    # 1. Get JohnsonSU results
    means, variances, skews, kurts, labels_info = extract_moments_from_config()
    stds = tf.sqrt(variances)
    fit_results = johnson_su_fit_steps1to7(means, stds, skews, kurts)
    
    # Only convert to numpy when needed (for Python control flow and JSON serialization)
    has_solution = fit_results['has_solution'].numpy()
    
    # Keep other parameters as tensors until serialization is needed
    gamma = fit_results['gamma']
    delta = fit_results['delta']
    xi = fit_results['xi']
    lambda_ = fit_results['lambda']
    
    # 2. Build JohnsonSU results
    johnsonsu_parameters = {}
    johnsonsu_statistics = {}
    
    for i, label in enumerate(labels_info):
        if not has_solution[i]:
            continue  # Skip failed ones
        
        var_name = label['variable_name']
        method = label['method']
        direction = label['subtype'] if label['type'] == 'dual' else None
        
        # JohnsonSU parameters - convert to float when serializing
        params = {
            "johnsonSU_has_result": True,
            "gamma": float(gamma[i].numpy()),
            "delta": float(delta[i].numpy()),
            "xi": float(xi[i].numpy()),
            "lambda": float(lambda_[i].numpy())
        }
        
        if method in ["MUNO", "LSG"]:
            if var_name not in johnsonsu_parameters:
                johnsonsu_parameters[var_name] = {"method": method}
                johnsonsu_statistics[var_name] = {"method": method}
            johnsonsu_parameters[var_name][direction] = params
        else:
            johnsonsu_parameters[var_name] = {
                "method": method,
                **params
            }
    
    # 2.1 Add Tobit variables to JohnsonSU results (they do not need fitting)
    for var_name, var_data in Flow_info.items():
        method = var_data.get('method', '')
        if method == 'Tobit':
            scale_value = var_data.get('scale', None)
            if scale_value is not None:
                johnsonsu_parameters[var_name] = {
                    "method": method,
                    "johnsonSU_has_result": False,
                    "scale": scale_value
                }
                johnsonsu_statistics[var_name] = {
                    "method": method
                }
    
    # 3. Get GMM results
    gmm_parameters, gmm_statistics = fit_remaining_variables_with_gmm()
    
    # 4. Merge results
    final_parameters = {**johnsonsu_parameters, **gmm_parameters}
    final_statistics = {**johnsonsu_statistics, **gmm_statistics}
    
    # 5. Handle partially successful dual distribution variables
    for var_name in list(final_parameters.keys()):
        if var_name in johnsonsu_parameters and var_name in gmm_parameters:
            # Merge partially successful dual distribution variables
            merged_params = {"method": johnsonsu_parameters[var_name]["method"]}
            merged_stats = {"method": johnsonsu_statistics[var_name]["method"]}
            
            # Merge JohnsonSU successful parts
            for direction in ["pos", "neg"]:
                if direction in johnsonsu_parameters[var_name]:
                    merged_params[direction] = johnsonsu_parameters[var_name][direction]
                elif direction in gmm_parameters[var_name]:
                    merged_params[direction] = gmm_parameters[var_name][direction]
                    
                # Merge statistics
                for key in gmm_statistics[var_name]:
                    if direction in key:
                        merged_stats[key] = gmm_statistics[var_name][key]
            
            final_parameters[var_name] = merged_params
            final_statistics[var_name] = merged_stats
            
            # Remove duplicate entries
            if var_name in gmm_parameters:
                del gmm_parameters[var_name]
                del gmm_statistics[var_name]
    
    return final_parameters, final_statistics

def save_combined_results():
    """
    Save merged results to JSON files
    """
    final_parameters, final_statistics = merge_with_johnsonsu_results()
    
    # Save parameters file
    with open("combined_noise_parameters.json", "w", encoding="utf-8") as f:
        json.dump(final_parameters, f, indent=2, ensure_ascii=False)
    
    # Save statistics file
    with open("combined_noise_statistics.json", "w", encoding="utf-8") as f:
        json.dump(final_statistics, f, indent=2, ensure_ascii=False)
    
    print(f"Merged results saved:")
    print(f"  Parameters file: combined_noise_parameters.json")
    print(f"  Statistics file: combined_noise_statistics.json")
    print(f"  Total variables: {len(final_parameters)}")
    
    # Count statistics for each method
    method_counts = {}
    johnsonsu_count = 0
    gmm_count = 0
    tobit_count = 0
    
    for var_name, var_data in final_parameters.items():
        method = var_data["method"]
        method_counts[method] = method_counts.get(method, 0) + 1
        
        if method == "Tobit":
            # Tobit method variables
            tobit_count += 1
        elif method in ["MUNO", "LSG"]:
            # Dual distribution variables
            for direction in ["pos", "neg"]:
                if direction in var_data:
                    if var_data[direction].get("johnsonSU_has_result", False):
                        johnsonsu_count += 1
                    else:
                        gmm_count += 1
        else:
            # Single distribution variables
            if var_data.get("johnsonSU_has_result", False):
                johnsonsu_count += 1
            else:
                gmm_count += 1
    
    print(f"\nDistribution statistics:")
    print(f"  JohnsonSU successfully fitted: {johnsonsu_count}")
    print(f"  GMM fitted: {gmm_count}")
    print(f"  Tobit method: {tobit_count}")
    print(f"\nStatistics by method:")
    for method, count in method_counts.items():
        print(f"  {method}: {count} variables")

if __name__ == "__main__":
    save_combined_results() 