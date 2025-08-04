import sys
import os

# append
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noises.generate_noise_distributions import (
    generate_all_distributions,
    sample_from_distributions,
)

# example usage
if __name__ == "__main__":
    # Example usage
    print("Generating noise distributions...")

    # Generate all distributions
    distributions = generate_all_distributions()

    print(f"Successfully generated distributions for {len(distributions)} variables")

    # ------------Start Show distribution type statistics ------------
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
    johnson_su_count = sum(1 for dt in distribution_types.values() if "JohnsonSU" in dt)
    transformed_count = sum(
        1 for dt in distribution_types.values() if "TransformedDistribution" in dt
    )
    mixture_count = sum(
        1 for dt in distribution_types.values() if "MixtureSameFamily" in dt
    )
    tobit_count = sum(1 for dt in distribution_types.values() if "None" in dt)

    print(f"  Johnson SU distributions: {johnson_su_count}")
    print(f"  Linear transformed GMM distributions: {transformed_count}")
    print(f"  Original GMM distributions: {mixture_count}")
    print(f"  Tobit method: {tobit_count}")

    # ------------End Show distribution type statistics ------------

    # ------------Start Sampling example ------------
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
    print(
        "Note: GMM distributions have been automatically linearly transformed to match target mean and variance"
    )
    print("All sample shapes are [batch_size, 1]")
