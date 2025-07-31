# Noise Distribution Fitting and Sampling

This folder contains scripts that implement two key functionalities:
1. Matching given **mean**, **variance**, **skewness**, and **kurtosis** using **Gaussian Mixture Models (GMM)** and **JohnsonSU distributions**.
2. Generating distribution functions with **TensorFlow Probability (TFP)** for direct sampling.

## File Structure

- `fitting_gmm.py`: Contains the function to fit variables using Gaussian Mixture Models (GMM).
- `fitting_johnsonSU.py`: Contains the function to fit variables using JohnsonSU distributions.
- `fit_all_noise.py`: Calls the above two scripts to fit all variables.

## Usage

1. **Fitting Noise Distributions**
   - Run `fit_all_noise.py` to fit the noise of all variables. 
   - The results, including the parameters of the fitted distributions, are saved in `combined_noise_parameters.json`.

2. **Generating and Sampling Distributions**
   - In `generate_noise_distribution.py`:
     - The `generate_all_distributions` function reads the parameters from `combined_noise_parameters.json` and generates TensorFlow Probability distributions for all variables.
     - The `sample_from_distributions` function samples data points from these distributions, with dimensions `[batchsize, 1]`.

3. **Generating Starting Points for Variables (for reference only)**
   - In `company_tensor_generator.py`:
     - The `generate_company_tensors` function generates starting points for each variable, with dimensions `[batchsize, 1]`.

## Example
   - Navigate to `noises` folder first.
   - In `__main__` of `generate_noise_distributions.py` illustrates an example on how the functions are used. The return is a dictionary. Each item in the dictionary is a tensor of noises with `[batchsize,1]`
   - In `__main__` of `company_tensor_generator.py` an example is also illustrated. Simply use `generate_company_tensors(batch_size=10000)`.