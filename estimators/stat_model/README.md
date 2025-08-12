# Statistical Models Directory

This directory contains implementations of statistical models for the simulator project, with a focus on Complementary Log-Log (CLogLog) link functions using Iteratively Reweighted Least Squares (IRLS) algorithms.

## Files Overview

### Binary Classification Models

#### `cloglog_irls.py`
**Original Function-Based Implementation**
- Contains the original `cloglog_irls_estimator()` function
- Uses CPU-only execution with `tf.device("/CPU:0")`
- Implements classic econometric GLM methodology
- Includes early stopping based on validation loss
- **Use case**: Reference implementation and for comparison with new models

#### `binary_cloglog_irls.py`
**TensorFlow Model Implementation (CPU-Optimized)**
- TensorFlow Keras model that closely follows the original algorithm
- CPU-only execution to ensure numerical consistency with original
- Uses full diagonal matrix construction like the original function
- Compatible with standard TensorFlow model interface (`model.fit()`, `model.predict()`)
- **Use case**: When you need TensorFlow model interface but want identical results to original

#### `binary_cloglog_irls_original.py`
**Backup Copy**
- Identical copy of `binary_cloglog_irls.py`
- **Use case**: Backup/reference version

#### `binary_cloglog_irls_gpu.py`
**GPU-Optimized Implementation**
- Memory-efficient version designed for GPU execution
- Uses vectorized operations instead of full diagonal matrices
- Includes `@tf.function` decorators for performance
- Uses Cholesky decomposition for better numerical stability
- **Use case**: When working with larger datasets or when GPU acceleration is needed

### Multinomial Classification Models

#### `cloglog_multi_irls.py`
**Multinomial CLogLog with Newton-Raphson**
- `MultinomialCLogLogIRLS` class for 3-category classification
- Uses Newton-Raphson optimization with Hessian matrix computation
- Supports ordered categorical outcomes (e.g., -1, 0, +1)
- Includes cutoff parameters for category boundaries
- **Use case**: Multi-class classification with CLogLog link

## Algorithm Comparison

| Implementation | Device | Memory Usage | Speed | Numerical Precision | Use Case |
|---------------|--------|--------------|-------|-------------------|----------|
| `cloglog_irls.py` | CPU | High | Slow | Reference | Original/Comparison |
| `binary_cloglog_irls.py` | CPU | High | Slow | Identical to Original | TF Model + Original Results |
| `binary_cloglog_irls_gpu.py` | GPU/CPU | Low | Fast | Very Close | Large Datasets |
| `cloglog_multi_irls.py` | GPU/CPU | Medium | Medium | High | Multi-class Problems |

## Usage Examples

### Binary Classification (Original-Compatible)
```python
from estimators.stat_model.binary_cloglog_irls import BinaryCLogLogIRLS

model = BinaryCLogLogIRLS(
    n_features=X_train.shape[1],
    max_iterations=25,
    tolerance=1e-6,
    patience=5
)

model.fit(X_train, y_train, validation_data=(X_test, y_test))
predictions = model.predict(X_test)
coefficients = model.get_coefficients()
```

### Binary Classification (GPU-Optimized)
```python
from estimators.stat_model.binary_cloglog_irls_gpu import BinaryCLogLogIRLSGPU

model = BinaryCLogLogIRLSGPU(
    n_features=X_train.shape[1],
    max_iterations=25,
    regularization=1e-6
)

model.fit(X_train, y_train, validation_data=(X_test, y_test))
```

### Multinomial Classification
```python
from estimators.stat_model.cloglog_multi_irls import MultinomialCLogLogIRLS

model = MultinomialCLogLogIRLS(
    n_features=X_train.shape[1],
    n_categories=3,
    max_iterations=30,
    tolerance=1e-6
)

model.fit(X_train, y_train, validation_data=(X_test, y_test))
```

## Mathematical Background

### Complementary Log-Log Link Function
The complementary log-log (cloglog) link function is defined as:
```
η = log(-log(1 - π))
```
Where π is the probability and η is the linear predictor.

The inverse link function is:
```
π = 1 - exp(-exp(η))
```

### IRLS Algorithm
The Iteratively Reweighted Least Squares algorithm:
1. Calculate working weights: `s = (dμ/dη)² / var(y)`
2. Calculate working response: `z = η + (y - μ) / (dμ/dη)`
3. Solve weighted least squares: `W = (X'SX)⁻¹X'Sz`
4. Repeat until convergence

### Newton-Raphson (Multinomial)
For multinomial models, uses full Newton-Raphson with Hessian matrix:
1. Compute gradient: `∇L(θ)`
2. Compute Hessian: `H(θ)`
3. Update: `θ_{new} = θ_{old} - αH⁻¹∇L`

## Performance Notes

- **Memory**: GPU version uses ~50% less memory than CPU version for large datasets
- **Speed**: GPU version can be 2-5x faster depending on dataset size
- **Precision**: All implementations maintain high numerical precision
- **Convergence**: Newton-Raphson typically converges faster than IRLS for multinomial models

## Dependencies

- TensorFlow >= 2.x
- NumPy
- scikit-learn (for train_test_split in examples)

## Error Handling

All implementations include:
- Numerical stability checks (clipping extreme values)
- Regularization for ill-conditioned matrices
- Fallback to pseudo-inverse when matrix inversion fails
- Early stopping to prevent overfitting
