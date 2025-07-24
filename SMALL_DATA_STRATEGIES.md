# Small-Data-Friendly Trading Strategies

This document describes the new machine learning strategies that have been implemented to enhance trading signal classification, particularly for small datasets and robust performance.

## Overview

The following strategies have been added to the trading signals framework:

1. **Ridge Classifier Strategy** (`ridge_strategy.py`)
2. **Gaussian Process Classifier Strategy** (`gp_strategy.py`) 
3. **Linear Discriminant Analysis Strategy** (`lda_strategy.py`)
4. **PyTorch Neural Network Strategy** (`pytorch_nn_strategy.py`) - New!
5. **NGBoost Strategy** (`ngboost_strategy.py`) - Previously implemented
6. **TabNet Strategy** (`tabnet_strategy.py`) - Previously implemented

## Strategy Descriptions

### 1. Ridge Classifier Strategy (RidgeClassifierOptunaStrategy)

**Best for:** Small datasets, high-dimensional data, noisy features

**Key Features:**
- L2 regularization helps prevent overfitting on small datasets
- Built-in feature scaling with StandardScaler
- Optional feature selection using SelectKBest
- Fast training and prediction
- Robust to multicollinearity

**Hyperparameters optimized:**
- `alpha`: Regularization strength (1e-6 to 1000.0)
- `solver`: Optimization algorithm ('auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga')
- `fit_intercept`: Whether to fit intercept term
- `use_feature_selection`: Whether to apply feature selection
- `k_features`: Number of features to select (if feature selection enabled)

**Advantages:**
- Excellent for small datasets (works well with 10+ samples)
- Very fast training and prediction
- Regularization prevents overfitting
- Handles multicollinearity well
- Provides decision function scores for confidence estimation

### 2. Gaussian Process Classifier Strategy (GaussianProcessOptunaStrategy)

**Best for:** Small datasets with complex patterns, uncertainty quantification

**Key Features:**
- Provides prediction uncertainties (key advantage)
- Non-parametric approach - flexible model
- Automatically limits dataset size to 200 samples for efficiency (O(n³) complexity)
- Multiple kernel options for different data patterns
- Built-in feature scaling and optional feature selection

**Hyperparameters optimized:**
- `kernel_type`: Kernel function ('rbf', 'matern', 'rbf_white', 'matern_white')
- `length_scale`: Characteristic length scale (0.1 to 10.0)
- `nu`: Smoothness parameter for Matern kernel (0.5, 1.5, 2.5)
- `noise_level`: White noise level for robustness (1e-5 to 1e-1)
- `max_iter_predict`: Maximum iterations for prediction (50 to 200)
- Feature selection parameters

**Advantages:**
- Provides prediction uncertainties - critical for risk management
- Works very well on small datasets
- Non-parametric - doesn't assume specific functional form
- Kernels can capture complex patterns
- Bayesian approach provides principled uncertainty estimates

**Disadvantages:**
- O(n³) computational complexity - limited to ~200 samples
- Can be slow for larger datasets
- Memory intensive

### 3. Linear Discriminant Analysis Strategy (LDAOptunaStrategy)

**Best for:** Small datasets, linear decision boundaries, when classes are well-separated

**Key Features:**
- Assumes Gaussian distributions for each class
- Finds linear combinations that best separate classes
- Built-in dimensionality reduction capability
- Works well when class covariances are similar
- Very efficient for small datasets

**Hyperparameters optimized:**
- `solver`: Algorithm for eigenvalue decomposition ('svd', 'lsqr', 'eigen')
- `shrinkage`: Regularization parameter (0.0 to 1.0, for lsqr/eigen solvers)
- `n_components`: Number of discriminant components (for svd/eigen solvers)
- `tol`: Tolerance for convergence (1e-6 to 1e-2)
- Feature selection parameters

**Advantages:**
- Very efficient on small datasets
- Built-in dimensionality reduction
- Provides explained variance ratios
- Fast training and prediction
- Good interpretability with linear discriminant coefficients

**Disadvantages:**
- Assumes Gaussian distributions
- Assumes equal class covariances for optimal performance
- Linear decision boundaries only

### 4. PyTorch Neural Network Strategy (PyTorchNeuralNetOptunaStrategy)

**Best for:** Small to medium datasets with complex patterns, economic data with non-linear relationships

**Key Features:**
- **Optimized for small datasets** with simplified architecture
- Automatic fallback to LogisticRegression for very small/problematic datasets
- Uses skorch for scikit-learn compatibility
- GPU acceleration with CUDA support
- Aggressive regularization (high dropout, weight decay)
- Smaller batch sizes suitable for economic data
- Feature selection for high-dimensional economic indicators
- Early stopping to prevent overfitting

**Hyperparameters optimized:**
- `n_layers`: Number of hidden layers (1 to 2) - reduced for small datasets
- `hidden_dim_X`: Number of units in each layer (16-64 for first layer, 8-32 for second)
- `lr`: Learning rate (1e-4 to 1e-2) - more conservative range
- `batch_size`: Training batch size (16, 32, 64) - smaller for economic data
- `dropout_rate`: Dropout probability (0.3 to 0.7) - higher for regularization
- `activation`: Activation function ('relu', 'tanh', 'elu', 'selu')
- `weight_decay`: L2 regularization strength (1e-4 to 1e-1) - stronger regularization
- `max_epochs`: Maximum training epochs (50 to 200) - fewer to prevent overfitting
- `use_feature_selection`: Whether to apply feature selection (crucial for economic data)
- `k_features`: Number of features to select (aggressive selection for small datasets)

**Advantages:**
- **Specifically optimized for small datasets** (30+ samples)
- Automatic detection of problematic datasets with fallback
- Strong regularization prevents overfitting on economic data
- Feature selection handles high-dimensional economic indicators
- Smaller architectures generalize better on limited data
- GPU acceleration for faster training
- Handles imbalanced classes (common in economic data)

**Disadvantages:**
- May not utilize full potential of neural networks on larger datasets
- Still requires more data than simple linear models
- Less interpretable than linear models
- Computational overhead compared to simpler methods

**Fallback Behavior:**
- Automatically switches to LogisticRegression with strong regularization for:
  - Datasets with < 30 samples
  - Highly imbalanced data (minority class < 10%)
  - Training failures or numerical instability

## Usage Examples

### Running Individual Strategies

```python
# Ridge Classifier
from ridge_strategy import RidgeClassifierOptunaStrategy
strategy = RidgeClassifierOptunaStrategy(n_trials=20, n_splits=5)

# Gaussian Process  
from gp_strategy import GaussianProcessOptunaStrategy
strategy = GaussianProcessOptunaStrategy(n_trials=10, n_splits=3)  # Fewer trials due to computational cost

# Linear Discriminant Analysis
from lda_strategy import LDAOptunaStrategy
strategy = LDAOptunaStrategy(n_trials=15, n_splits=5)

# PyTorch Neural Network (optimized for small datasets)
from pytorch_nn_strategy import PyTorchNeuralNetOptunaStrategy
strategy = PyTorchNeuralNetOptunaStrategy(n_trials=15, n_splits=5)  # Reduced trials for small data efficiency
```

### Switching Strategies in Performance Analysis

Edit `performance_analysis.py` and uncomment the desired strategy:

```python
# strategy = RidgeClassifierOptunaStrategy()
# strategy = GaussianProcessOptunaStrategy()
# strategy = LDAOptunaStrategy()
# strategy = PyTorchNeuralNetOptunaStrategy()
```

## Recommendations by Dataset Size

### Very Small Datasets (< 50 samples)
1. **Ridge Classifier** - Most robust, fast
2. **LDA** - If classes are well-separated
3. **Gaussian Process** - If uncertainty is critical
4. **PyTorch Neural Network** - Will automatically use fallback for very small datasets

### Small Datasets (50-200 samples)
1. **PyTorch Neural Network** - Now optimized for small datasets with strong regularization
2. **Gaussian Process** - Best for uncertainty quantification
3. **Ridge Classifier** - Fast and robust baseline
4. **LDA** - If linear separation assumption holds

### Medium Datasets (200-1000 samples)
1. **PyTorch Neural Network** - Great for complex patterns with sufficient data
2. **Ridge Classifier** - Very fast, good baseline
3. **NGBoost** - If you need probabilistic predictions
4. **TabNet** - For complex non-linear patterns

### Large Datasets (1000+ samples)
1. **PyTorch Neural Network** - Excellent for complex patterns
2. **TabNet** - Advanced neural network with attention
3. **NGBoost** - Robust probabilistic predictions
4. **CatBoost/XGBoost** - Tree-based ensemble methods

## Feature Importance and Interpretability

- **Ridge Classifier**: Absolute values of linear coefficients
- **Gaussian Process**: Feature importance not directly available, but provides prediction uncertainties
- **LDA**: Linear discriminant coefficients and explained variance ratios
- **PyTorch Neural Network**: Feature importance not directly available, but provides prediction probabilities
- **NGBoost**: Feature importance from base learners (when not using fallback)
- **TabNet**: Built-in attention-based feature importance

## Error Handling and Robustness

All strategies include:
- Data cleaning (remove NaN and infinity values)
- Cross-validation with TimeSeriesSplit
- Error handling in optimization loops
- Fallback mechanisms for problematic datasets
- Data validation checks

## Performance Considerations

- **Fastest**: Ridge Classifier, LDA
- **Most Memory Efficient**: Ridge Classifier, LDA
- **Best for Uncertainty**: Gaussian Process, NGBoost
- **Most Flexible**: PyTorch Neural Network, Gaussian Process, TabNet
- **Best for Small Data**: Ridge Classifier, LDA, Gaussian Process
- **Best for Large Data**: PyTorch Neural Network, TabNet, NGBoost
- **GPU Accelerated**: PyTorch Neural Network, TabNet

## Dependencies

Ensure the following packages are installed:
```bash
pip install scikit-learn optuna ngboost pytorch-tabnet torch skorch
```

## Integration with Backtest Pipeline

All strategies follow the same interface and are compatible with:
- Optuna hyperparameter optimization
- TimeSeriesSplit cross-validation
- The existing backtest framework
- Performance analysis tools

The strategies can be easily swapped in the performance analysis by changing one line of code, making comparative analysis straightforward.
