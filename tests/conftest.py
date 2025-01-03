import pytest
import numpy as np
import torch

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 10)  # 100 samples, 10 features
    y_reg = np.random.randn(100, 1)  # Regression target with shape (n_samples, 1)
    y_class = np.random.randint(0, 3, 100)  # Classification target (3 classes)
    return {
        'X': X,
        'y_reg': y_reg,
        'y_class': y_class
    }

@pytest.fixture
def device():
    """Return available device (cuda if available, else cpu)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def model_params():
    """Return default model parameters for testing."""
    return {
        'n_est': 5,  # Small number for quick testing
        'max_depth': 3,
        'drop': 0.1,
        'xgb_lr': 0.1,
        'batch_size': 32,
        'n_epoch': 2,  # Small number for quick testing
        'hidden_dim': 64,
        'n_layers': 2,
        'debug': True
    } 