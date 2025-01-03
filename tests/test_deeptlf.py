import pytest
import numpy as np
import torch
from deeptlf import DeepTFL

def test_model_initialization(model_params):
    """Test if model initializes correctly with different parameters."""
    # Test classification
    model_class = DeepTFL(task='class', **model_params)
    assert model_class.task == 'class'
    
    # Test regression
    model_reg = DeepTFL(task='reg', **model_params)
    assert model_reg.task == 'reg'
    
    # Test device assignment
    assert hasattr(model_class, 'device')
    assert isinstance(model_class.device, torch.device)

def test_invalid_parameters():
    """Test if model raises appropriate errors for invalid parameters."""
    with pytest.raises(ValueError):
        DeepTFL(n_est=-1)  # Invalid n_est
    
    with pytest.raises(ValueError):
        DeepTFL(drop=2.0)  # Invalid dropout
        
    with pytest.raises(ValueError):
        DeepTFL(task='invalid')  # Invalid task

def test_classification_training(sample_data, model_params, device):
    """Test classification model training."""
    X, y = sample_data['X'], sample_data['y_class']
    
    model = DeepTFL(task='class', **model_params)
    model.fit(X, y)
    
    # Test if model components are properly initialized
    assert model.xgb_model is not None
    assert model.nn_model is not None
    assert model.TDE_encoder is not None
    
    # Test predictions
    y_pred = model.predict(X)
    assert len(y_pred) == len(y)
    assert np.all(y_pred >= 0) and np.all(y_pred < 3)  # 3 classes

def test_regression_training(sample_data, model_params, device):
    """Test regression model training."""
    X, y = sample_data['X'], sample_data['y_reg']
    # Reshape y to match model output shape
    y = y.reshape(-1, 1)
    
    model = DeepTFL(task='reg', **model_params)
    model.fit(X, y)
    
    # Test predictions
    y_pred = model.predict(X)
    assert len(y_pred) == len(y)
    assert y_pred.shape == y.shape
    assert isinstance(y_pred, np.ndarray)

def test_empty_input(model_params):
    """Test model behavior with empty input."""
    model = DeepTFL(**model_params)
    
    with pytest.raises(ValueError, match="Empty input data"):
        model.fit(np.array([]), np.array([]))
        
    with pytest.raises(ValueError, match="Empty input data"):
        model.predict(np.array([]))

def test_nan_input(sample_data, model_params):
    """Test model behavior with NaN input."""
    X, y = sample_data['X'].copy(), sample_data['y_class'].copy()
    X[0, 0] = np.nan
    
    model = DeepTFL(**model_params)
    
    with pytest.raises(ValueError, match="Input contains NaN values"):
        model.fit(X, y)

def test_input_validation(sample_data, model_params):
    """Test input validation during fit and predict."""
    X, y = sample_data['X'], sample_data['y_class']
    model = DeepTFL(**model_params)
    
    # Test mismatched lengths
    with pytest.raises(ValueError):
        model.fit(X, y[:-1])
    
    # Test wrong dimensions
    with pytest.raises(ValueError):
        model.fit(X.reshape(-1), y)

def test_device_compatibility(sample_data, model_params, device):
    """Test model works on both CPU and CUDA (if available)."""
    X, y = sample_data['X'], sample_data['y_class']
    model = DeepTFL(**model_params)
    model.fit(X, y)
    
    # Check if model is on the correct device
    assert next(model.nn_model.parameters()).device == device
    
    # Test predictions work on the device
    y_pred = model.predict(X)
    assert isinstance(y_pred, np.ndarray)

def test_model_save_load(sample_data, model_params, tmp_path):
    """Test model checkpoint saving and loading."""
    X, y = sample_data['X'], sample_data['y_class']
    checkpoint_path = tmp_path / "model_checkpoint.pt"
    
    # Train and save model
    model = DeepTFL(checkpoint_name=str(checkpoint_path), **model_params)
    model.fit(X, y)
    
    # Load model and make predictions
    new_model = DeepTFL(checkpoint_name=str(checkpoint_path), **model_params)
    new_model.TDE_encoder = model.TDE_encoder  # Need to share the encoder
    new_model.input_shape = model.input_shape
    new_model.nn_model = model.nn_model.__class__(
        model.input_shape, 
        model.hidden_dim, 
        model.n_layers, 
        len(np.unique(y)), 
        model.drop
    ).to(model.device)
    
    y_pred = new_model.predict(X)
    assert len(y_pred) == len(y) 

def test_output_shapes(sample_data, model_params, tmp_path):
    """Test output shapes for both regression and classification."""
    # Test regression shapes
    X, y_reg = sample_data['X'], sample_data['y_reg']
    reg_checkpoint = str(tmp_path / "reg_checkpoint.pt")
    reg_model = DeepTFL(task='reg', checkpoint_name=reg_checkpoint, **model_params)
    reg_model.fit(X, y_reg)
    reg_pred = reg_model.predict(X)
    assert reg_pred.shape == (len(X), 1)  # Regression output should be (n_samples, 1)
    
    # Test classification shapes
    y_class = sample_data['y_class']
    class_checkpoint = str(tmp_path / "class_checkpoint.pt")
    class_model = DeepTFL(task='class', checkpoint_name=class_checkpoint, **model_params)
    class_model.fit(X, y_class)
    class_pred = class_model.predict(X)
    assert class_pred.shape == (len(X),)  # Classification output should be (n_samples,)
    
    # Test single sample prediction shapes
    single_X = X[0:1]  # Single sample with shape (1, n_features)
    reg_single_pred = reg_model.predict(single_X)
    class_single_pred = class_model.predict(single_X)
    assert reg_single_pred.shape == (1, 1)  # Single regression prediction
    assert class_single_pred.shape == (1,)  # Single classification prediction 