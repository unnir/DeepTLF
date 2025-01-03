"""
Minimal example demonstrating DeepTLF usage for both classification and regression tasks.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from deeptlf import DeepTFL

def run_classification_example():
    """Run classification example using synthetic data."""
    print("\n=== Classification Example ===")
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    model = DeepTFL(
        task='class',
        n_est=10,
        max_depth=3,
        n_epoch=5,
        hidden_dim=64,
        drop=0.1,
        debug=True
    )
    
    print("Training classification model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Classification accuracy: {accuracy:.4f}")

def run_regression_example():
    """Run regression example using synthetic data."""
    print("\n=== Regression Example ===")
    
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=0.1,
        random_state=42
    )
    
    # Scale features and target
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X = X_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y.reshape(-1, 1))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train model
    model = DeepTFL(
        task='reg',
        n_est=10,
        max_depth=3,
        n_epoch=5,
        hidden_dim=64,
        drop=0.1,
        debug=True
    )
    
    print("Training regression model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"Regression MSE: {mse:.4f}")

if __name__ == "__main__":
    # Run both examples
    run_classification_example()
    run_regression_example() 