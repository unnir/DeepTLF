# DeepTLF Tests

This directory contains tests for the DeepTLF package. The tests cover both CPU and CUDA functionality, along with various edge cases and error conditions.

## Test Structure

- `conftest.py`: Contains pytest fixtures for data generation and model configuration
- `test_deeptlf.py`: Main test file containing all test cases

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run tests with detailed output:
```bash
pytest tests/ -v
```

To run a specific test:
```bash
pytest tests/test_deeptlf.py::test_name
```

## Test Coverage

The tests cover:
1. Model initialization and parameter validation
2. Classification and regression training
3. Device compatibility (CPU/CUDA)
4. Input validation and error handling
5. Model saving and loading
6. Edge cases (empty inputs, NaN values)

## Requirements

Make sure you have pytest installed:
```bash
pip install pytest
```

## Notes

- Tests use small datasets and few epochs to run quickly
- CUDA tests will automatically skip if no GPU is available
- Some tests use temporary directories for model checkpoints 