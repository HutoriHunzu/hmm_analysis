# HMM Analysis Toolkit

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Numba Accelerated](https://img.shields.io/badge/Numba-Accelerated-orange.svg)](https://numba.pydata.org/)

A fast Python package for Hidden Markov Model analysis using NumPy and Numba. Implements the Baum-Welch algorithm for parameter estimation and forward-backward reconstruction for hidden state inference.

## Installation

```bash
git clone https://github.com/HutoriHunzu/hmm_analysis.git
cd hmm_analysis
pip install -e .
```

## Quick Start

### Parameter Estimation (Baum-Welch)

```python
import numpy as np
from hmm_analysis import baum_welch

# Your observation sequence
observations = np.array([0, 1, 0, 2, 1, 0])

# Initial parameter guesses (required)
transition_guess = np.array([[0.7, 0.3], [0.4, 0.6]])
emission_guess = np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])
initial_guess = np.array([0.6, 0.4])

# Estimate parameters
result = baum_welch(observations, transition_guess, emission_guess, initial_guess, niters=100)
print("Estimated transition matrix:", result.transition)
print("Estimated emission matrix:", result.emission)
```

### Hidden State Reconstruction

```python
from hmm_analysis import reconstruct

# With known HMM parameters
hidden_states = reconstruct(observations, transition_matrix, emission_matrix, initial_probs)
print("Most likely hidden states:", hidden_states)
```

### Iterator Access to Training Progress

```python
from hmm_analysis import baum_welch_iter

# Access intermediate results during training
for i, result in enumerate(baum_welch_iter(observations, transition_guess, emission_guess, initial_guess, niters=50)):
    if i % 10 == 0:  # Print every 10th iteration
        print(f"Iteration {i+1}: Likelihood = {result.likelihood_log}")
```

## Important Notes

- **Matrix Orientation**: Uses left multiplication convention: `P(X_i) × T` where `P(X_i)` is a row vector and `T` is the transition matrix
- **Initial Guesses**: Baum-Welch requires reasonable initial parameter estimates to converge properly
- **Numerical Stability**: All computations performed in log-space using Numba-optimized functions

## Examples

See the `examples/` directory for detailed usage patterns:
- `baum_welch_example.py` - Parameter estimation walkthrough
- `reconstruction_example.py` - Hidden state inference
- `multi_sequence_example.py` - Multiple observation sequences

## Algorithm Reference

For theoretical background, see: https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm