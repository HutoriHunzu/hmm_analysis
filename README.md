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

## Baum-Welch Parameter Estimation

The main function for HMM parameter estimation. Automatically detects single vs multi-sequence data.

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

# Access results
print("Final log-likelihood:", result.likelihood_log)
print("Estimated transition matrix:", result.transition)
print("Estimated emission matrix:", result.emission)
print("Estimated initial probabilities:", result.initial)
```

### Multi-Sequence Support

For multiple observation sequences, simply pass a list of arrays:

```python
# Multiple sequences - automatically detected
multi_observations = [
    np.array([0, 1, 0, 2]),
    np.array([1, 2, 1, 0, 2]),
    np.array([0, 0, 1])
]

result = baum_welch(multi_observations, transition_guess, emission_guess, initial_guess, niters=50)
```

## Return Object

Both `baum_welch()` and `baum_welch_iter()` return a `BaumWelchResult` object with:

- **`result.transition`**: Estimated transition matrix (regular space)
- **`result.emission`**: Estimated emission matrix (regular space)
- **`result.initial`**: Estimated initial probability vector (regular space)
- **`result.likelihood_log`**: Log-likelihood of the data given current parameters

## Matrix Dimensions and Roles

### Transition Matrix
- **Shape**: `(n_states, n_states)`
- **Role**: `transition[i, j] = P(next_state=j | current_state=i)`
- **Rows**: Each row represents a current state
- **Columns**: Each column represents a next state
- **Row sums**: Each row sums to 1 (probability distribution over next states)

### Emission Matrix
- **Shape**: `(n_states, n_observations)`
- **Role**: `emission[i, j] = P(observation=j | state=i)`
- **Rows**: Each row represents a hidden state
- **Columns**: Each column represents an observation symbol
- **Row sums**: Each row sums to 1 (probability distribution over observations)

### Initial Probabilities
- **Shape**: `(n_states,)`
- **Role**: `initial[i] = P(first_state=i)`
- **Constraint**: Vector sums to 1

### Matrix Multiplication Convention
Uses **left multiplication**: `P(X_i) * T` where `P(X_i)` is a row vector and `T` is the transition matrix.

## Advanced: Custom Stopping Criteria

For advanced users who need custom stopping mechanisms beyond fixed iterations, use `baum_welch_iter()`:

```python
from hmm_analysis import baum_welch_iter

# Infinite iterator - you control when to stop
iterator = baum_welch_iter(observations, transition_guess, emission_guess, initial_guess)

# Example 1: Stop on convergence
prev_likelihood = None
for i, result in enumerate(iterator):
    likelihood = result.likelihood_log

    if prev_likelihood is not None:
        improvement = abs(likelihood - prev_likelihood)
        if improvement < 1e-6:  # Convergence threshold
            print(f"Converged after {i+1} iterations")
            break

    prev_likelihood = likelihood
    if i >= 1000:  # Safety limit
        break

# Example 2: Stop on time limit
import time
start_time = time.time()
for i, result in enumerate(iterator):
    if time.time() - start_time > 30:  # 30 second limit
        print(f"Stopped after {i+1} iterations due to time limit")
        break

# Example 3: Custom criteria
for i, result in enumerate(iterator):
    if result.likelihood_log > -10.0:  # Good enough likelihood
        print(f"Reached target likelihood after {i+1} iterations")
        break
```

If you have your own stopping function, you can easily integrate it:

```python
def my_custom_stop_criterion(result, iteration, start_time):
    """Your custom stopping logic"""
    if iteration > 500:
        return True
    if result.likelihood_log > -5.0:
        return True
    if time.time() - start_time > 60:
        return True
    return False

# Use with iterator
start_time = time.time()
for i, result in enumerate(baum_welch_iter(observations, transition_guess, emission_guess, initial_guess)):
    if my_custom_stop_criterion(result, i, start_time):
        break
```

## Hidden State Reconstruction

```python
from hmm_analysis import reconstruct

# With estimated or known HMM parameters
hidden_states = reconstruct(observations, result.transition, result.emission, result.initial)
print("Most likely hidden states:", hidden_states)
```

## Examples

See the `examples/` directory for detailed usage patterns:
- `baum_welch_example.py` - Parameter estimation walkthrough
- `reconstruction_example.py` - Hidden state inference
- `multi_sequence_example.py` - Multiple observation sequences

## Citation

If you use this software in your research, please cite it using the information provided in the repository's `CITATION.cff` file, or use the following BibTeX entry:

```bibtex
@software{goldblatt2025hmm,
  title = {HMM Analysis Toolkit: Fast Hidden Markov Model Analysis with Numba},
  author = {Goldblatt, Uri},
  year = {2025},
  url = {https://github.com/HutoriHunzu/hmm_analysis},
  version = {0.1.0}
}
```

## Algorithm Reference

For theoretical background, see: https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm