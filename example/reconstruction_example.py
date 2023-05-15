from hmm_analysis import reconstruct
import numpy as np

"""
The goal of the function is to find the hidden state given a set of observations, transition and emission
parameters. That is: given observations Y and model parameters O we want to find list of states X
such that Pr(X|Y,O) is maximal.
"""

"""
We'll use the following data to demonstrate the reconstruction.
To keep everything clear we'll denote the dimension of the possible states as N
and the dimension of observation possible outcome as K and the length of observations as L
In this case N = 2 (two possible state names) and K = 3 as we have 3 unique observations names and
L = 3 (we have only 3 observations)
"""


observations_as_strings = ("normal", "cold", "dizzy")       # dim = L = 3
possible_hidden_states = ("Healthy", "Fever")               # dim = N = 2
initial_raw = {"Healthy": 0.6, "Fever": 0.4}                # dim = N = 2
transition_raw = {                                          # dim = N x N
    "Healthy": {"Healthy": 0.7, "Fever": 0.3},
    "Fever": {"Healthy": 0.4, "Fever": 0.6},
}
emission_raw = {                                            # dim N x K
    "Healthy": {"normal": 0.5, "cold": 0.4, "dizzy": 0.1},
    "Fever": {"normal": 0.1, "cold": 0.3, "dizzy": 0.6},
}

"""
First we need to convert all the parameters into numpy matrices.
Instead of working with strings we change it to integers
"""

observations_as_numbers = np.array([0, 1, 2])  # 'normal' = 0, 'cold' = 1, ...
state_to_index = {s: i for i, s in enumerate(possible_hidden_states)}  # mapping state to index. e.g. Healthy = 0, ...
index_to_state = {idx: state_name for state_name, idx in state_to_index.items()}  # the inverse mapping, this
# will be useful at the end when we want to convert back the number to their state name

initial = list(initial_raw.values())  # [0.6, 0.4] meaning we start with prob 0.6 at state 0 which is Healthy
transition = [list(v.values()) for v in transition_raw.values()]
emission = [list(v.values()) for v in emission_raw.values()]

"""
Converting the matrices to numpy arrays
"""

initial = np.array(initial)
transition = np.array(transition)
emission = np.array(emission)

"""
Now we can reconstruct the hidden state:
"""

reconstructed_path_as_numbers = reconstruct(observations_as_numbers, transition, emission, initial)
print(f'Reconstruction as numbers: {reconstructed_path_as_numbers}')

"""
We see that we got [0, 0, 1], these are the best estimation of the true states.
We can also cast them back to their string names:
"""
reconstructed_path_as_strings = list(map(lambda x: index_to_state[x], reconstructed_path_as_numbers))
print(f'Reconstruction as strings: {reconstructed_path_as_strings}')
