from hmm_analysis import baum_welch
import numpy as np


"""
Going over the usage of Baum-Welch algorithm. 
The inputs will be some guesses about the parameters of the model and a sequence of observations.
The function should return a new set of model parameters that would maximize likelihood.

NOTE: this implementation is log based, and all model parameters are log casted INSIDE the function
"""


"""
We'll demonstrate it by generating a data set with the model parameters from the reconstruction example
and let the algorithm to estimate them only given observation:

(one can look at the reconstruction example first to understand the model parameters)
"""

initial_raw = np.array([0.6, 0.4])
"""
Initial probabilities:
Defines probabilities of starting in each hidden state.
Shape: (N,)
Where N is the number of possible hidden states.
Entry A[i] = P(state_0 = i)
In this example we have two hidden states: Healthy and Fever.
"""
transition_matrix = np.array([[0.7, 0.3],
                              [0.4, 0.6]])
"""
Transition matrix:
Defines probabilities of moving between hidden states.
Shape: (N, N)
Entry A[i, j] = P(state_t = j | state_{t-1} = i)
Rows must sum to 1.
"""
emission_matrix = np.array([[0.5, 0.4, 0.1],
                            [0.1, 0.3, 0.6]])
"""
Emission matrix:
Defines probabilities of observing each symbol from a given hidden state.
Shape: (N, M)
Where M is the number of possible observations.
Entry B[i, k] = P(observation = k | state = i)
Rows must sum to 1.
In this example we have three possible observations: normal, cold, dizzy.
"""

true_states = []
prev_state_prob = initial_raw
for _ in range(1000000):
    new_state_prob = prev_state_prob @ transition_matrix
    new_state = np.random.choice([0, 1], p=new_state_prob)
    if new_state == 0:
        prev_state_prob = np.array([1, 0])
    else:
        prev_state_prob = np.array([0, 1])

    true_states.append(new_state)

observations = []
for state in true_states:
    observations.append(np.random.choice([0, 1, 2], p=emission_matrix[state]))


"""
After we created the observations we can start feeding the algorithm the parameters and see whether it can
estimate our emission, transition and initial parameters.

First we need to write some guesses and cast the observations to be a np.ndarray
"""
initial_guess = np.array([0.7, 0.3])


transition_guess = np.array([[0.6, 0.4], [0.2, 0.8]])
emission_guess = np.array([[0.7, 0.25, 0.05], [0.1, 0.25, 0.65]])
observations = np.array(observations)

"""
Running the algorithm
"""
result = baum_welch(observations, transition_guess, emission_guess, initial_guess, niters=100)
transition_estimation, emission_estimation, initial_estimation = result.transition, result.emission, result.initial

"""
We can compare them to the original parameters and see that the difference is rather small except for the 
initial probabilities
"""


def diff(a, b):
    return np.max(np.abs(a - b))


print('-' * 20)
print(f'Transition original: \n{transition_matrix}\nTransition estimation: \n{transition_estimation}')
print(f'Transition matrix difference: {diff(transition_estimation, transition_matrix)}')
print('-' * 20)
print(f'Emission original: \n{emission_matrix}\nEmission estimation: \n{emission_estimation}')
print(f'Emission matrix difference: {diff(emission_estimation, emission_matrix)}')
print('-' * 20)
print(f'Initial original: \n{initial_raw}\nInitial estimation: \n{initial_estimation}')
print('-' * 20)

