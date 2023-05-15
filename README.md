# Hidden Markov Model analysis toolkit
In this package you can find tools that might help you analyze hidden markov model data.
Usually what we want is either you estimate the model parameters (transmission, emission and initial)
or to estimate the hidden states given data set and model parameters. To address both problems we implemented 
the baum-welch algorithm (model's parameters estimation) and reconstruction (max likelihood of hidden states).

To install the package simply clone the git folder and run
```shell
pip install -e <PATH_TO_DIR>
```


## Baum-Welch
### Intro
This algorithm is a learning algorithm for hidden markov models that allows one to estimate the 
model parameters given observations. Denoting the observations as $Y_i$, hidden states $X_i$ and model parameters as 
$\theta$, then it tried to find a $\theta$ such that: $Pr(Y|\theta)$ is maximal.  
You can find more about it at: https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm

### Usage
Please refer the `baum_welch_example.py` in the example folder.  

```python
from hmm_analysis import baum_welch

estimations = baum_welch(observations, transition_guess, emission_guess, initial_guess, niters=100)
transition_estimation, emission_estimation, initial_estimation = estimations
```
* Note: that the algorithm requires some initial guesses for the model parameters, that is a transmission matrix, 
emission matrix and initial probability vector.
* Note: we are using a left multiplication for vector and matrix, e.g. the probability of hidden state $Pr(X_{i+1})$ 
is calculated as follows: $Pr(X_i) \times T$. Where $Pr(X_i)$ is a row vector and $T$ is the transition matrix.


## Reconstruction
### Intro
This algorithm is a maximum likelihood based estimation. Given a model with its parameters and a set of observations
we can use the forward-backward calculation to estimate the probability for all hidden states, that is: $Pr(X_i=j)$ 
for all $i, j$. Next we simply take the arg maximum over the value, meaning: $X_i = \argmax_{j}Pr(X_i=j)$

### Usage
Please refer the `reconstruction_example.py` in the example folder.

```python
from hmm_analysis import reconstruct

reconstructed_path = reconstruct(observations, transition, emission, initial)
```