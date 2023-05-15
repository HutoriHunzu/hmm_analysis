from hmm_analysis import reconstruct
import numpy as np

obs = ("normal", "cold", "dizzy")
states = ("Healthy", "Fever")
start_p = {"Healthy": 0.6, "Fever": 0.4}
trans_p = {
    "Healthy": {"Healthy": 0.7, "Fever": 0.3},
    "Fever": {"Healthy": 0.4, "Fever": 0.6},
}
emit_p = {
    "Healthy": {"normal": 0.5, "cold": 0.4, "dizzy": 0.1},
    "Fever": {"normal": 0.1, "cold": 0.3, "dizzy": 0.6},
}

expected_states_names = ['Healthy', 'Healthy', 'Fever']


def test_reconstruction():

    # preparing data
    obs = np.array(list(range(3)))

    initial_log = np.log(list(start_p.values()))
    transition_log = np.log([list(v.values()) for v in trans_p.values()])
    emission_log = np.log([list(v.values()) for v in emit_p.values()])

    reconstructed_path = reconstruct(obs, transition_log, emission_log, initial_log)

    # expected path
    states_to_index = {s: i for i, s in enumerate(states)}
    expected_path = [states_to_index[n] for n in expected_states_names]

    assert (np.all(expected_path == reconstructed_path))
