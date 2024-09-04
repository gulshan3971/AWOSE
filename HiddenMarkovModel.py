import numpy as np

class HiddenMarkovModel:
    def __init__(self, num_hidden_states, num_observations):
        self.num_hidden_states = num_hidden_states
        self.emission_matrix = np.random.rand(num_hidden_states, num_observations)
        self.normalize_emission_matrix()

    def normalize_emission_matrix(self):
        for i in range(self.num_hidden_states):
            self.emission_matrix[i] /= np.sum(self.emission_matrix[i])
    
    def emit_observable_symbol(self, hidden_state):
        return np.random.choice(range(self.emission_matrix.shape[1]), p=self.emission_matrix[hidden_state])
