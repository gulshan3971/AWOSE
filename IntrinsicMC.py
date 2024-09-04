import numpy as np

class IntrinsicMC:
    def __init__(self, num_states):
        self.num_states = num_states
        self.transition_matrix = np.zeros((num_states, num_states))
        
    def set_transition_matrix(self, matrix):
        self.transition_matrix = matrix
    
    def perform_state_transition(self, current_state):
        return np.random.choice(range(self.num_states), p=self.transition_matrix[current_state])

    def update_transition_matrix(self, state_i, state_j, value):
        self.transition_matrix[state_i, state_j] += value
        self.transition_matrix[state_i] /= np.sum(self.transition_matrix[state_i])
