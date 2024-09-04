from IntrinsicMC import IntrinsicMC
from HiddenMarkovModel import HiddenMarkovModel
from ContextualLayer import ContextualLayer
import numpy as np

class AWOSE:
    def __init__(self, num_states, num_hidden_states, num_observations):
        self.intrinsic_mc = IntrinsicMC(num_states)
        self.hmm = HiddenMarkovModel(num_hidden_states, num_observations)
        self.contextual_layer = ContextualLayer(self.intrinsic_mc, self.hmm)

    def train(self, observed_symbols, hidden_states):
        for observed_symbol, hidden_state in zip(observed_symbols, hidden_states):
            self.contextual_layer.compute_consensus_messages(observed_symbol, hidden_state)
            self.contextual_layer.update_transition_matrix()

    def generate_sequence(self, length):
        state_sequence = []
        observation_sequence = []
        current_state = np.random.choice(range(self.intrinsic_mc.num_states))
        for _ in range(length):
            state_sequence.append(current_state)
            hidden_state = np.random.choice(range(self.hmm.num_hidden_states))
            observed_symbol = self.hmm.emit_observable_symbol(hidden_state)
            observation_sequence.append(observed_symbol)
            current_state = self.intrinsic_mc.perform_state_transition(current_state)
        return state_sequence, observation_sequence
