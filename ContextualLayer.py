import numpy as np

class ContextualLayer:
    def __init__(self, intrinsic_mc, hmm):
        self.intrinsic_mc = intrinsic_mc
        self.hmm = hmm
        self.consensus_messages = np.zeros(intrinsic_mc.num_states)

    def compute_consensus_messages(self, observed_symbol, hidden_state):
        for i in range(self.intrinsic_mc.num_states):
            self.consensus_messages[i] += np.log(self.hmm.emission_matrix[hidden_state, observed_symbol])
        self.consensus_messages = np.exp(self.consensus_messages)
        self.consensus_messages /= np.sum(self.consensus_messages)
    
    def update_transition_matrix(self):
        for i in range(self.intrinsic_mc.num_states):
            for j in range(self.intrinsic_mc.num_states):
                self.intrinsic_mc.update_transition_matrix(i, j, self.consensus_messages[j])
