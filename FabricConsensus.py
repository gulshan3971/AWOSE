from hfc.fabric import Client as FabricClient
from AWOSE import AWOSE

class FabricConsensus:
    def __init__(self):
        self.client = FabricClient(net_profile="fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/connection-org1.yaml")
        self.client.new_channel('mychannel')
        self.num_states = 5
        self.num_hidden_states = 3
        self.num_observations = 5
        
        self.awose = AWOSE(num_states=self.num_states, num_hidden_states=self.num_hidden_states, num_observations=self.num_observations)
        self.awose.intrinsic_mc.set_transition_matrix(self._initialize_transition_matrix())

    def _initialize_transition_matrix(self):
        transition_matrix = np.array([
            [0.1, 0.4, 0.3, 0.1, 0.1],
            [0.2, 0.2, 0.4, 0.1, 0.1],
            [0.3, 0.1, 0.2, 0.3, 0.1],
            [0.4, 0.1, 0.2, 0.2, 0.1],
            [0.2, 0.3, 0.2, 0.1, 0.2]
        ])
        return transition_matrix

    def invoke_chaincode(self, func_name, args):
        response = self.client.chaincode_invoke(
            requestor='Admin',
            channel_name='mychannel',
            peer_names=['peer0.org1.example.com'],
            cc_name='basic',
            fcn=func_name,
            args=args
        )
        return response

    def query_chaincode(self, func_name, args):
        response = self.client.chaincode_query(
            requestor='Admin',
            channel_name='mychannel',
            peer_names=['peer0.org1.example.com'],
            cc_name='basic',
            fcn=func_name,
            args=args
        )
        return response

    def train_awose(self, observed_symbols, hidden_states):
        self.awose.train(observed_symbols, hidden_states)

    def simulate(self, sequence_length):
        states, observations = self.awose.generate_sequence(sequence_length)
        for state, observation in zip(states, observations):
            print(f"State: {state}, Observation: {observation}")
