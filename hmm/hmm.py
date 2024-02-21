import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        Implements the forward algorithm to compute the likelihood of an observed sequence.

        Args:
            input_observation_states (np.ndarray): Sequence of observed states.

        Returns:
            float: Log likelihood of the observed sequence under the model.
        """        
        
        # Initialize the forward probability matrix with zeros
        self.forward_p = np.zeros([len(self.hidden_states), len(input_observation_states)])
        # Convert the sequence of observed states to their corresponding indices
        obs_sequence = [self.observation_states_dict[i] for i in input_observation_states]
        # Initialize the first column of the forward matrix
        for j in range(len(self.hidden_states)):
            self.forward_p[j][0] = np.log(self.prior_p[j]) + np.log(self.emission_p[j][obs_sequence[0]])
       
        # Iteratively fill in the forward matrix
        for i in range(1, len(obs_sequence)):
            for j in range(len(self.hidden_states)):
                self.forward_p[j][i] = np.logaddexp.reduce(
                    [self.forward_p[k][i - 1] + np.log(self.transition_p[k][j]) for k in range(len(self.hidden_states))]
                ) + np.log(self.emission_p[j, obs_sequence[i]])
    
        # Compute the log likelihood of the observed sequence
        return np.logaddexp.reduce(self.forward_p[:, -1])

    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        Implements the Viterbi algorithm to find the most likely sequence of hidden states given an observed sequence.

        Args:
            decode_observation_states (np.ndarray): Sequence of observed states.

        Returns:
            list: The most likely sequence of hidden states.
        """        
        
        # Initialize the Viterbi and backtrace matrices
        n_obs = len(decode_observation_states)
        n_states = len(self.hidden_states)
        viterbi_table = np.zeros((n_states, n_obs))
        backtrace = np.zeros((n_states, n_obs), dtype=int)

        # Convert the sequence of observed states to their corresponding indices
        obs_indices = [self.observation_states_dict[obs] for obs in decode_observation_states]
        
        # Initialize the first column of the Viterbi matrix
        for s in range(n_states):
            viterbi_table[s, 0] = np.log(self.prior_p[s]) + np.log(self.emission_p[s, obs_indices[0]])

        # Fill in the Viterbi matrix and keep track of back pointers
        for t in range(1, n_obs):
            for s in range(n_states):
                max_prob = -np.inf
                for ss in range(n_states):
                    prob = viterbi_table[ss, t-1] + np.log(self.transition_p[ss, s]) + np.log(self.emission_p[s, obs_indices[t]])
                    if prob > max_prob:
                        max_prob = prob
                        max_state = ss
                viterbi_table[s, t] = max_prob
                backtrace[s, t] = max_state
        
        # Traceback to find the most likely sequence of hidden states
        last_state = np.argmax(viterbi_table[:, -1])
        best_path = [last_state]
        for t in range(n_obs - 1, 0, -1):
            last_state = backtrace[last_state, t]
            best_path.insert(0, last_state)

        # Return the most likely sequence of hidden states
        return [self.hidden_states[state] for state in best_path]