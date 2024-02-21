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
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables
        self.forward_p = np.zeros([self.prior_p.shape[0], input_observation_states.shape[0]])
        obs_sequence = [np.where(self.observation_states == i)[0][0] for i in input_observation_states]
        for j in range(self.prior_p.shape[0]):
            self.forward_p[j][0] = np.log(self.prior_p[j]) + np.log(self.emission_p[j][obs_sequence[0]])
       
        # Step 2. Calculate probabilities
        for i in range(1, len(obs_sequence)):
            for j in range(self.prior_p.shape[0]):
                self.forward_p[j][i] = np.logaddexp.reduce(
                    [self.forward_p[k][i - 1] + np.log(self.transition_p[k][j]) for k in range(self.prior_p.shape[0])]
                ) + np.log(self.emission_p[j, obs_sequence[i]])
    
        # Step 3. Return final probability
        return np.logaddexp.reduce(self.forward_p[:, -1])
        


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        #store probabilities of hidden state at each step
        n_obs = len(decode_observation_states)
        n_states = len(self.hidden_states)
        viterbi_table = np.zeros((n_states, n_obs))
        #store best path for traceback
        best_path = np.zeros(len(decode_observation_states))    
        
        backtrace = np.zeros((n_states, n_obs), dtype = int)
        obs_indices = [self.observation_states_dict[obs] for obs in decode_observation_states]
        
        # Step 2. Calculate probability
        for s in range(n_states):
            viterbi_table[s, 0] = np.log(self.prior_p[s]) + np.log(self.emission_p[s, obs_indices[0]])
        for t in range(1, n_obs):
            for s in range(n_states):
                max_prob = -np.inf
                max_state = 0
                for ss in range(n_states):
                    prob = viterbi_table[ss, t-1] + np.log(self.transition_p[ss, s]) + np.log(self.emission_p[s, obs_indices[t]])
                    if prob > max_prob:
                        max_prob = prob
                        max_state = ss
                viterbi_table[s, t] = max_prob
                backtrace[s, t] = max_state
        
        # Step 3. Traceback 
        last_state = np.argmax(viterbi_table[:, -1])
        best_path = [last_state]
        for t in range(n_obs - 1, 0, -1):
            last_state = backtrace[last_state, t]
            best_path.insert(0, last_state)

        # Step 4. Return best hidden state sequence 
        return [self.hidden_states[state] for state in best_path]