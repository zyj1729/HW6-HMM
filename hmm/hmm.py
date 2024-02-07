import numpy as np
class HiddenMarkovModel:
    """_summary_
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
        self.observation_states_dict = {observation_state: index \
                                  for index, observation_state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: hidden_state \
                                   for index, hidden_state in enumerate(list(self.hidden_states))}
        

        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p




    def forward(self, observation_states)-> float:
        """
        TODO 
        Args:


        Returns:
            float: maximum likelihood value for the given pair of hidden sequence 
        """

        #Step 1: Init




        #Step 2:



        #Step 3: 




        return total_likelihood




    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO
        THIS FUNCTION  ... 


        Args:
            decode_observation_states (np.ndarray): observation states to decode 

        Returns:
            list: most likely list of hidden states that generated the sequence observed states
        """        
        
        # Initialize path (i.e., np.arrays) to store the hidden sequence states returning the maximum probability
        path = np.zeros((len(decode_observation_states), 
                         len(self.hidden_states)))
        path[0,:] = [hidden_state_index for hidden_state_index in range(len(self.hidden_states))]

        best_path = np.zeros((len(decode_observation_states), 
                         len(self.hmm_object.hidden_states)))        
        
        # Step 1: Init
        # Probability of prior and emission probabilities 
  

        #Step 2: recursion 
        for trellis_node in range(1, len(decode_observation_states)):
            pass

           

        
        #Step 3: Backtrace
        best_hidden_state_sequence = []

        return best_hidden_state_sequence