import pytest
from hmm import HiddenMarkovModel
import numpy as np
import hmmlearn as hm

def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')
    
    assert mini_hmm["observation_states"].shape[0] == mini_hmm["emission_p"].shape[1], "emission_p dimension should match the number of observation states"
    assert mini_hmm["hidden_states"].shape[0] == mini_hmm["emission_p"].shape[0], "emission_p dimension should match the number of hidden states"
    assert mini_hmm["transition_p"].shape[0] == mini_hmm["transition_p"].shape[1] and mini_hmm["hidden_states"].shape[0] == mini_hmm["transition_p"].shape[0], "transition_p should have the dimension n x n, where n is the number of hidden states"
    assert mini_hmm["prior_p"].shape[0] == mini_hmm["hidden_states"].shape[0], "prior_p size should be the number of hidden states"
    assert mini_hmm["prior_p"].sum() == 1, "prior_p should sum up to 1"
    assert (mini_hmm["transition_p"].sum(axis = 1) == 1).all() == True, "Transition probability should sum up to 1 for a hidden state"
    assert (mini_hmm["emission_p"].sum(axis = 1) == 1).all() == True, "Emission probability should sum up to 1 for a hidden state"
    
    mine = HiddenMarkovModel(mini_hmm["observation_states"], mini_hmm["hidden_states"], mini_hmm["prior_p"], mini_hmm["transition_p"], mini_hmm["emission_p"])
    prob = mine.forward(mini_input["observation_state_sequence"])
    pred = mine.viterbi(mini_input["observation_state_sequence"])
    
    hidden_states = np.array(['hot', 'cold'])
    model = hm.CategoricalHMM(n_components=2)
    model.startprob_ = mini_hmm["prior_p"]
    model.transmat_ = mini_hmm["transition_p"]
    model.emissionprob_ = mini_hmm["emission_p"]
    observation_states = np.array(['sunny', 'rainy'])
    observation_map = {state: index for index, state in enumerate(observation_states)}
    observed_sequence = np.array([observation_map[state] for state in full_input["observation_state_sequence"]]).reshape(-1, 1)

    logprob = model.score(observed_sequence)
    
    assert abs(prob - logprob) < 0.0000001, "The forward probability is incorrect"
    assert pred == mini_input["best_hidden_state_sequence"], "The predicted hidden state sequence is incorrect"



def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """
    
    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')

    mine = HiddenMarkovModel(mini_full["observation_states"], mini_full["hidden_states"], mini_full["prior_p"], mini_full["transition_p"], mini_full["emission_p"])
    prob = mine.forward(full_input["observation_state_sequence"])
    pred = mine.viterbi(full_input["observation_state_sequence"])
    
    hidden_states = np.array(['hot', 'temperate', 'cold', 'freezing'])
    model = hmm.CategoricalHMM(n_components=4)
    model.startprob_ = full_hmm["prior_p"]
    model.transmat_ = full_hmm["transition_p"]
    model.emissionprob_ = full_hmm["emission_p"]
    observation_states = np.array(['sunny', 'cloudy', 'rainy', 'snowy', 'hailing'])
    observation_map = {state: index for index, state in enumerate(observation_states)}
    observed_sequence = np.array([observation_map[state] for state in full_input["observation_state_sequence"]]).reshape(-1, 1)

    logprob = model.score(observed_sequence)
    
    assert abs(prob - logprob) < 0.0000001, "The forward probability is incorrect"
    assert pred == full_input["best_hidden_state_sequence"], "The predicted hidden state sequence is incorrect"














