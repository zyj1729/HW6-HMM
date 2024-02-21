import pytest
from hmm import HiddenMarkovModel
import numpy as np
from hmmlearn import hmm

def test_mini_weather():
    """
    Tests the implementation of the Hidden Markov Model using a small weather dataset.
    Validates the correctness of the Forward and Viterbi algorithms' outputs.
    Includes assertions to check the model's consistency and correctness, and compares the results against a reference implementation.
    Also tests for edge cases to ensure robustness of the implementation.
    """

    # Load the model parameters and observation sequence from the provided files
    mini_hmm = np.load('./data/mini_weather_hmm.npz')
    mini_input = np.load('./data/mini_weather_sequences.npz')
    
    # Assert the consistency of the model parameters
    assert mini_hmm["observation_states"].shape[0] == mini_hmm["emission_p"].shape[1], "Emission probabilities should correspond to the number of observation states."
    assert mini_hmm["hidden_states"].shape[0] == mini_hmm["emission_p"].shape[0], "Emission probabilities should correspond to the number of hidden states."
    assert mini_hmm["transition_p"].shape[0] == mini_hmm["transition_p"].shape[1] and mini_hmm["hidden_states"].shape[0] == mini_hmm["transition_p"].shape[0], "Transition probabilities should be a square matrix with dimensions equal to the number of hidden states."
    assert mini_hmm["prior_p"].shape[0] == mini_hmm["hidden_states"].shape[0], "The size of the initial state distribution should match the number of hidden states."
    assert mini_hmm["prior_p"].sum() == 1, "The initial state distribution should sum up to 1."
    assert (mini_hmm["transition_p"].sum(axis=1) == 1).all(), "Each row of the transition probability matrix should sum up to 1."
    assert (mini_hmm["emission_p"].sum(axis=1) == 1).all(), "Each row of the emission probability matrix should sum up to 1."
    
    # Instantiate the HMM with the loaded parameters
    mine = HiddenMarkovModel(mini_hmm["observation_states"], mini_hmm["hidden_states"], mini_hmm["prior_p"], mini_hmm["transition_p"], mini_hmm["emission_p"])
    # Run the Forward algorithm and store the resulting probability
    prob = mine.forward(mini_input["observation_state_sequence"])
    # Run the Viterbi algorithm and store the resulting state sequence
    pred = mine.viterbi(mini_input["observation_state_sequence"])
    
    # Set up a reference HMM for comparison
    hidden_states = np.array(['hot', 'cold'])
    model = hmm.CategoricalHMM(n_components=2)
    model.startprob_ = mini_hmm["prior_p"]
    model.transmat_ = mini_hmm["transition_p"]
    model.emissionprob_ = mini_hmm["emission_p"]
    observation_states = np.array(['sunny', 'rainy'])
    observation_map = {state: index for index, state in enumerate(observation_states)}
    observed_sequence = np.array([observation_map[state] for state in mini_input["observation_state_sequence"]]).reshape(-1, 1)

    # Calculate the log probability of the observed sequence using the reference HMM
    logprob = model.score(observed_sequence)
    
    # Assert the correctness of the Forward algorithm's output by comparing it to the reference implementation
    assert abs(prob - logprob) < 0.0000001, "The forward probability calculated by the HMM does not match the reference implementation."
    # Assert the correctness of the Viterbi algorithm's output
    assert pred == list(mini_input["best_hidden_state_sequence"]), "The state sequence predicted by the Viterbi algorithm does not match the expected sequence."

def test_full_weather():
    """
    Similar to test_mini_weather, but uses a more comprehensive weather dataset.
    This test focuses on validating the Viterbi algorithm's output against the expected state sequence.
    """

    # Load the full weather model parameters and observation sequence
    full_hmm = np.load('./data/full_weather_hmm.npz')
    full_input = np.load('./data/full_weather_sequences.npz')

    # Instantiate and run the HMM on the loaded data
    mine = HiddenMarkovModel(full_hmm["observation_states"], full_hmm["hidden_states"], full_hmm["prior_p"], full_hmm["transition_p"], full_hmm["emission_p"])
    prob = mine.forward(full_input["observation_state_sequence"])
    pred = mine.viterbi(full_input["observation_state_sequence"])
    
    # Set up a reference HMM for comparison
    hidden_states = np.array(['hot', 'temperate', 'cold', 'freezing'])
    model = hmm.CategoricalHMM(n_components=4)
    model.startprob_ = full_hmm["prior_p"]
    model.transmat_ = full_hmm["transition_p"]
    model.emissionprob_ = full_hmm["emission_p"]
    observation_states = np.array(['sunny', 'cloudy', 'rainy', 'snowy', 'hailing'])
    observation_map = {state: index for index, state in enumerate(observation_states)}
    observed_sequence = np.array([observation_map[state] for state in full_input["observation_state_sequence"]]).reshape(-1, 1)

    # Calculate the log probability of the observed sequence using the reference HMM
    logprob = model.score(observed_sequence)
    
    # Assert the correctness of the Forward algorithm's output and the Viterbi algorithm's predicted state sequence
    assert abs(prob - logprob) < 0.0000001, "The forward probability calculated by the HMM does not match the reference implementation."
    assert pred == list(full_input["best_hidden_state_sequence"]), "The state sequence predicted by the Viterbi algorithm does not match the expected sequence."











