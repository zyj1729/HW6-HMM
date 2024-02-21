![BuildStatus](https://github.com/zyj1729/HW6-HMM/actions/workflows/main.yml/badge.svg?event=push)
# Hidden Markov Model Implementation

This repository contains a Python implementation of a Hidden Markov Model (HMM), a statistical model which is especially known for its application in temporal pattern recognition such as speech, handwriting, gesture recognition, part-of-speech tagging, musical score following, partial discharges, and bioinformatics.

## Description

The implementation includes two main algorithms:

1. **Forward Algorithm**: Used to calculate the likelihood of a sequence of observed events given the model parameters. This is essential for understanding how well the model explains the observed data.

2. **Viterbi Algorithm**: Utilized to determine the most likely sequence of hidden states that results in a sequence of observed events. This is particularly useful for making predictions based on observed sequences.

## Methods

### Initialization

The `HiddenMarkovModel` class is initialized with the following parameters:

- `observation_states`: An array of observable states in the model.
- `hidden_states`: An array of hidden states in the model.
- `prior_p`: The initial probability distribution over hidden states.
- `transition_p`: The state transition probability matrix.
- `emission_p`: The emission probability matrix.

### Forward Algorithm

The `forward` method implements the Forward algorithm to compute the log likelihood of an observed sequence under the model. It initializes the forward probabilities, iteratively computes the probabilities for each state at each time step, and finally aggregates the probabilities to get the total likelihood of the observation sequence.

### Viterbi Algorithm

The `viterbi` method implements the Viterbi algorithm, which computes the most likely sequence of hidden states given the observed sequence. It initializes the Viterbi table and backtrace table, iteratively updates these tables to keep track of the maximum probabilities and their corresponding states, and performs a traceback from the final state to construct the most likely sequence of hidden states.

## Testing

The repository also includes test cases for validating the implementation of the Forward and Viterbi algorithms using small and full weather datasets. These tests ensure the correctness of the algorithm implementations by comparing the results against known expected outcomes and a reference implementation.

## Usage

To use this HMM implementation, simply import the `HiddenMarkovModel` class from the module and instantiate it with your model parameters. Then, you can use the `forward` method to calculate the likelihood of an observation sequence or the `viterbi` method to find the most likely sequence of hidden states.

```python
from hmm_implementation import HiddenMarkovModel

# Initialize the HMM with your parameters
hmm = HiddenMarkovModel(observation_states, hidden_states, prior_p, transition_p, emission_p)

# Calculate the likelihood of an observation sequence
likelihood = hmm.forward(observation_sequence)

# Find the most likely sequence of hidden states
hidden_states_sequence = hmm.viterbi(observation_sequence)
