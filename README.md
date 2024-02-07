# HW6-HMM

In this assignment, you'll implement the Viterbi Algorithm (dynamic programming). 


# Assignment

## Overview 

The goal of this assignment is to implement the Viterbi Algorithm for Hidden Markov Models (HMMs)

For a helpful refresher on HMMs and the Viterbi Algorithm you can check out the resources [here](https://web.stanford.edu/~jurafsky/slp3/A.pdf), 
[here](https://towardsdatascience.com/markov-and-hidden-markov-model-3eec42298d75), and [here](https://pieriantraining.com/viterbi-algorithm-implementation-in-python-a-practical-guide/). 





## Tasks and Data 
Please complete the `viterbi` function in the HiddenMarkovModel class. 

We have provided two HMM models (mini_weather_hmm.npz and full_weather_hmm.npz) which explore the relationships between observable weather phenomenon and the temperature outside. Both include arrays with the hidden and observation states, along with their corresponding prior, transition, and emission probabilities. Start with the mini_weather_hmm model for testing and debugging. 

For both datasets, we provide input observation sequences and the solution for their best hidden state sequences. 

Create an HMM class instance for both models and test that your Viterbi implementation returns the correct hidden state sequence for each of the observation sequences.

Finally, please update your README with a brief description of your methods. 

Within your code, consider the scope of the inputs and how the different parameters of the input data could break the bounds of your implementation.
  * Do your model probabilites add up to the correct values? Is scaling required?
  * How will your model handle zero-probability transitions? 
  * Are the inputs in compatible shapes/sizes which each other? 
  * Ensure that your code accomodates at least 2 of these possible edge cases. 



## Task List

[TODO] Implement the Viterbi Algorithm
  [ ] complete the `viterbi` function in the HiddenMarkovModelClass

[TODO] Unit Testing  
  [ ] Ensure functionality on mini and full weather dataset 
  [ ] Account for edge cases 

[TODO] Packaging 
  [ ] Update README with description of your methods 
  [ ] pip installable module (optional)
  [ ] github actions (install + pytest) (optional)


## Completing the Assignment 
Push your code to GitHub with passing unit tests, and submit a link to your repository [here](insert link here)

### Grading 

* Viterbi implementation (6 points)
    * Viterbi forward is correct (3)
    * Backtracing is correct (1)
    *  Output is correct on small weather dataset (1)
    * Output is correct on full weather dataset (1)

* Unit Tests (3 points)
    * Mini model unit test (1)
    * Full model unit test (1)
    * Edge cases (1)

* Style (1 point)
    * Readable code and updated README with a description of your methods 

* Extra credit (0.5 points)
    * Pip installable and Github actions (0.5)