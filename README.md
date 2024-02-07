# HW6-HMM

In this assignment, you'll implement the Viterbi Algorithm (dynamic programming). 


# Assignment

## Overview 

The goal of this assignment is to implement the Viterbi Algorithm for Hidden Markov Models (HMMs)

For a helpful refresher on HMMs and the Viterbi Algorithm you can check out the resources [here](https://web.stanford.edu/~jurafsky/slp3/A.pdf), 
[here](https://towardsdatascience.com/markov-and-hidden-markov-model-3eec42298d75), and [here](https://pieriantraining.com/viterbi-algorithm-implementation-in-python-a-practical-guide/). 





### Data 
We have provided two HMM models (small_weather_hmm.npz and full_weather_hmm.npz) which explore the relationships between observable weather phenomenon and the temperature outside. Both include arrays with the hidden and observation states, along with their corresponding prior, transition, and emission probabilities. 

In addition, for both datasets, we provide input observation sequences and the solution for their best hidden state sequences. In your unit tests, please test that your Viterbi implementation returns the correct hidden state sequence for both observation sequences. 

Within your code, consider the scope of the inputs and how the different parameters of the input data could break the bounds of your implementation.
  * Do your model probabilites add up to the correct values? Is scaling required?
  * How will your model handle zero-probability transitions? 
  * Test for edge cases, such as looking at the 
  * Include at least 2 possible edge cases. 



## Tasks

[TODO] Complete the ViterbiAlgorithm class with your implementation of the algorithm

  [x] complete the `viterbi` method  

[TODO] Unit Testing  
  [x] Edge cases in HiddenMarkovModel Class 
  [x] Edge cases in `ViterbiAlgorithm``
  [x] Ensure functionality on mini and full weather dataset 

[TODO] Packaging 
  [x] Update README with description of your methods 
  [x] pip installable module (optional)
  [x] github actions (install + pytest) (optional)


## Completing the Assignment 
Push your code to GitHub with passing unit tests, and submit a link to your repository [here](insert link here)

### Grading 

* Viterbi implementation (6 points)
    * Viterbi forward is correct (3)
    * Backtracing (1)
    *  Output is correct on small weather dataset (1)
    * Output is correct on full weather dataset (1)

* Unit Tests (3 points)
    * Unit tests for 
    * Edge cases (1)

* Style (1 point)
    * Readable code and updated README with a description of your methods 

* Extra credit (0.5 points)
    * Pip installable and Github actions (0.5)