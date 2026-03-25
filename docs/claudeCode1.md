## Predicting the scores of college basketball games

I would like to build a python implementation that runs on my laptop. I would like to implement the model described in this paper: https://discovery.ucl.ac.uk/id/eprint/16040/1/16040.pdf 

## The data

The data is in data/2023Data.csv . 

## Laptop specs

Apple M4 Pro, 24 GB memory, 14 cores; The GPU has 20 cores

## Ways to validate the model
Hold out a few games from the training and estimate a distribution over their scores and report on the probability of the actual outcome.

## Ways to estimate the model: 

This relatively new implementation looks promising https://github.com/acerbilab/pyvbmc . Please consider using it.

## Ways to use the model
Sample it to give a distribution over the spread for a game, the total of the two scores, etc. 

Since the teams are both playing “away”, I would like to estimate their scores assuming they are both playing away.
