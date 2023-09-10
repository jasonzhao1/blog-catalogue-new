### Context
- The goal of this repo is to perform graph learning on the `blogcatalogue` dataset

### Methodology
- The architecture/idea comes from this paper: https://arxiv.org/abs/2010.13993
- The basic idea is:
  - Use a simple machine learning model, e.g. linear/MLP
  - After that, use label propagation to perform correct and smoothing on the results of the model
 

### Current progress
- Set up the overall logic on a code level
- Need to fine tune all the steps


### Current challenge
- There is no initial feature for the nodes, i.e. all we have is the label
- Current proposed solution:
  - Solution 1: Use spectral embedding as initial feature
  - Solution 2: Use simple label propagation to generate some initial features
 
