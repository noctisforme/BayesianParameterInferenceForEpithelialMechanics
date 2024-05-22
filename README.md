Bayesian paramter inference for epithelial mechanics
===

## Description

This is a script for parameter esitmation and model selection for epithelial mechanics, proposed in Yan and Ogita et al. 2024. 
The detailed readme will be updated soon.

## Requirement

* pymc3==3.11.4

* arviz==0.14.0

* scipy==1.9.3

* sparseqr==1.2.1

* theano==1.1.2

## Usage

1. Prepare input files from the same developmental stage in the same format as the attached sample (./Samples/*/, where * is the stage name).
2. Change the variable "filename" in BayesParameterEstimation.py to the input file from ./Samples/*/ in step 1, or the variable "stage" in HBayesParameterEstimation.py to the stage name *.
3. Run BayesianParameterEstimation.py or HBayesianParameterEstimation.py on IDE or IPython.
