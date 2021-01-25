# Tranductive Dropout

implementation code for the paper:

## [Unlabelled Data Improves Bayesian Uncertainty Calibration under Covariate Shift](https://arxiv.org/abs/2006.14988)

### Alex J. Chan, Ahmed M. Alaa, Zhaozhi Qian, and Mihaela van der Schaar

### International Conference on Machine Learning (ICML) 2020

Based on the code of Alex Chan (ajc340@cam.ac.uk)


An implementation of a transductive dropout network class for 2-dimensional response-surface can be found in models_multivariate_response_surface.py
To run an example model run main2.py

An example for 2-dimensional regression problem and an example for a 1-dimensional transductive dropout network under a Gaussian mixture in the source domain is provided in mixed_distribution_and_2d_response_surface_testing.ipynb


Dependencies:
Autograd, 
matplotlib, 
tqdm
