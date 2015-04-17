notGHMM
=======

This is a C library for Hidden Markov Model (HMM). 

## Implemented Functions
* HMM with continuous observation space (Gaussian mixtures).
  * Probability Computation (forward/backward procedure)
  * Viterbi Algorithm
  * Initial Parameter Estimation using k-means or randomly
  * Parameter Estimation (Baum-Welch Algorithm) with parallel acceleration
  * Random sequence generation

## Dependent Libraries
* GSL
* Flann
* OpenMP

Build system:
* CMake

## Build and Install
Simply do:
```
  mkdir build
  cd build
  cmake ..
  make
  make install
```

## Support
Since I am the only user, the library is not well documented.
I will be glad to add documentation if requested.

Send me an email : Weipeng He <heweipeng@gmail.com>

