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

For documentation:
* Doxygen

## Build and Install
Like all CMake projects, to build and install, simply do:
```
  mkdir build
  cd build
  cmake ..
  make && make install
```

To make the documentation, do:
```
  make doc
```

## Compile and Link
The configuration of the library can be found from pkg-config.
```
  cc `pkg-config --cflags --libs notghmm` source.c
```

## Support
Since I am the only user, the library is not well documented.
I will be glad to add documentation if requested.

Send me an email : Weipeng He <heweipeng@gmail.com>

