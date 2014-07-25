// utils.h
// Utility functions
//
// Author : Weipeng He <heweipeng@gmail.com>
// Copyright (c) 2014, All rights reserved.

#ifndef UTILS_H_
#define UTILS_H_

#include <gsl/gsl_rng.h>

/**
 * Generate a random sample consistent to a multivariate 
 * Gaussian distribution.
 *
 * @param rng the GSL random number generator.
 * @param dim number of dimension.
 * @param mu mean of the distribution, vector of size dim.
 * @param sigma covariance matrix, size of dim * dim
 * @param[out] result the generated sample is saved in result,
 *   of which the space should be allocated before calling.
 */
void gen_gaussian(gsl_rng* rng, int dim, double* mu,
    double* sigma, double* result);

/**
 * Generate a random sample from a discrete distribution.
 *
 * @param rng the GSL random number generator.
 * @param dist the distribution, a vector of size n.
 *   Each element should be non-negative. The sum should be 1. 
 *   In implementation, the value of the last element is
 *   ignored. Instead, it is considered as 1 - sum of 
 *   previous elements.
 * @param n number of elements.
 */
int gen_discrete(gsl_rng *rng, double* dist, int n);

#endif  // UTILS_H_

