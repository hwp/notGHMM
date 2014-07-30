// utils.h
// Utility functions
//
// Author : Weipeng He <heweipeng@gmail.com>
// Copyright (c) 2014, All rights reserved.

#ifndef UTILS_H_
#define UTILS_H_

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>

/**
 * Multivariate Gaussian distribution.
 */
typedef struct {
  /**
   * Number of dimension.
   */
  size_t dim;

  /**
   * Mean of the distribution.
   */
  gsl_vector* mean;

  /**
   * Covariance matrix.
   */
  gsl_matrix* cov;
} gaussian_t;

/**
 * Gaussian mixture model.
 */
typedef struct {
  /**
   * Number of dimension.
   */
  size_t dim;

  /**
   * Number of mixture components.
   */
  size_t k;

  /**
   * Mixture weight.
   */
  gsl_vector* weight;

  /**
   * Components.
   */
  gaussian_t** comp;
} gmm_t;

/**
 * Allocate memory for a Gaussian distribution.
 * dim is set.
 * mean and cov are allocated but not initialized.
 *
 * @param dim number of dimension.
 *
 * @return pointer to the allocated space.
 *         NULL, if error occurs.
 */
gaussian_t* gaussian_alloc(size_t dim);

/**
 * Free memory.
 *
 * @param dist a pointer which is returned by previous call
 *    of gaussian_alloc. If dist is NULL, no operation is
 *    performed.
 */
void gaussian_free(gaussian_t* dist);

/**
 * Copy a Gaussian distribution into another.
 * The two models must have the same dim.
 *
 * @param dest the distribution to be copied to.
 * @param src the distribution to be copied from.
 */
void gaussian_memcpy(gaussian_t* dest, const gaussian_t* src);

/**
 * Allocate memory for a Gaussian Mixture model.
 * dim and k are set.
 * weight and comp are allocated but not initialized.
 *
 * @param dim number of dimension.
 * @param k number of components.
 *
 * @return pointer to the allocated space.
 *         NULL, if error occurs.
 */
gmm_t* gmm_alloc(size_t dim, size_t k);

/**
 * Free memory.
 *
 * @param gmm a pointer which is returned by previous call
 *    of gmm_alloc. If gmm is NULL, no operation is
 *    performed.
 */
void gmm_free(gmm_t* gmm);

/**
 * Copy a GMM into another. The two models must have
 * the same k and dim;
 *
 * @param dest the model to be copied to.
 * @param src the model to be copied from.
 */
void gmm_memcpy(gmm_t* dest, const gmm_t* src);

/**
 * Generate a random sample from a discrete distribution.
 *
 * @param rng the GSL random number generator.
 * @param dist the distribution.
 *   Each element must be non-negative. The sum must be 1.
 *   In implementation, the value of the last element is
 *   ignored. Instead, it is considered as 1 - sum of
 *   previous elements.
 */
size_t discrete_gen(const gsl_rng *rng, const gsl_vector* dist);

/**
 * Probability density function of multivariate Gaussian
 * distribution.
 *
 * @param dist the Gaussian distribution.
 * @param x variable.
 *
 * @return pdf
 */
double gaussian_pdf(const gaussian_t* dist,
    const gsl_vector* x);

/**
 * Generate a random sample consistent to a Gaussian
 * distribution.
 *
 * @param rng the GSL random number generator.
 * @param dist the Guassian distribution.
 * @param[out] result the generated sample is saved in
 *   result, which is allocated before calling.
 */
void gaussian_gen(const gsl_rng* rng, const gaussian_t* dist,
    gsl_vector* result);

/**
 * Probability density function of Gaussian mixture model.
 *
 * @param gmm the Gaussian mixture model.
 * @param x variable.
 *
 * @return pdf
 */
double gmm_pdf(const gmm_t* gmm, const gsl_vector* x);

/**
 * Generate a random sample consistent to a Gaussian
 * mixture model.
 *
 * @param rng the GSL random number generator.
 * @param gmm the Guassian mixture model.
 * @param[out] result the generated sample is saved in
 *   result, which is allocated before calling.
 */
void gmm_gen(const gsl_rng* rng, const gmm_t* gmm,
    gsl_vector* result);

/**
 * Calcuate the sum of logarithm variables, e.g.
 *     @f[ \log\sum_i\exp(v_i) @f]
 * To avoid underflow (of at least one element),
 * it is calculated as:
 *     @f[ -M + \log\sum_i\exp(v_i + M) @f]
 * where @f$ M = -max_i\{v_i\} @f$
 *
 * @param v the vector to be sumed.
 * 
 * @return the log sum of exponent of all elements.
 */
double log_sum_exp(const gsl_vector* v);

/**
 * Find the maximum value and its index from a vector.
 *
 * @param v the vector.
 * @param[out] index the index of the maximum element. 
 *   If there are more than one maximum elements, the one 
 *   with the lowest index will be returned.
 *   If index is NULL, the result will not be saved.
 *
 * @return the maximum value.
 */
double max_index(gsl_vector* v, size_t* index);

#endif  // UTILS_H_

