/* NotGHMM
 * Copyright (c) 2014 Weipeng He <heweipeng@gmail.com>
 * utils.c : Utility functions.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

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

  /**
   * Diagonal of the covariance matrix.
   * @note either cov or diag is NULL.
   *   if diag is not null, then the rest of the matrix is ignored,
   *   that is, each dimension is considered to be independent of each other.
   */
  gsl_vector* diag;
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
 * mean and cov (or diag) are allocated but not initialized.
 *
 * @param dim number of dimension.
 * @param cov_diag if non-zero, the covariance matrix is diagonal.
 *
 * @return pointer to the allocated space.
 *         NULL, if error occurs.
 */
gaussian_t* gaussian_alloc(size_t dim, int cov_diag);

/**
 * Free memory.
 *
 * @param dist a pointer which is returned by previous call
 *    of gaussian_alloc. If dist is NULL, no operation is
 *    performed.
 */
void gaussian_free(gaussian_t* dist);

/**
 * @return 1 if the distribution has diagonal covariance matrix,
 *   0 otherwise.
 */
int gaussian_isdiagonal(const gaussian_t* dist);

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
 * @param cov_diag if non-zero, the covariance matrices of the 
 *    Gaussian distributions are diagonal.
 *
 * @return pointer to the allocated space.
 *         NULL, if error occurs.
 */
gmm_t* gmm_alloc(size_t dim, size_t k, int cov_diag);

/**
 * Free memory.
 *
 * @param gmm a pointer which is returned by previous call
 *    of gmm_alloc. If gmm is NULL, no operation is
 *    performed.
 */
void gmm_free(gmm_t* gmm);

/**
 * Check if a GMM is valid.
 *
 * @return 1 if valid, 0 if invalid.
 */
int gmm_valid(const gmm_t* gmm);

/**
 * Copy a GMM into another. The two models must have
 * the same k and dim;
 *
 * @param dest the model to be copied to.
 * @param src the model to be copied from.
 */
void gmm_memcpy(gmm_t* dest, const gmm_t* src);

/**
 * Check if a discrete distribution is valid.
 *
 * @return 1 if valid, 0 if invalid.
 */
int discrete_valid(const gsl_vector* dist);

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
 * @return log pdf
 */
double gaussian_pdf_log(const gaussian_t* dist,
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
 * @return log pdf
 */
double gmm_pdf_log(const gmm_t* gmm, const gsl_vector* x);

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
 * math function call with exception report.
 */
double math_func_fe_except(double (*func)(double),
    double x, const char* func_name, const char* file,
    unsigned int line);

#define DEBUG_LOG(x) (x == 0.0 ? -HUGE_VAL : math_func_fe_except(log, x, "log", __FILE__, __LINE__))
#define DEBUG_EXP(x) math_func_fe_except(exp, x, "exp", __FILE__, __LINE__)
#define DEBUG_SQRT(x) math_func_fe_except(sqrt, x, "sqrt", __FILE__, __LINE__)

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

/**
 * Print a vector in human readable format.
 *
 * @param stream output stream.
 * @param v the vector.
 */
void vector_fprint(FILE* stream, const gsl_vector* v);

/**
 * Read a vector from text in the format that is
 * generated by vector_fprint().
 *
 * @param stream input stream.
 * @param[in,out] v the vector to be stored. It must be
 *   preallocated with the correct size.
 */
void vector_fscan(FILE* stream, gsl_vector* v);

/**
 * Print a matrix in human readable format.
 *
 * @param stream output stream.
 * @param m the matrix.
 */
void matrix_fprint(FILE* stream, const gsl_matrix* m);

/**
 * Read a matrix from text in the format that is
 * generated by matrix_fprint().
 *
 * @param stream input stream.
 * @param[in,out] m the matrix to be stored. It must be
 *   preallocated with the correct size.
 */
void matrix_fscan(FILE* stream, gsl_matrix* m);

/**
 * Cluster the data using k-means algorithm.
 * 
 * @param data the data to be clustered.
 * @param size number of data.
 * @param k number of clusters.
 * @param[out] index cluster indices of the data.
 *   It must be preallocated with the correct size.
 *   set null if no output expected.
 * @param[out] center cluster centers.
 *   It must be preallocated with the correct size.
 *   set null if no output expected.
 */
void kmeans_cluster(gsl_vector** data, size_t size,
    size_t k, size_t* index, gsl_vector** center);

#endif  // UTILS_H_

