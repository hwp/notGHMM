// notghmm.h
// Header file for basic data structure and functions.
//
// Author : Weipeng He <heweipeng@gmail.com>
// Copyright (c) 2014, All rights reserved.

#ifndef NOTGHMM_H_
#define NOTGHMM_H_

#include "utils.h"

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

/**
 * Sequence of observed data.
 */
typedef struct {
  /**
   * Length of the sequence.
   */
  size_t size;

  /**
   * Number of dimension of observed data.
   */
  size_t dim;

  /**
   * The observed data.
   * data[i] is the observed data (a vector of size dim)
   * at time i.
   */
  gsl_vector** data;
} seq_t;

/**
 * A HMM-GMM model.
 */
typedef struct {
  /**
   * Number of hidden states.
   */
  size_t n;

  /**
   * Number of components of each mixture.
   */
  size_t k;

  /**
   * Number of dimension of observed data.
   */
  size_t dim;

  /**
   * Probability of initial state. Its size is n.
   */
  gsl_vector* pi;

  /**
   * Transition probability. Its size is n * n.
   * a[i,j] = P(q_{t+1}=j|q_t=i)
   */
  gsl_matrix* a;

  /**
   * GMMs of each states.
   */
  gmm_t** states;
} hmmgmm_t;

/**
 * Allocate memory for a sequence.
 * size and dim are set according to the arguments.
 * data is allocated but not initialized.
 *
 * @param size length of the sequence.
 * @param dim number of dimensions.
 *
 * @return pointer to the allocated space.
 *         NULL, if error occurs.
 */
seq_t* seq_alloc(size_t size, size_t dim);

/**
 * Free memory for the sequence.
 */
void seq_free(seq_t* seq);

/**
 * Allocate memory for a HMM-GMM model.
 * n, k and dim are set according to the arguments.
 * pi, a and dist are allocated but not initialized.
 *
 * @param n number of hidden states.
 * @param k number of components.
 * @param dim number of dimension.
 *
 * @return pointer to the allocated space.
 *         NULL, if error occurs.
 */
hmmgmm_t* hmmgmm_alloc(size_t n, size_t k, size_t dim);

/**
 * Free memory for the model.
 */
void hmmgmm_free(hmmgmm_t* model);

// TODO : some initialization functions.

/**
 * Generate a sequence of observed data according to a HMM model.
 *
 * @param model the HMM model.
 * @param size length of the sequence.
 *
 * @return the generated sequence.
 *
 * @warning the returned sequence should be freed after use.
 */
seq_t* seq_gen(const hmmgmm_t* model, size_t size);

/**
 * Forward procedure.
 * Calculate all forward variable defined as:
 *     @f[ \alpha_t(i) = P(o_1, ..., o_t, q_t = i) @f]
 * recursively using:
 *     @f[ \alpha_1(i) = \pi_i b_i(o_1) @f]
 *     @f[ \alpha_{t+1}(j) = (\sum_{i=1}^N \alpha_t(i)a_{ij})b_j(o_{t+1}) @f]
 *
 * @param model the HMM model.
 * @param seq the observed sequence.
 * @param[out] alpha the result forward variable, which is
 *   a matrix of size seq->size * model->n.
 *   The matrix should be allocated before calling.
 *
 * @warning underflow issue is not considered in this function.
 *   For implementation that take scaling into account,
 *   use forward_proc_log().
 */
void forward_proc(const hmmgmm_t* model, const seq_t* seq,
    gsl_matrix* alpha);

/**
 * Forward procedure, which take scaling into account.
 * Calculate the logarithm of all forward variables, aka @f$ \alpha @f$.
 * The calculation use the following formulas:
 *     @f[ \log\alpha_1(i) = \log\pi_i + \log b_i(o_1) @f]
 *     @f[ \log\alpha_{t+1}(j) = \log\sum_i
 *       \exp(\log a_{ij} + \log\alpha_t(i))
 *       + \log b_j(o_{t+1}) @f]
 * here the log sum exp is calculated by log_sum_exp() to 
 * avoid underflow.
 * 
 * @param model the HMM model.
 * @param seq the observed sequence.
 * @param[out] logalpha the logartithm of forward variable,
 *   which is a matrix of size seq->size * model->n.
 *   The matrix should be allocated before calling.
 */
void forward_proc_log(const hmmgmm_t* model,
    const seq_t* seq, gsl_matrix* logalpha);

/**
 * Backward procedure.
 * Calculate all backward variable defined as:
 *     @f[ \beta_t(i) = P(o_{t+1}, ..., o_T | q_t = i) @f]
 * recursively using:
 *     @f[ \beta_T(i) = 1 @f]
 *     @f[ \beta_t(i) = \sum_{j=1}^N 
 *       a_{ij}b_j(o_{t+1})\beta_{t+1}(j) @f]
 *
 * @param model the HMM model.
 * @param seq the observed sequence.
 * @param[out] beta the result back variable, which is
 *   a matrix of size seq->size * model->n.
 *   The matrix should be allocated before calling.
 *
 * @warning underflow issue is not considered in this function.
 *   For implementation that take scaling into account,
 *   use backward_proc_log().
 */
void backward_proc(const hmmgmm_t* model, const seq_t* seq,
    gsl_matrix* beta);

/**
 * Back procedure, which take scaling into account.
 * Calculate the logarithm of all back variables, aka @f$ \beta @f$.
 * The calculation use the following formulas:
 *     @f[ \log \beta_T(i) = 0 @f]
 *     @f[ \log \beta_{t}(i) = \log\sum_j \exp(\log a_{ij}
 *       + \log b_j(o_{t+1}) + \log \beta_{t+1}(j)) @f]
 * here the log sum exp is calculated by log_sum_exp() to 
 * avoid underflow.
 * 
 * @param model the HMM model.
 * @param seq the observed sequence.
 * @param[out] logbeta the logartithm of backward variable,
 *   which is a matrix of size seq->size * model->n.
 *   The matrix should be allocated before calling.
 */
void backward_proc_log(const hmmgmm_t* model,
    const seq_t* seq, gsl_matrix* logbeta);

/**
 * Re-estimate the model parameters using Baum-Welch algorithm.
 *
 * @param[in,out] model the HMM model to be re-estimated.
 * @param data a set of observed sequences.
 * @param nos number of sequences.
 */
void baum_welch(hmmgmm_t* model, const seq_t** data, size_t nos);

#endif  // NOTGHMM_H_

