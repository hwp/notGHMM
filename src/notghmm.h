// notghmm.h
// Header file for basic data structure and functions.
//
// Author : Weipeng He <heweipeng@gmail.com>
// Copyright (c) 2014, All rights reserved.

#ifndef NOTGHMM_H_
#define NOTGHMM_H_

/**
 * Sequence of observed data.
 */
typedef struct {
  /**
   * Length of the sequence. 
   */
  int size;

  /**
   * Number of dimension of observed data.
   */
  int dim;

  /**
   * The observed data.
   * data[i*dim+j] is the j-th dimension of the observed data at time i.
   */
  double* data;
} seq_t;

#define SEQ_DATA_AT(s, t) ((s)->data + (t) * (s)->dim)

/**
 * A HMM-GMM model.
 */
typedef struct {
  /**
   * Number of hidden states.
   */
  int n;

  /**
   * Number of components of each mixture.
   */
  int m;

  /**
   * Number of dimension of observed data.
   */
  int dim;

  /**
   * Probability of initial state. Its size is n.
   */
  double* pi;

  /**
   * Transition probability. Its size is n^2.
   * a[i * n + j] = P(q_{t+1}=j|q_t=i)
   */
  double* a;

  /**
   * Mixture weight. Its size is n * m;
   * c[i*m+j] is the mixture weight of the j-th component of state i.
   */
  double* c;

  /**
   * Mean of the components. Its size is n*m*dim.
   * mu[(i*m+j)*dim+k] is the k-th dimension 
   *   of the j-th component of state i.
   */
  double* mu;

  /**
   * Covariance of the components. Its size is n*m*dim^2.
   * sigma[(i*m+j)*dim^2+k*dim+l] is the k-th row, l-th column 
   *   of the covariance matrix of the j-th component of state i.
   */
  double* sigma;
} hmmgmm_t;

#define HMMGMM_A(mo, i, j) (mo)->a[(i) * (mo)->n + (j)]
#define HMMGMM_A_ROW(mo, i) ((mo)->a + (i) * (mo)->n)
#define HMMGMM_C_ROW(mo, i) ((mo)->c + (i) * (mo)->m)
#define HMMGMM_MU(mo, i, j) ((mo)->mu + ((i) * (mo)->m + (j)) * (mo)->dim)
#define HMMGMM_SIGMA(mo, i, j) ((mo)->sigma + ((i) * (mo)->m + (j)) * (mo)->dim * (mo)->dim)

/**
 * Allocate memory for a sequence.
 * @p size and @p dim are set according to the arguments.
 * @p data is allocated but not initialized.
 *
 * @param size length of the sequence.
 * @param dim number of dimensions.
 *
 * @return pointer to the allocated space. 
 *         NULL, if error occurs.
 */
seq_t* seq_alloc(int size, int dim);

/**
 * Free memory for the sequence.
 */
void seq_free(seq_t* seq);

/**
 * Allocate memory for a HMM-GMM model.
 * @p n, @p m and @p dim are set according to the arguments.
 * @p pi, @p a, @p c, @p mu and @p sigma are allocated but not initialized.
 *
 * @param n number of hidden states.
 * @param m number of components.
 * @param dim number of dimension.
 *
 * @return pointer to the allocated space.
 *         NULL, if error occurs.
 */
hmmgmm_t* hmmgmm_alloc(int n, int m, int dim);

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
seq_t* gen_sequence(hmmgmm_t* model, int size);

/**
 * Re-estimate the model parameters using Baum-Welch algorithm.
 * 
 * @param[in,out] model the HMM model to be re-estimated.
 * @param data a set of observed sequences.
 * @param nos number of sequences.
 */
void baum_welch(hmmgmm_t* model, seq_t** data, int nos);

#endif  // NOTGHMM_H_

