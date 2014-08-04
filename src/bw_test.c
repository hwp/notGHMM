// bw_test.c
// Test Baum-Welch algorithm.
//
// Author : Weipeng He <heweipeng@gmail.com>
// Copyright (c) 2014, All rights reserved.

#include "notghmm.h"

#include <stdio.h>
#include <math.h>
#include <assert.h>

#include <gsl/gsl_rng.h>

int main(int argc, char** argv) {
  gsl_rng_env_setup();

  FILE* in = fopen(argv[1], "r");
  assert(in);
  hmmgmm_t* model = hmmgmm_fscan(in);;

  size_t i;
  size_t size = 100;
  size_t nos = 100;
  seq_t** data = calloc(nos, sizeof(seq_t*));
  for (i = 0; i < nos; i++) {
    data[i] = seq_gen(model, size);
  }
  
  hmmgmm_t* model2 = hmmgmm_alloc(model->n, model->k,
      model->dim);
  random_init(model2, data, nos);

  printf("\n================\nModel 1\n");
  hmmgmm_fprint(stdout, model);
  printf("\n================\nModel 2\n");
  hmmgmm_fprint(stdout, model2);

  gsl_matrix* logalpha = gsl_matrix_alloc(size, model->n);

  double po = 0.0;
  double pn = 0.0;
  for (i = 0; i < nos; i++) {
    forward_proc_log(model, data[i], logalpha);
    po += hmm_log_likelihood(logalpha);
    forward_proc_log(model2, data[i], logalpha);
    pn += hmm_log_likelihood(logalpha);
  }

  baum_welch(model2, data, nos);

  printf("\n================\nModel 2 (re-estimated)\n");
  hmmgmm_fprint(stdout, model2);

  double pr = 0.0;
  for (i = 0; i < nos; i++) {
    forward_proc_log(model2, data[i], logalpha);
    pr += hmm_log_likelihood(logalpha);
  }

  printf("\n P = %g, %g, %g\n", po, pn, pr);

  hmmgmm_free(model);
  hmmgmm_free(model2);
  for (i = 0; i < nos; i++) {
    seq_free(data[i]);
  }
  free(data);
  gsl_matrix_free(logalpha);

  return 0;
}

