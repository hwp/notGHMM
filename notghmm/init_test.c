// init_test.c
// Test the init functions
//
// Author : Weipeng He <heweipeng@gmail.com>
// Copyright (c) 2014, All rights reserved.

#include "notghmm.h"

#include <stdio.h>
#include <assert.h>

int main(int argc, char** argv) {
  FILE* in = fopen("hmm0", "r");
  assert(in);
  hmmgmm_t* model = hmmgmm_fread(in);
  fclose(in);

  gsl_rng_env_setup();
  gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);

  size_t i;
  size_t size = 1000;
  size_t nos = 100;
  seq_t** data = calloc(nos, sizeof(seq_t*));
  for (i = 0; i < nos; i++) {
    data[i] = seq_gen(model, size, rng);
  }

  hmmgmm_t* rinit = hmmgmm_alloc(2, 2, 2, 0);
  random_init(rinit, data, nos, rng);
  assert(hmmgmm_valid(rinit));

  hmmgmm_t* kinit = hmmgmm_alloc(2, 2, 2, 0);
  random_init(kinit, data, nos, rng);
  assert(hmmgmm_valid(kinit));

  printf("original model : %g\n", hmm_log_likelihood_all(model, data, nos));
  printf("random init    : %g\n", hmm_log_likelihood_all(rinit, data, nos));
  printf("kmeans init    : %g\n", hmm_log_likelihood_all(kinit, data, nos));

  // estimate 
  baum_welch(rinit, data, nos);
  baum_welch(kinit, data, nos);

  printf("rinit estimate : %g\n", hmm_log_likelihood_all(rinit, data, nos));
  printf("kinit estimate : %g\n", hmm_log_likelihood_all(kinit, data, nos));

  hmmgmm_free(model);
  hmmgmm_free(rinit);
  hmmgmm_free(kinit);

  for (i = 0; i < nos; i++) {
    seq_free(data[i]);
  }
  free(data);

  gsl_rng_free(rng);

  return 0;
}


