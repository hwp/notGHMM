// init_test.c
// Test the init functions
//
// Author : Weipeng He <heweipeng@gmail.com>
// Copyright (c) 2014, All rights reserved.

#include "notghmm.h"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>

int main(int argc, char** argv) {
  // get options
  int showhelp = 0;
  unsigned int n_states = 0;
  unsigned int n_comp = 0;
  int cov_diag = 0;
  size_t length = 1000;
  size_t nos = 100;

  int opt;
  while ((opt = getopt(argc, argv, "n:k:c:l:s:h")) != -1) {
    switch (opt) {
      case 'h':
        showhelp = 1;
        break;
      case 'n':
        n_states = atoi(optarg);
        break;
      case 'k':
        n_comp = atoi(optarg);
        break;
      case 'c':
        cov_diag = atoi(optarg);
        break;
      case 'l':
        length = atoi(optarg);
        break;
      case 's':
        nos = atoi(optarg);
        break;
      default:
        showhelp = 1;
        break;
    }
  }

  if (showhelp || n_states <= 0 || n_comp <= 0 || length <= 0 || nos <= 0) {
    fprintf(stderr, "Usage: %s -n num_states -k num_components "
        "[-c cov_diag] -l length -s num_seq\n", argv[0]);
    exit(EXIT_SUCCESS);
  }

  FILE* in = fopen("hmm0", "r");
  assert(in);
  hmmgmm_t* model = hmmgmm_fread(in);
  fclose(in);

  gsl_rng_env_setup();
  gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);

  size_t i;
  seq_t** data = calloc(nos, sizeof(seq_t*));
  for (i = 0; i < nos; i++) {
    data[i] = seq_gen(model, length, rng);
  }

  hmmgmm_t* rinit = hmmgmm_alloc(n_states, n_comp, model->dim, cov_diag);
  random_init(rinit, data, nos, rng);
  assert(hmmgmm_valid(rinit));

  hmmgmm_t* kinit = hmmgmm_alloc(n_states, n_comp, model->dim, cov_diag);
  kmeans_init(kinit, data, nos, rng);
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


