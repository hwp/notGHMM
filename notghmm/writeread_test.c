// writeread_test.c
// Test the write and read functions
//
// Author : Weipeng He <heweipeng@gmail.com>
// Copyright (c) 2014, All rights reserved.

#include "notghmm.h"

#include <stdio.h>
#include <assert.h>

int main(int argc, char** argv) {
  hmmgmm_t* model = hmmgmm_alloc(2, 1, 2);

  gsl_vector_set(model->pi, 0, 1.0);
  gsl_vector_set(model->pi, 1, 0.0);

  gsl_matrix_set(model->a, 0, 0, 8.0 / 17.0);
  gsl_matrix_set(model->a, 0, 1, 9.0 / 17.0);
  gsl_matrix_set(model->a, 1, 0, 8.0 / 9.0);
  gsl_matrix_set(model->a, 1, 1, 1.0 / 9.0);

  gsl_vector_set(model->states[0]->weight, 0, 1.0);
  gsl_vector_set(model->states[1]->weight, 0, 1.0);

  gmm_t* state = model->states[0];
  gsl_vector_set(state->comp[0]->mean, 0, 0.0);
  gsl_vector_set(state->comp[0]->mean, 1, 0.0);
  gsl_matrix_set(state->comp[0]->cov, 0, 0, 1.0);
  gsl_matrix_set(state->comp[0]->cov, 0, 1, 0.0);
  gsl_matrix_set(state->comp[0]->cov, 1, 0, 0.0);
  gsl_matrix_set(state->comp[0]->cov, 1, 1, 1.0);

  state = model->states[1];
  gsl_vector_set(state->comp[0]->mean, 0, 1.0);
  gsl_vector_set(state->comp[0]->mean, 1, 1.0);
  gsl_matrix_set(state->comp[0]->cov, 0, 0, 1.0);
  gsl_matrix_set(state->comp[0]->cov, 0, 1, -1.0);
  gsl_matrix_set(state->comp[0]->cov, 1, 0, -1.0);
  gsl_matrix_set(state->comp[0]->cov, 1, 1, 2.0);

  FILE* out = fopen("hmm0", "w");
  assert(out);
  hmmgmm_fprint(out, model);
  fclose(out);

  FILE* in = fopen("hmm0", "r");
  assert(in);
  hmmgmm_t* modelr = hmmgmm_fscan(in);
  fclose(in);

  hmmgmm_fprint(stdout, model);
  printf("==============\n");
  hmmgmm_fprint(stdout, modelr);

  gsl_matrix* a = gsl_matrix_alloc(model->n, model->n);
  gsl_matrix_memcpy(a, modelr->a);
  gsl_matrix_sub(a, model->a);
  printf("==============\n");
  matrix_fprint(stdout, a);

  hmmgmm_free(model);
  hmmgmm_free(modelr);

  return 0;
}

