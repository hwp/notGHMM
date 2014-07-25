// genseq_test.c
// Test the generate sequence function.
//
// Author : Weipeng He <heweipeng@gmail.com>
// Copyright (c) 2014, All rights reserved.

#include "notghmm.h"

#include <stdio.h>

int main(int argc, char** argv) {
  hmmgmm_t* model = hmmgmm_alloc(2, 1, 2);

  HMMGMM_A(model, 0, 0) = .9;
  HMMGMM_A(model, 0, 1) = .1;
  HMMGMM_A(model, 1, 0) = .3;
  HMMGMM_A(model, 1, 1) = .7;

  HMMGMM_C_ROW(model, 0)[0] = 1.0;
  HMMGMM_C_ROW(model, 1)[0] = 1.0;

  HMMGMM_MU(model, 0, 0)[0] = 0.0;
  HMMGMM_MU(model, 0, 0)[1] = 0.0;
  HMMGMM_MU(model, 1, 0)[0] = 10.0;
  HMMGMM_MU(model, 1, 0)[1] = 5.0;

  HMMGMM_SIGMA(model, 0, 0)[0] = 1.0;
  HMMGMM_SIGMA(model, 0, 0)[1] = 0.0;
  HMMGMM_SIGMA(model, 0, 0)[2] = 0.0;
  HMMGMM_SIGMA(model, 0, 0)[3] = 1.0;

  HMMGMM_SIGMA(model, 1, 0)[0] = 2.0;
  HMMGMM_SIGMA(model, 1, 0)[1] = -1.0;
  HMMGMM_SIGMA(model, 1, 0)[2] = -1.0;
  HMMGMM_SIGMA(model, 1, 0)[3] = 2.0;

  int size = 1000;
  seq_t* seq = gen_sequence(model, size);
  int i;
  for (i = 0; i < size; i++) {
    printf("%g %g\n", SEQ_DATA_AT(seq, i)[0], SEQ_DATA_AT(seq, i)[1]);
  }

  return 0;
}

