#include <assert.h>
#include <math.h>
#include <iostream>
#include "utility.h"


void dirct_circulant_solve(double *soln, double *rhs, int rhs_rows, int rhs_cols, int r, double diag) {

  double *U = (double *) malloc(rhs_rows*r*sizeof(double));
  for (int j=0; j<r; j++)
    for (int i=0; i<rhs_rows; i++)
      U[i+j*rhs_rows] = (i+j)%r;

  double *A = (double *) calloc(rhs_rows*rhs_rows, sizeof(double));
  for (int i=0; i<rhs_rows; i++)
    A[i*(rhs_rows+1)] = diag;

  char transa = 'n';
  char transb = 't';
  int  m = rhs_rows;
  int  n = rhs_rows;
  int  k = r;
  int  lda = rhs_rows;
  int  ldb = rhs_rows;
  int  ldc = rhs_rows;
  double alpha = 1.0;
  double beta  = 1.0;
  
  blas::dgemm_(&transa, &transb, &m, &n, &k, &alpha, U, &lda, U, &ldb, &beta, A, &ldc);

  int INFO;
  int IPIV[m];
  lapack::dgesv_(&m, &rhs_cols, A, &lda, IPIV, rhs, &lda, &INFO);
  assert(INFO == 0);

  double diff  = 0;
  double denom = 0;
  for (int j=0; j<rhs_cols; j++)
    for (int i=0; i<rhs_rows; i++) {
      diff += (soln[i+j*rhs_rows] - rhs[i+j*rhs_rows]) *(soln[i+j*rhs_rows] - rhs[i+j*rhs_rows]);
      denom += rhs[i+j*rhs_rows] * rhs[i+j*rhs_rows];
    }

  std::cout << "Err: " << sqrt(diff/denom) << std::endl;
  
  free(U);
  free(A);
}


