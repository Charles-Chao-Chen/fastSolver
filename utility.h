#ifndef __UTILITY_H
#define __UTILITY_H

#include <stdlib.h>

/* Uniform random number generator */
#define frand(xmin,xmax) ((double)xmin+(double)(xmax-xmin)*rand()/ \
			  (double)RAND_MAX) 

void dirct_circulant_solve(double *soln, double *rhs, int rhs_rows, int rhs_cols, int r, double diag);


/* Note for using blas and lapack library:
   
 * The object is easily portable, disregarding of the BLAS LAPACK
 * library the person uses (original netlib BLAS and LAPACK,
 * GotoBLAS, Intel MKL, AMD CML, ...)

 * Most of these libraries contain the FORTRAN versions
 * with/without a c wrapper. CLAPACK and CBLAS on the
 * other hand, are fully f2c versions of the original FORTRAN code
 * and need F2Clibs to work.

 * This means that if you take a look at the symbols whit for example
 * the gnutool nm or objdump, you will notice that all the same symbols
 * (representing the functions) are there. For BLAS this means srotg_,
 * drotg_ scopy_, dcopy_, ... (notice the underscore at the end).

 * For this reason, directly using these symbols is suggested to call the
 * functions, and this according the rules of engagement when it comes to
 * calling FORTRAN from C(++). This makes sure that you don't have to
 * implement any different BLAS wrappers like there exist several and are
 * also included in the Intel and AMD versions.
 */

namespace blas {
  extern "C" {
    // Declaration for BLAS matrix-vector multiply
    // note op(A) is m x k and op(B) is k x n, so C is m x n
    void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha,
		double *A, int *lda, double *B, int *ldb, double *beta,
		double *C, int *ldc);
  }
}

  
namespace lapack {
  extern "C" {
    // Declaration for lapack LU solve routine
    void dgesv_(int *N, int *NRHS, double *A, int *LDA, int *IPIV, double *B, int *LDB, int *INFO);
  }
}




#endif // __UTILITY_H
