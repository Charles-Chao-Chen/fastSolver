#ifndef __DIRECT_SOLVE
#define __DIRECT_SOLVE

#include "Htree.h"

/*
void dirct_circulant_solve(double *soln, double *rhs, int rhs_rows, int rhs_cols, int r, double diag);

void dirct_circulant_solve(double *soln, int rand_seed, int rhs_rows,
			   int nregions, int rhs_cols, int r, double
			   diag);

void
dirct_circulant_solve(std::string soln_file, int rand_seed,
		      int rhs_rows, int nregions, int rhs_cols,
		      int r, double diag);
*/

void
compute_L2_error(HodlrMatrix &lr_mat, int rand_seed, int rhs_rows,
		 int nregions, int rhs_cols, int rank,
		 double diag, Context ctx, HighLevelRuntime *runtime);


  
#endif
