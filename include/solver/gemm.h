#ifndef _GEMM_H
#define _GEMM_H

#include "hodlr_matrix.h"
#include "legion.h"


// These are the two types of GEMM that are needed.

// This GEMM requires a reduction.
//void gemm(double alpha, Node * v, range rv, Node * u, range ru, double beta,
//	  LogicalRegion & res, Context ctx, HighLevelRuntime *runtime);
// Returns res = alpha * v(rv)^T * u(ru) + beta * R;
// rv is always ALL in practice.
// Number of columns in res must match ru.
// Note the transpose on v.

void register_gemm_tasks();


void gemm_reduce
  (const double alpha,
   const Node *v, const Node *u, const Range &ru,
   const double beta,   LMatrix *(&result),  const Range taskTag,
   double& tCreate,
   Context ctx, HighLevelRuntime *runtime);


void gemm_broadcast
  (const double alpha, const Node * u, const Range &ru,
   LMatrix *(&eta),
   const double beta,  const Node * v, const Range &rv,
   const Range tag,
   Context ctx, HighLevelRuntime *runtime);


#endif // _GEMM_H
