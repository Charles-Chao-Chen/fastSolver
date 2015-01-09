#ifndef _GEMM_H
#define _GEMM_H

#include "Htree.h"
#include "legion.h"


// These are the two types of GEMM that are needed.

// This GEMM requires a reduction.
//void gemm(double alpha, FSTreeNode * v, range rv, FSTreeNode * u, range ru, double beta,
//	  LogicalRegion & res, Context ctx, HighLevelRuntime *runtime);
// Returns res = alpha * v(rv)^T * u(ru) + beta * R;
// rv is always ALL in practice.
// Number of columns in res must match ru.
// Note the transpose on v.

void
register_gemm_tasks();

void
gemm_reduce(double alpha, FSTreeNode *v, FSTreeNode *u, range ru,
	    double beta, LogicalRegion & res,
	    Range task_tag,
	    Context ctx, HighLevelRuntime *runtime);

void gemm_broadcast
(double alpha, FSTreeNode * u, range ru,
 LogicalRegion &eta,
 double beta,  FSTreeNode * v, range rv,
 const Range tag,
 Context ctx, HighLevelRuntime *runtime);


#endif // _GEMM_H
