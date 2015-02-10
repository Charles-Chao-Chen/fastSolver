#ifndef TEST_H
#define TEST_H

#include "legion.h"
#include "fast_solver.h"
#include "direct_solve.h"


void run_test
(int rank, int N, int threshold, int nleaf_per_legion_node,
 double diag, int num_proc, bool compute_accuracy,
 Context ctx, HighLevelRuntime *runtime );


void test_accuracy(Context ctx, HighLevelRuntime *runtime);

void test_performance(Context ctx, HighLevelRuntime *runtime);

void test1(int, Context ctx, HighLevelRuntime *runtime);

void test2(int, Context ctx, HighLevelRuntime *runtime);

void test3(int, int, Context ctx, HighLevelRuntime *runtime);


#endif // TEST_H
