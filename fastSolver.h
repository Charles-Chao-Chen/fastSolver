#ifndef _FAST_SOLVER
#define _FAST_SOLVER

#include "legion.h"
#include "Htree.h"


void register_solver_tasks();


class FastSolver {

 public:
  void solve_dfs(LR_Matrix &, int, Context, HighLevelRuntime *);
  void solve_bfs(LR_Matrix &, int, Context, HighLevelRuntime *);
  
 private:
  void solve_dfs(FSTreeNode *, FSTreeNode *, Range,
		 Context, HighLevelRuntime *);

  void solve_bfs(FSTreeNode *, FSTreeNode *, Range,
		 Context, HighLevelRuntime *);

  void visit(FSTreeNode *, FSTreeNode *, Range,
	     Context, HighLevelRuntime *);

 private:
  double err_L2;        // error in l2 norm
  double time_launcher; // time of launching all the tasks
};


#endif // _FAST_SOLVER
