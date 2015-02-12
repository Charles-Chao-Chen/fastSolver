#ifndef _FAST_SOLVER
#define _FAST_SOLVER

#include "legion.h"
#include "hodlr_matrix.h"


void register_solver_tasks();


class FastSolver {

 public:
  FastSolver();

  // wrapper for solve functions
  void solve_dfs(HodlrMatrix &, int, Context, HighLevelRuntime *);
  //void solve_bfs(HodlrMatrix &, int, Context, HighLevelRuntime *);
  void bfs_solve(HodlrMatrix &, int, Context, HighLevelRuntime *);
 
  // get err and time
  double get_elapsed_time() const {return time_launcher;}
  
 private:
  void solve_dfs(FSTreeNode *, FSTreeNode *, Range,
		 Context, HighLevelRuntime *);

    /*
  void solve_bfs(FSTreeNode *, FSTreeNode *, Range, 
		 Context, HighLevelRuntime *);

  void visit(FSTreeNode *, FSTreeNode *, const Range,
	     double&, double&, double&,
	     Context, HighLevelRuntime *);
  */
 private:
  double time_launcher; // time of launching all the tasks
};


#endif // _FAST_SOLVER
