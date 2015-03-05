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
  void bfs_solve(HodlrMatrix &, const Range&,
		 Context, HighLevelRuntime *);
 
  void display_launch_time() const {
    std::cout << "Time for launching solve-tasks : " << time_launcher
	      << std::endl;}
  
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
