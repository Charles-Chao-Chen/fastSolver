#ifndef _FAST_SOLVER
#define _FAST_SOLVER

#include "legion.h"
#include "hodlr_matrix.h"

void register_solver_tasks();

class FastSolver {
 public:
  FastSolver();
  
  void solve_top(const HodlrMatrix&, const Range& mappingTag,
		 Context ctx, HighLevelRuntime *runtime);
  
  // wrapper for solve functions
  void solve_dfs(HodlrMatrix &, int, Context, HighLevelRuntime *);
  //void solve_bfs(HodlrMatrix &, int, Context, HighLevelRuntime *);
  void bfs_solve(HodlrMatrix &, const Range&,
		 Context, HighLevelRuntime *);
 
  void display_launch_time() const {
    std::cout << "Time for launching solve-tasks : " << time_launcher
	      << std::endl;}
  
 private:
  void solve_dfs(Node *, Node *, Range,
		 Context, HighLevelRuntime *);

    /*
  void solve_bfs(Node *, Node *, Range, 
		 Context, HighLevelRuntime *);

  void visit(Node *, Node *, const Range,
	     double&, double&, double&,
	     Context, HighLevelRuntime *);
  */
 private:
  double time_launcher; // time of launching all the tasks
};


#endif // _FAST_SOLVER
