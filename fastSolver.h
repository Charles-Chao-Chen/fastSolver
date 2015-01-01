#ifndef _FAST_SOLVER
#define _FAST_SOLVER

#include <string>
#include "legion.h"
#include "Htree.h"


void register_solver_tasks();


class FastSolver {

 public:
  FastSolver();

  // wrapper for solve functions
  void solve_dfs(LR_Matrix &, int, Context, HighLevelRuntime *);
  void solve_bfs(LR_Matrix &, int, Context, HighLevelRuntime *);

  // get err and time
  double get_L2norm_error() const {return err_L2;}
  double get_elapsed_time() const {return time_launcher;}
  
 private:
  void solve_dfs(FSTreeNode *, FSTreeNode *, Range,
		 Context, HighLevelRuntime *);

  void solve_bfs(FSTreeNode *, FSTreeNode *, Range,
		 Context, HighLevelRuntime *);

  void visit(FSTreeNode *, FSTreeNode *, Range,
	     Context, HighLevelRuntime *);

  //void save_solution(LR_Matrix &, Context, HighLevelRuntime *);

 private:
  double err_L2;        // error in l2 norm
  double time_launcher; // time of launching all the tasks
  std::string soln_file;
};


#endif // _FAST_SOLVER
