#ifndef sub_solve_task_hpp
#define sub_solve_task_hpp

#include "range.h"
#include "fast_solver.h"
#include "direct_solve.h"
#include "matrix_array.hpp"
#include "legion.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

class SubSolveTask : public TaskLauncher {
 public:
  struct TaskArgs {
    TaskArgs(int sl, int gl, long s,
	     const std::string& str,
	     const Range& p) :
      subLvl(sl), gloLvl(gl),
      seed(s), procs(p)
    { strcpy(name, str.c_str()); }

    int subLvl;
    int gloLvl;
    int nRHS;
    int nRow;
    long seed;
    char name[50];
    Range procs;
  };
  
  SubSolveTask(TaskArgument arg,
	      Predicate pred = Predicate::TRUE_PRED,
	      MapperID id = 0,
	      MappingTagID tag = 0);
  
  static int TASKID;
  static void register_tasks(void);
 public:
  static LMatrixArray
  cpu_task (const Task *task,
	    const std::vector<PhysicalRegion> &regions,
	    Context ctx, HighLevelRuntime *runtime);
};

/* ---- SubSolveTask implementation ---- */
/*static*/
int SubSolveTask::TASKID;

SubSolveTask::
SubSolveTask(TaskArgument arg,
	    Predicate pred /*= Predicate::TRUE_PRED*/,
	    MapperID id /*= 0*/,
	    MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag)
{
}

/*static*/
void SubSolveTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <LMatrixArray, SubSolveTask::cpu_task>(AUTO_GENERATE_ID,
			     Processor::LOC_PROC, 
			     true,
			     true,
			     AUTO_GENERATE_ID,
			     TaskConfigOptions(false/*leaf*/,
					       true/*inner*/),
			     "sub-solve inner task");
#ifdef SHOW_REGISTER_TASKS
  printf("Register inner task %d : sub-solve\n", TASKID);
#endif
}

LMatrixArray SubSolveTask::cpu_task
(const Task *task,
 const std::vector<PhysicalRegion> &regions,
 Context ctx, HighLevelRuntime *runtime)
{
  const TaskArgs* args = (TaskArgs*)task->args;
  const int subLevel = args->subLvl; // level of sub problem
  const int gloLevel = args->gloLvl; // level of global problem
  const long seed = args->seed; // random seed
  const char* name = args->name;
  const Range procs = args->procs;

  // rank, threshold and other parameters
  //  are set inside the sub-tasks
  
  int nRHS = 2;             // # of rhs
  int rank = 90;
  int threshold = 150;
  int leafSize = 1;         // legion leaf size
  double diagonal = 1.0e4;
  int nRow = threshold*(1<<subLevel);

  HodlrMatrix hMatrix(nRHS, nRow, gloLevel, subLevel, rank,
		      threshold, leafSize, name);
  hMatrix.create_tree(ctx, runtime);
  int nleaf = hMatrix.get_num_leaf();
  std::cout << "Legion leaf : "	      << nleaf << std::endl
	    << "Legion leaf / node: " << nleaf / procs.size
	    << std::endl;
  
  // random right hand size
  hMatrix.init_rhs(seed, procs, ctx, runtime);
  hMatrix.init_circulant_matrix(diagonal, procs, ctx, runtime);
  hMatrix.display_launch_time();
 
  FastSolver fs;
  fs.bfs_solve(hMatrix, procs, ctx, runtime);
  fs.display_launch_time();  

  if (false) {
    assert( nRow%threshold == 0 );
    int nregion = nleaf;
    compute_L2_error(hMatrix, seed, nRow, nregion, nRHS,
		     rank, diagonal, ctx, runtime);
  }
  
  // return all regions to the parent task
  LMatrixArray matArr;
  matArr.get_regions( hMatrix );
  return matArr;
}

/*
// leaf_size: how many real leaves every legion
// node has
void run_test(int rank, int N, int threshold,
	      int leaf_size, double diag, int num_proc,
	      bool compute_accuracy, const std::string& name,
	      Context ctx, HighLevelRuntime *runtime) {

#ifdef DEBUG
  std::cout << "************* debugging mode ************"
	    << std::endl;
#endif

  printf(" %d processes\n %d leaves\n\n", num_proc, leaf_size);
     
  int rhs_cols = 2;
  int rhs_rows = N;
  long int rand_seed = 1123;
  
  HodlrMatrix hMatrix;
  // create H-tree with legion leaf
  hMatrix.create_tree(N, threshold, rhs_cols, rank,
		      leaf_size, num_proc, name, ctx, runtime);
  int nleaf = hMatrix.get_num_leaf();
  std::cout << "Legion leaf : "
	    << nleaf
	    << std::endl
	    << "Legion leaf / node: "
	    << nleaf / num_proc
	    << std::endl;

  std::cout << "Launch node : "
	    << hMatrix.get_num_launch_node()
	    << std::endl;
  
  Timer t; t.start();
  // random right hand size
  hMatrix.init_rhs(rand_seed, rhs_cols, ctx, runtime);  
  // A = U U^T + diag and U is a circulant matrix
  hMatrix.init_circulant_matrix(diag, ctx, runtime);
  t.stop(); t.get_elapsed_time("init launch");
 
  FastSolver fs;
  fs.bfs_solve(hMatrix, num_proc, ctx, runtime);
  //fs.solve_bfs(hMatrix, num_proc, ctx, runtime);
  //fs.solve_dfs(hMatrix, num_proc, ctx, runtime);
  fs.display_launch_time();
  
  if (compute_accuracy) {
    assert( N%threshold == 0 );
    int nregion = nleaf;
    compute_L2_error(hMatrix, rand_seed, rhs_rows,
		     nregion, rhs_cols,
		     rank, diag,
		     ctx, runtime);
  }
}
*/

#endif // sub_solve_task_hpp
