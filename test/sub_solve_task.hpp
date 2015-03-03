#ifndef sub_solve_task_hpp
#define sub_solve_task_hpp

#include "range.h"
#include "fast_solver.h"
#include "direct_solve.h"
#include "timer.h"
#include "legion.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


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
  
  double t0 = timer();
  
  // random right hand size
  hMatrix.init_rhs(rand_seed, rhs_cols, ctx, runtime);
  
  // A = U U^T + diag and U is a circulant matrix
  hMatrix.init_circulant_matrix(diag, ctx, runtime);

  double t1 = timer();
  std::cout << "Init launching time: " << t1 - t0
	    << std::endl;
  
 
  FastSolver fs;
  fs.bfs_solve(hMatrix, num_proc, ctx, runtime);
  //fs.solve_bfs(hMatrix, num_proc, ctx, runtime);
  //fs.solve_dfs(hMatrix, num_proc, ctx, runtime);
  
  std::cout << "Tasks launching time: " << fs.get_elapsed_time()
	    << std::endl;

  if (compute_accuracy) {
    assert( N%threshold == 0 );
    int nregion = nleaf;
    compute_L2_error(hMatrix, rand_seed, rhs_rows,
		     nregion, rhs_cols,
		     rank, diag,
		     ctx, runtime);
  }
}


class SubSolveTask : public TaskLauncher {
 public:
  struct TaskArgs {
    TaskArgs(int l, Range tag, const std::string& str) :
      treeLevel(l), taskTag(tag)
    { strcpy(name, str.c_str()); }
    int treeLevel;
    Range taskTag;
    char name[20];
  };
  
  SubSolveTask(TaskArgument arg,
	      Predicate pred = Predicate::TRUE_PRED,
	      MapperID id = 0,
	      MappingTagID tag = 0);
  
  static int TASKID;
  static void register_tasks(void);

 public:
  static void cpu_task(const Task *task,
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
    <SubSolveTask::cpu_task>(AUTO_GENERATE_ID,
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

void SubSolveTask::cpu_task
(const Task *task,
 const std::vector<PhysicalRegion> &regions,
 Context ctx, HighLevelRuntime *runtime)
{
  const TaskArgs args  = *((TaskArgs*)task->args);
  const int level = args.treeLevel;
  const Range taskTag = args.taskTag;
  const char* testName = args.name;
  int nleaf = 1;
  int nproc = 1;
  run_test(6,   /* rank */
	   15*(2<<2),  /* N */
	   15,   /* threshold*/
	   nleaf,/* nleaf_per_legion_node */
	   1.e1, /* diagonal */
	   nproc,    /* # of processors */
	   true, /* compute accuracy */
	   testName,
	   ctx,
	   runtime);


  /*
  HTree tree( level );
  tree.init_matrices();

  // solve the local problem
  FastSolver solver;
  solver.solve( tree );

  // return all regions to the parent task
  int numRegions = pow(2, level);
  LogicalRegion regions[ numRegions ];
  tree.get_regions( regions );
  return regions;
  */
}


#endif // sub_solve_task_hpp
