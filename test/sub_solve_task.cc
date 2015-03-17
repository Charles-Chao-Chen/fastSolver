#include "sub_solve_task.hpp"

void ExtractRegions
(const FSTreeNode* node, LMatrixArray& matArr) {
  if (node->is_legion_leaf()) {
    matArr.append( (*node->lowrank_matrix) );
  }
  else {
    ExtractRegions(node->lchild, matArr);
    ExtractRegions(node->rchild, matArr);
  }
}

void ExtractRegions
( const HodlrMatrix& hMatrix, LMatrixArray& matArr ) {
  ExtractRegions( hMatrix.uroot, matArr );
}

/* ---- SubSolveTask implementation ---- */

SubSolveTask::TaskArgs::TaskArgs
(int sl, int gl, long s, const std::string& str, const Range& p,
 int rhs, int r, int t, int ls, double d)
  : subLvl(sl), gloLvl(gl), seed(s), procs(p),
    nRHS(rhs), rank(r), threshold(t), leafSize(ls), diagonal(d) {

  strcpy(name, str.c_str());
}

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

  // receive h-matrix configuration from paraent task
  const int nRHS = args->nRHS;
  const int rank = args->rank;
  const int threshold = args->threshold;
  const int leafSize = args->leafSize;
  const double diagonal = args->diagonal;
  const int nRow = threshold*(1<<subLevel);
    
  HodlrMatrix hMatrix(nRHS, nRow, gloLevel, subLevel, rank,
  		      threshold, leafSize, name);
  hMatrix.create_tree(ctx, runtime);
  
  // random right hand side
  hMatrix.init_rhs(seed, procs, ctx, runtime);
  hMatrix.init_circulant_matrix(diagonal, procs, ctx, runtime);
 
  FastSolver fs;
  fs.bfs_solve(hMatrix, procs, ctx, runtime);
  
  //#ifdef DEBUG
  #if true
  std::cout << "\n================================" << std::endl;
  std::cout << "sub-problem information:" << std::endl;
  int nleaf = hMatrix.get_num_leaf();
  std::cout << "  Legion leaf : "       << nleaf << std::endl
	    << "  Legion leaf / node: " << nleaf / procs.size
	    << std::endl;
  hMatrix.display_launch_time();
  fs.display_launch_time();

  assert( nRow%threshold == 0 );
  int nregion = nleaf;
  compute_L2_error(hMatrix, seed, nRow, nregion, nRHS,
		   rank, diagonal, ctx, runtime);

  std::cout << "================================\n" << std::endl;
#endif
    
  // return all regions to the parent task
  LMatrixArray matArr;
  ExtractRegions( hMatrix, matArr );
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


