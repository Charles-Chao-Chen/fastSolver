#include <algorithm>
#include <assert.h>
#include <list>
#include "fast_solver.h"
#include "solver_tasks.h"
#include "gemm.h"
#include "launch_node_task.h"
#include "zero_matrix_task.h"
#include "init_matrix_tasks.h"
#include "save_region_task.h"
#include "htree_helper.h"
#include "lapack_blas.h"
#include "timer.h"
#include "macros.h"
#include "unistd.h"

void solve_bfs_launch
(FSTreeNode *uroot, FSTreeNode *vroot,
 Range mappingTag, Context ctx, HighLevelRuntime *runtime);

void visit_launch_node
(FSTreeNode *unode, FSTreeNode *vnode, const Range mappingTag,
 double& tRed, double& tBroad, double& tCreate,
 Context ctx, HighLevelRuntime *runtime);

void launch_solve_tasks
(FSTreeNode *unode, FSTreeNode *vnode, const Range task_tag,
 Context ctx, HighLevelRuntime *runtime);

void solve_bfs
(FSTreeNode * uroot, FSTreeNode *vroot,
 Range mappingTag, Context ctx, HighLevelRuntime *runtime);

void visit
(FSTreeNode *unode, FSTreeNode *vnode, const Range mappingTag,
 double& tRed, double& tBroad, double& tCreate,
 Context ctx, HighLevelRuntime *runtime);

void add_subtree_regions
(LaunchNodeTask &launcher, FSTreeNode *uroot, FSTreeNode *vroot);

void add_umat_regions 
(LaunchNodeTask &launcher, FSTreeNode *unode);

void add_vmat_regions 
(LaunchNodeTask &launcher, FSTreeNode *vnode);

void add_hmat_regions 
(LaunchNodeTask &launcher, FSTreeNode *hnode);

void add_kmat_regions 
(LaunchNodeTask &launcher, FSTreeNode *vnode);



void register_solver_tasks() {

  char hostname[1024];
  gethostname(hostname, 1024);
  std::cout << "Registering all solver tasks on "
	    << hostname
	    << std::endl;
  register_solver_operators();  
  register_gemm_tasks();
  register_launch_node_task();
  register_zero_matrix_task();
  register_init_tasks();
  register_save_region_task();
  std::cout << std::endl;
}


FastSolver::FastSolver():
  time_launcher(-1) {}


//void FastSolver::solve_bfs
void FastSolver::bfs_solve
(HodlrMatrix &lr_mat, int nProc,
 Context ctx, HighLevelRuntime *runtime)
{
  std::cout << "Launch tasks in breadth first order."
	    << std::endl;

  Range tag(nProc);
  double t0 = timer();
  solve_bfs_launch(lr_mat.uroot, lr_mat.vroot, tag, ctx, runtime);
  //solve_bfs(lr_mat.uroot, lr_mat.vroot, tag, ctx, runtime);
  double t1 = timer();
  this->time_launcher = t1 - t0;
}


// TODO: this can also be implemented using dfs
//  and there is no difference in traversal order
//  when only leaf nodes are visited
// rmk: bfs may be more expensive because of the
//  push operations
void solve_bfs_launch
(FSTreeNode *uroot, FSTreeNode *vroot,
 Range mappingTag, Context ctx, HighLevelRuntime *runtime) {

  std::list<FSTreeNode *> ulist;
  std::list<FSTreeNode *> vlist;
  ulist.push_back(uroot);
  vlist.push_back(vroot);
  typedef std::list<FSTreeNode *>::iterator         Titer;
  typedef std::list<FSTreeNode *>::reverse_iterator RTiter;

  std::list<Range> rglist;
  rglist.push_back(mappingTag);
  typedef std::list<Range>::iterator         Riter;
  typedef std::list<Range>::reverse_iterator RRiter;

  Titer uit = ulist.begin();
  Titer vit = vlist.begin();
  Riter rit = rglist.begin();
  for (; uit != ulist.end(); uit++, vit++, rit++) {
    Range rglchild = rit->lchild();
    Range rgrchild = rit->rchild();
    FSTreeNode *ulchild = (*uit)->lchild;
    FSTreeNode *urchild = (*uit)->rchild;
    FSTreeNode *vlchild = (*vit)->lchild;
    FSTreeNode *vrchild = (*vit)->rchild;
    //if (      ! (*uit)->is_legion_leaf() ) {
    //assert( ! (*vit)->is_legion_leaf() );
    if ( ! (*uit)->is_launch_node() ) {
      ulist.push_back( ulchild );
      ulist.push_back( urchild );
      vlist.push_back( vlchild );
      vlist.push_back( vrchild );
      rglist.push_back( rglchild );
      rglist.push_back( rgrchild );
    }
  }
  RTiter ruit  = ulist.rbegin();
  RTiter rvit  = vlist.rbegin();
  RRiter rrgit = rglist.rbegin();

  std::cout << "ulist size: " << ulist.size() << std::endl;

  
  double tRed = 0, tCreate = 0, tBroad = 0;
  for (; ruit != ulist.rend(); ruit++, rvit++, rrgit++) {
    if ((*ruit)->is_launch_node())
      visit_launch_node(*ruit, *rvit, *rrgit,
			tRed, tBroad, tCreate,
			ctx, runtime);
    else
      break;
  }
  
    //visit(*ruit, *rvit, *rrgit,
    //	  tRed, tBroad, tCreate,
    //	  ctx, runtime);
  /*
  std::cout << "launch reduction task: " << tRed   << std::endl
	    << "launch create task: " << tCreate   << std::endl
	    << "launch broadcast task: " << tBroad << std::endl;
  */
}


void visit_launch_node
(FSTreeNode *unode, FSTreeNode *vnode, const Range mappingTag,
 double& tRed, double& tBroad, double& tCreate,
 Context ctx, HighLevelRuntime *runtime)
{
  
  if ( unode->is_launch_node() ) {
    launch_solve_tasks(unode, vnode, mappingTag, ctx, runtime);
    return;
  }

  FSTreeNode * b0 = unode->lchild;
  FSTreeNode * b1 = unode->rchild;  
  FSTreeNode * V0 = vnode->lchild;
  FSTreeNode * V1 = vnode->rchild;

  const Range mappingTag0 = mappingTag.lchild();
  const Range mappingTag1 = mappingTag.rchild();

  assert( ! unode->is_legion_leaf() );
  assert( V0->Hmat != NULL );
  assert( V1->Hmat != NULL );

  // This involves a reduction for V0Tu0, V0Td0, V1Tu1, V1Td1
  // from leaves to root in the H tree.
  //LogicalRegion V0Tu0, V0Td0, V1Tu1, V1Td1;
  LMatrix *V0Tu0 = 0;
  LMatrix *V0Td0 = 0;
  LMatrix *V1Tu1 = 0;
  LMatrix *V1Td1 = 0;
  Range ru0(b0->col_beg, b0->ncol);
  Range ru1(b1->col_beg, b1->ncol);
  Range rd0(0,           b0->col_beg);
  Range rd1(0,           b1->col_beg);

  double t0 = timer();
  gemm_reduce(1., V0->Hmat, b0, ru0, 0., V0Tu0,
	      mappingTag0, tCreate, ctx, runtime);
  gemm_reduce(1., V1->Hmat, b1, ru1, 0., V1Tu1,
	      mappingTag1, tCreate, ctx, runtime);
  gemm_reduce(1., V0->Hmat, b0, rd0, 0., V0Td0,
	      mappingTag0, tCreate, ctx, runtime);
  gemm_reduce(1., V1->Hmat, b1, rd1, 0., V1Td1,
	      mappingTag1, tCreate, ctx, runtime);
  tRed += timer() - t0;
  
  // V0Td0 and V1Td1 contain the solution on output.
  // eta0 = V1Td1
  // eta1 = V0Td0
  solve_node_matrix(V0Tu0, V1Tu1,
		    V0Td0, V1Td1,
		    mappingTag0, ctx, runtime);  

  // This step requires a broadcast of V0Td0 and V1Td1
  // from root to leaves.
  // Assemble x from d0 and d1: merge two trees

  double t1 = timer();
  gemm_broadcast(-1., b0, ru0, V1Td1, 1., b0, rd0,
		 mappingTag0, ctx, runtime);
  gemm_broadcast(-1., b1, ru1, V0Td0, 1., b1, rd1,
		 mappingTag1, ctx, runtime);
  tBroad += timer() - t1;
}


void launch_solve_tasks
(FSTreeNode *unode, FSTreeNode *vnode, const Range task_tag,
 Context ctx, HighLevelRuntime *runtime)
{
  int nleaf = count_leaf(unode);
  int subtree_size = nleaf * 2;
  int arraySize = subtree_size*2+2;
  assert(arraySize < MAX_TREE_SIZE);

  LaunchNodeTask::TaskArgs<MAX_TREE_SIZE> args;
  args.taskTag = task_tag;
  args.treeSize = subtree_size;
  args.treeArray[0] = *vnode;
  tree_to_array(vnode, args.treeArray, 0);
  args.treeArray[subtree_size] = *unode;
  tree_to_array(unode, args.treeArray, 0, subtree_size);

  // special mapping ID for launch node tasks
  //  the regions will be virtually mapped
  LaunchNodeTask launcher(TaskArgument( &args, sizeof(args)),
			  Predicate::TRUE_PRED,
			  0,
			  -(task_tag.begin)
			  );

  
  // add regions
  add_subtree_regions(launcher, unode, vnode);
  
  // add fields
  for (unsigned i=0; i<launcher.region_requirements.size(); i++)
    launcher.region_requirements[i].add_field(FID_X);


  std::cout << "Total # of regions : "
	    << launcher.region_requirements.size()
	    << std::endl;

    
  Future ft = runtime->execute_task(ctx, launcher);
			  
#ifdef DEBUG
  ft.get_void_result();
  std::cout << "Waiting for node_launch task ..."
	    << std::endl;
#endif
}


void add_subtree_regions 
(LaunchNodeTask &launcher, FSTreeNode *unode, FSTreeNode *vnode){

  int n0 = launcher.region_requirements.size();
  assert( n0 == 0 );
  std::cout << "# of u, v, k regions : ";

  add_umat_regions( launcher, unode );

  n0 = launcher.region_requirements.size();
  std::cout << n0;

  add_vmat_regions( launcher, vnode );

  int n1 = launcher.region_requirements.size();
  std::cout << ", " << n1;
  
  add_kmat_regions( launcher, vnode );

  int n2 = launcher.region_requirements.size();
  std::cout << ", " << n2 << std::endl;
}


void add_umat_regions 
(LaunchNodeTask &launcher, FSTreeNode *unode) {
    
  if (unode->is_legion_leaf()) {
    launcher.add_region_requirement(
	       RegionRequirement(unode->lowrank_matrix->data,
				 READ_WRITE,
				 EXCLUSIVE,
				 unode->lowrank_matrix->data)
				    );
  }
  else {
    add_umat_regions(launcher, unode->lchild);
    add_umat_regions(launcher, unode->rchild);
  }
}


void add_vmat_regions 
(LaunchNodeTask &launcher, FSTreeNode *vnode) {
    
  if ( vnode->Hmat != NULL ) {
    add_hmat_regions(launcher, vnode->Hmat);
  }

  if ( vnode->is_legion_leaf() ) {
    launcher.add_region_requirement(
	       RegionRequirement(vnode->lowrank_matrix->data,
				 READ_ONLY,
				 EXCLUSIVE,
				 vnode->lowrank_matrix->data)
				    );
  }
  else {
    // recursive call
    add_vmat_regions(launcher, vnode->lchild);
    add_vmat_regions(launcher, vnode->rchild);
  }
}


// Hmat has the same structure as umat, except
//  the region privilege is READ_ONLY
void add_hmat_regions 
(LaunchNodeTask &launcher, FSTreeNode *hnode) {
    
  if ( hnode->is_real_leaf() ) {
    launcher.add_region_requirement(
	       RegionRequirement(hnode->lowrank_matrix->data,
				 READ_ONLY,
				 EXCLUSIVE,
				 hnode->lowrank_matrix->data)
				    );
  }
  else {
    add_hmat_regions(launcher, hnode->lchild);
    add_hmat_regions(launcher, hnode->rchild);
  }
}


void add_kmat_regions 
(LaunchNodeTask &launcher, FSTreeNode *vnode) {
    
  if ( vnode->is_legion_leaf() ) {
    assert( vnode->dense_matrix != NULL );
      
    launcher.add_region_requirement(
	       RegionRequirement(vnode->dense_matrix->data,
				 READ_ONLY,
				 EXCLUSIVE,
				 vnode->dense_matrix->data)
				    );
  }
  else {
    // recursive call
    add_kmat_regions(launcher, vnode->lchild);
    add_kmat_regions(launcher, vnode->rchild);
  }
}



/* ---- NodeLaunchTask implementation ---- */

/*static*/
int LaunchNodeTask::TASKID;

LaunchNodeTask::
LaunchNodeTask(TaskArgument arg,
	       Predicate pred /*= Predicate::TRUE_PRED*/,
	       MapperID id /*= 0*/,
	       MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

/*static*/
void LaunchNodeTask::register_tasks(void)
{
  TASKID =
    HighLevelRuntime::register_legion_task
    <LaunchNodeTask::cpu_task>(AUTO_GENERATE_ID,
			       Processor::LOC_PROC, 
			       true,
			       true,
			       AUTO_GENERATE_ID,
			       TaskConfigOptions(false),
			       "launch_node");
#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : Launch_Node_Task\n", TASKID);
#endif
}


void LaunchNodeTask::
cpu_task(const Task *task,
	 const std::vector<PhysicalRegion> &regions,
	 Context ctx, HighLevelRuntime *runtime)
{
  //assert(regions.size() == 0);
  //assert(task->regions.size() == 0);

  typedef TaskArgs<MAX_TREE_SIZE> ARGT;
  ARGT *args = (ARGT *)task->args;
  
  // extract the arguments
  Range taskTag = args->taskTag;
  int treeSize = args->treeSize;
  FSTreeNode *treeArray = args->treeArray;
  assert(task->arglen == sizeof(ARGT));

  // recover the tree structure
  FSTreeNode *vroot = treeArray;
  FSTreeNode *uroot = &treeArray[treeSize];
  array_to_tree(treeArray, 0);
  array_to_tree(treeArray+treeSize, 0);

  solve_bfs(uroot, vroot, taskTag, ctx, runtime);
}

  
void solve_bfs
(FSTreeNode *uroot, FSTreeNode *vroot,
 Range mappingTag, Context ctx, HighLevelRuntime *runtime) {

  std::list<FSTreeNode *> ulist;
  std::list<FSTreeNode *> vlist;
  ulist.push_back(uroot);
  vlist.push_back(vroot);
  typedef std::list<FSTreeNode *>::iterator         Titer;
  typedef std::list<FSTreeNode *>::reverse_iterator RTiter;

  std::list<Range> rglist;
  rglist.push_back(mappingTag);
  typedef std::list<Range>::iterator         Riter;
  typedef std::list<Range>::reverse_iterator RRiter;

  Titer uit = ulist.begin();
  Titer vit = vlist.begin();
  Riter rit = rglist.begin();
  for (; uit != ulist.end(); uit++, vit++, rit++) {
    Range rglchild = rit->lchild();
    Range rgrchild = rit->rchild();
    FSTreeNode *ulchild = (*uit)->lchild;
    FSTreeNode *urchild = (*uit)->rchild;
    FSTreeNode *vlchild = (*vit)->lchild;
    FSTreeNode *vrchild = (*vit)->rchild;
    if (      ! (*uit)->is_legion_leaf() ) {
      assert( ! (*vit)->is_legion_leaf() );
      ulist.push_back( ulchild );
      ulist.push_back( urchild );
      vlist.push_back( vlchild );
      vlist.push_back( vrchild );
      rglist.push_back( rglchild );
      rglist.push_back( rgrchild );
    }
  }
  RTiter ruit  = ulist.rbegin();
  RTiter rvit  = vlist.rbegin();
  RRiter rrgit = rglist.rbegin();

  std::cout << "ulist size: " << ulist.size() << std::endl;


    
  double tRed = 0, tCreate = 0, tBroad = 0;
  for (; ruit != ulist.rend(); ruit++, rvit++, rrgit++)
    visit(*ruit, *rvit, *rrgit,
	  tRed, tBroad, tCreate,
	  ctx, runtime);
  std::cout << "launch reduction task: " << tRed   << std::endl
	    << "launch create task: " << tCreate   << std::endl
	    << "launch broadcast task: " << tBroad << std::endl;
}


void visit
(FSTreeNode *unode, FSTreeNode *vnode, const Range mappingTag,
 double& tRed, double& tBroad, double& tCreate,
 Context ctx, HighLevelRuntime *runtime)
{
  
  if (      unode->is_legion_leaf() ) {
    assert( vnode->is_legion_leaf() );
    solve_legion_leaf(unode, vnode, mappingTag, ctx, runtime);
    return;
  }

  FSTreeNode * b0 = unode->lchild;
  FSTreeNode * b1 = unode->rchild;  
  FSTreeNode * V0 = vnode->lchild;
  FSTreeNode * V1 = vnode->rchild;

  const Range mappingTag0 = mappingTag.lchild();
  const Range mappingTag1 = mappingTag.rchild();

  assert( ! unode->is_legion_leaf() );
  assert( V0->Hmat != NULL );
  assert( V1->Hmat != NULL );

  // This involves a reduction for V0Tu0, V0Td0, V1Tu1, V1Td1
  // from leaves to root in the H tree.
  //LogicalRegion V0Tu0, V0Td0, V1Tu1, V1Td1;
  LMatrix *V0Tu0 = 0;
  LMatrix *V0Td0 = 0;
  LMatrix *V1Tu1 = 0;
  LMatrix *V1Td1 = 0;
  Range ru0(b0->col_beg, b0->ncol);
  Range ru1(b1->col_beg, b1->ncol);
  Range rd0(0,           b0->col_beg);
  Range rd1(0,           b1->col_beg);

  double t0 = timer();
  gemm_reduce(1., V0->Hmat, b0, ru0, 0., V0Tu0,
	      mappingTag0, tCreate, ctx, runtime);
  gemm_reduce(1., V1->Hmat, b1, ru1, 0., V1Tu1,
	      mappingTag1, tCreate, ctx, runtime);
  gemm_reduce(1., V0->Hmat, b0, rd0, 0., V0Td0,
	      mappingTag0, tCreate, ctx, runtime);
  gemm_reduce(1., V1->Hmat, b1, rd1, 0., V1Td1,
	      mappingTag1, tCreate, ctx, runtime);
  tRed += timer() - t0;
  
  // V0Td0 and V1Td1 contain the solution on output.
  // eta0 = V1Td1
  // eta1 = V0Td0
  solve_node_matrix(V0Tu0, V1Tu1,
		    V0Td0, V1Td1,
		    mappingTag0, ctx, runtime);  

  // This step requires a broadcast of V0Td0 and V1Td1
  // from root to leaves.
  // Assemble x from d0 and d1: merge two trees

  double t1 = timer();
  gemm_broadcast(-1., b0, ru0, V1Td1, 1., b0, rd0,
		 mappingTag0, ctx, runtime);
  gemm_broadcast(-1., b1, ru1, V0Td0, 1., b1, rd1,
		 mappingTag1, ctx, runtime);
  tBroad += timer() - t1;
}


void register_launch_node_task() {
  LaunchNodeTask::register_tasks();
}


void FastSolver::solve_dfs
(HodlrMatrix &matrix, int nProc,
 Context ctx, HighLevelRuntime *runtime)
{
  std::cout << "Launch tasks in depth first order."
	    << std::endl;
    
  Range taskTag(nProc);
  double t0 = timer();
  solve_dfs(matrix.uroot, matrix.vroot, taskTag, ctx, runtime);
  double t1 = timer();
  this->time_launcher = t1 - t0;
}


void FastSolver::solve_dfs
(FSTreeNode * unode, FSTreeNode * vnode,
 Range taskTag, Context ctx, HighLevelRuntime *runtime) {

  /*
  if (      unode->is_legion_leaf() ) {
    assert( vnode->is_legion_leaf() );
    solve_legion_leaf(unode, vnode, taskTag, ctx, runtime);
    return;
  }

  Range tag0 = taskTag.lchild();
  Range tag1 = taskTag.rchild();
  
  FSTreeNode * b0 = unode->lchild;
  FSTreeNode * b1 = unode->rchild;
  FSTreeNode * V0 = vnode->lchild;
  FSTreeNode * V1 = vnode->rchild;

  solve_dfs(b0, V0, tag0, ctx, runtime);
  solve_dfs(b1, V1, tag1, ctx, runtime);
  
  assert( !unode->is_legion_leaf() );
  assert( V0->Hmat != NULL );
  assert( V1->Hmat != NULL );
  
  // This involves a reduction for V0Tu0, V0Td0, V1Tu1, V1Td1
  // from leaves to root in the H tree.
  LMatrix *V0Tu0 = 0;
  LMatrix *V0Td0 = 0;
  LMatrix *V1Tu1 = 0;
  LMatrix *V1Td1 = 0;
  Range ru0(b0->col_beg, b0->ncol   );
  Range ru1(b1->col_beg, b1->ncol   );
  Range rd0(0,           b0->col_beg);
  Range rd1(0,           b1->col_beg);


#ifdef DEBUG_GEMM
  const char *gemmBf = "debug_umat.txt";
  if (remove(gemmBf) == 0)
    std::cout << "Remove file: " << gemmBf << std::endl;
  save_HodlrMatrix(unode, gemmBf, ctx, runtime);
  std::cout << "Create file: " << gemmBf << std::endl;
#endif

  
  gemm_reduce(1., V0->Hmat, b0, ru0, 0., V0Tu0, tag0, ctx, runtime);
  gemm_reduce(1., V1->Hmat, b1, ru1, 0., V1Tu1, tag1, ctx, runtime);
  gemm_reduce(1., V0->Hmat, b0, rd0, 0., V0Td0, tag0, ctx, runtime);
  gemm_reduce(1., V1->Hmat, b1, rd1, 0., V1Td1, tag1, ctx, runtime);

  
#if defined(DEBUG_NODE_SOLVE) || defined(DEBUG_GEMM)
  const char *nodeSolveBf = "debug_v0td0_bf.txt";
  if (remove(nodeSolveBf) == 0)
    std::cout << "Remove file: " << nodeSolveBf << std::endl;
  save_LMatrix(V0Td0, nodeSolveBf, ctx, runtime);
  std::cout << "Create file: " << nodeSolveBf << std::endl;

  const char *nodeSolveBf3 = "debug_v0tu0_bf.txt";
  if (remove(nodeSolveBf3) == 0)
    std::cout << "Remove file: " << nodeSolveBf3 << std::endl;
  save_LMatrix(V0Tu0, nodeSolveBf3, ctx, runtime);
  std::cout << "Create file: " << nodeSolveBf3 << std::endl;

#endif
  
    
  // V0Td0 and V1Td1 contain the solution on output.
  // eta0 = V1Td1, eta1 = V0Td0.
  solve_node_matrix(V0Tu0, V1Tu1,
		    V0Td0, V1Td1,
		    taskTag, ctx, runtime);


#ifdef DEBUG_NODE_SOLVE
  const char *nodeSolveAf = "debug_v0td0_af.txt";
  if (remove(nodeSolveAf) == 0)
    std::cout << "Remove file: " << nodeSolveAf << std::endl;
  save_LMatrix(V0Td0, nodeSolveAf, ctx, runtime);
  std::cout << "Create file: " << nodeSolveAf << std::endl;
#endif


  // This step requires a broadcast of V0Td0 and V1Td1
  // from root to leaves.
  // Assemble x from d0 and d1: merge two trees
  gemm_broadcast(-1., b0, ru0, V1Td1, 1., b0, rd0, tag0, ctx, runtime);
  gemm_broadcast(-1., b1, ru1, V0Td0, 1., b1, rd1, tag1, ctx,
  // runtime);
  */
}
