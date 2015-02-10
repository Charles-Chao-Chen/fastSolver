#include <algorithm>
#include <assert.h>
#include <list>
#include "fast_solver.h"
#include "solver_tasks.h"
#include "gemm.h"
#include "zero_matrix_task.h"
#include "init_matrix_tasks.h"
#include "save_region_task.h"
#include "lapack_blas.h"
#include "timer.h"
#include "macros.h"
#include "unistd.h"

void register_solver_tasks() {

  char hostname[1024];
  gethostname(hostname, 1024);
  std::cout << "Registering all solver tasks on "
	    << hostname
	    << std::endl;
  register_solver_operators();  
  register_gemm_tasks();
  register_zero_matrix_task();
  register_init_tasks();
  register_save_region_task();
  std::cout << std::endl;
}


FastSolver::FastSolver():
  time_launcher(-1) {}


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


void FastSolver::solve_bfs
(HodlrMatrix &lr_mat, int nProc,
 Context ctx, HighLevelRuntime *runtime)
{
  std::cout << "Launch tasks in breadth first order."
	    << std::endl;

  Range tag(nProc);
  double t0 = timer();
  solve_bfs(lr_mat.uroot, lr_mat.vroot, tag, ctx, runtime);
  double t1 = timer();
  this->time_launcher = t1 - t0;
}


void FastSolver::solve_bfs
(FSTreeNode * uroot, FSTreeNode *vroot,
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

  double tRed = 0, tCreate = 0, tBroad = 0;
  for (; ruit != ulist.rend(); ruit++, rvit++, rrgit++)
    visit(*ruit, *rvit, *rrgit,
	  tRed, tBroad, tCreate,
	  ctx, runtime);
  std::cout << "launch reduction task: " << tRed   << std::endl
	    << "launch create task: " << tCreate   << std::endl
	    << "launch broadcast task: " << tBroad << std::endl;
}


void FastSolver::visit
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
