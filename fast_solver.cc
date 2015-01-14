#include <algorithm>
#include <assert.h>
#include <list>
#include "fast_solver.h"
#include "solver_tasks.h"
#include "gemm.h"
#include "zero_matrix_task.h"
#include "init_matrix_tasks.h"
#include "save_task.h"
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
  register_output_tasks();
  std::cout << std::endl;
}


FastSolver::FastSolver():
  time_launcher(-1) {}


void
FastSolver::solve_dfs(HodlrMatrix &matrix, int tag_size,
		      Context ctx, HighLevelRuntime *runtime)
{
  std::cout << "Traverse tree in depth first order."
	    << std::endl;
  
  /*
  const char *save_file = "Umat.txt";
  if (remove(save_file) == 0)
    std::cout << "Remove file: " << save_file << std::endl;
  save_Htree(matrix.uroot, save_file, ctx, runtime);
*/

    
  Range tag(0, tag_size);
  double t0 = timer();
  solve_dfs(matrix.uroot, matrix.vroot, tag, ctx, runtime);
  double t1 = timer();
  time_launcher = t1 - t0;


  /*
  const char *save_file = "Ufinish.txt";
  if (remove(save_file) == 0)
    std::cout << "Remove file: " << save_file << std::endl;
  save_Htree(matrix.uroot, save_file, ctx, runtime);
  */
}


void
FastSolver::solve_dfs(FSTreeNode * unode, FSTreeNode * vnode,
		      Range tag,
		      Context ctx, HighLevelRuntime *runtime) {

  if (      unode->is_legion_leaf() ) {
    assert( vnode->is_legion_leaf() );

    // pick a task tag id from tag_beg to tag_end.
    // here the first tag is picked.
    //save_Htree(unode, "Uinit.txt", ctx, runtime);
    solve_legion_leaf(unode, vnode, tag, ctx, runtime);

    /*
    const char *save_file = "Umat.txt";
    if (remove(save_file) == 0)
      std::cout << "Remove file: " << save_file << std::endl;
    save_Htree(unode, save_file, ctx, runtime);
    */
    return;
  }

  int   half = tag.size/2;
  Range tag0 = tag.lchild();
  Range tag1 = tag.rchild();
  
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

  /*
  const char *nodeSolveBf1 = "debug_v1td1_bf.txt";
  if (remove(nodeSolveBf1) == 0)
    std::cout << "Remove file: " << nodeSolveBf1 << std::endl;
  save_LMatrix(V1Td1, nodeSolveBf1, ctx, runtime);
  std::cout << "Create file: " << nodeSolveBf1 << std::endl;

  const char *nodeSolveBf2 = "debug_v1tu1_bf.txt";
  if (remove(nodeSolveBf2) == 0)
    std::cout << "Remove file: " << nodeSolveBf2 << std::endl;
  save_LMatrix(V1Tu1, nodeSolveBf2, ctx, runtime);
  std::cout << "Create file: " << nodeSolveBf2 << std::endl;
*/
#endif
  
    
  // V0Td0 and V1Td1 contain the solution on output.
  // eta0 = V1Td1, eta1 = V0Td0.
  solve_node_matrix(V0Tu0, V1Tu1, V0Td0, V1Td1, tag, ctx, runtime);


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
  gemm_broadcast(-1., b1, ru1, V0Td0, 1., b1, rd1, tag1, ctx, runtime);
}


void
FastSolver::solve_bfs(HodlrMatrix &lr_mat, int tag_size,
		      Context ctx, HighLevelRuntime *runtime)
{
  std::cout << "Traverse tree in breadth first order."
	    << std::endl;

  Range tag(0, tag_size);
  solve_bfs(lr_mat.uroot, lr_mat.vroot, tag, ctx, runtime);
}


void
FastSolver::solve_bfs(FSTreeNode * uroot, FSTreeNode *vroot,
		      Range mappingTag,
		      Context ctx, HighLevelRuntime *runtime) {

  std::list<FSTreeNode *> ulist;
  std::list<FSTreeNode *> vlist;
  ulist.push_back(uroot);
  vlist.push_back(vroot);
  
  std::list<Range> rangeList;
  rangeList.push_back(mappingTag);

  typedef std::list<FSTreeNode *>::iterator         ITER;
  typedef std::list<FSTreeNode *>::reverse_iterator RITER;
  typedef std::list<Range>::reverse_iterator        RRITER;
  ITER uit = ulist.begin();
  ITER vit = vlist.begin();
  for (; uit != ulist.end(); uit++, vit++) {
    FSTreeNode *ulchild = (*uit)->lchild;
    FSTreeNode *urchild = (*uit)->rchild;
    FSTreeNode *vlchild = (*vit)->lchild;
    FSTreeNode *vrchild = (*vit)->rchild;
    if (      (*uit)->is_legion_leaf() == false ) {
      assert( (*vit)->is_legion_leaf() == false );
      ulist.push_back(ulchild);
      ulist.push_back(urchild);
      vlist.push_back(vlchild);
      vlist.push_back(vrchild);
      rangeList.push_back(mappingTag.lchild());
      rangeList.push_back(mappingTag.rchild());
    }
  }
  RITER ruit = ulist.rbegin();
  RITER rvit = vlist.rbegin();
  RRITER rrit = rangeList.rbegin();
  for (; ruit != ulist.rend(); ruit++, rvit++, rrit++)
    visit(*ruit, *rvit, *rrit, ctx, runtime);

}


void FastSolver::visit(FSTreeNode *unode, FSTreeNode *vnode,
		       const Range mappingTag,
		       Context ctx, HighLevelRuntime *runtime) {
  
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

  assert( !unode->is_legion_leaf() );
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
  gemm_reduce(1., V0->Hmat, b0, ru0, 0., V0Tu0,
	      mappingTag0, ctx, runtime);
  gemm_reduce(1., V1->Hmat, b1, ru1, 0., V1Tu1,
	      mappingTag1, ctx, runtime);
  gemm_reduce(1., V0->Hmat, b0, rd0, 0., V0Td0,
	      mappingTag0, ctx, runtime);
  gemm_reduce(1., V1->Hmat, b1, rd1, 0., V1Td1,
	      mappingTag1, ctx, runtime);

  // V0Td0 and V1Td1 contain the solution on output.
  // eta0 = V1Td1
  // eta1 = V0Td0
  solve_node_matrix(V0Tu0, V1Tu1, V0Td0, V1Td1, mappingTag0, ctx, runtime);  

  // This step requires a broadcast of V0Td0 and V1Td1 from root to leaves.
  // Assemble x from d0 and d1: merge two trees
  gemm_broadcast(-1., b0, ru0, V1Td1, 1., b0, rd0,
		 mappingTag0, ctx, runtime);
  gemm_broadcast(-1., b1, ru1, V0Td0, 1., b1, rd1,
		 mappingTag1, ctx, runtime);
}
