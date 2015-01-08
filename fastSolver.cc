#include <algorithm>
#include <assert.h>
#include <list>
#include "fastSolver.h"
#include "solverTasks.h"
#include "gemm.h"
#include "initMatrixTasks.h"
#include "saveTask.h"
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
  register_init_tasks();
  register_output_tasks();
  std::cout << std::endl;
}


FastSolver::FastSolver():
  time_launcher(-1) {}


void
FastSolver::solve_dfs(LR_Matrix &matrix, int tag_size,
		      Context ctx, HighLevelRuntime *runtime)
{
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

  if (unode->isLegionLeaf) {
    assert(vnode->isLegionLeaf);

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
  
  assert(unode->isLegionLeaf == false);
  assert(V0->Hmat != NULL);
  assert(V1->Hmat != NULL);


  /*
  const char *save_file0 = "Umat.txt";
  if (remove(save_file0) == 0)
    std::cout << "Remove file: " << save_file0 << std::endl;
  save_Htree(unode, save_file0, ctx, runtime);
  */


  
  // This involves a reduction for V0Tu0, V0Td0, V1Tu1, V1Td1
  // from leaves to root in the H tree.
  LogicalRegion V0Tu0, V0Td0, V1Tu1, V1Td1;
  range ru0 = {b0->col_beg, b0->ncol};
  range ru1 = {b1->col_beg, b1->ncol};
  range rd0 = {0,           b0->col_beg};
  range rd1 = {0,           b1->col_beg};
  gemm_reduce(1., V0->Hmat, b0, ru0, 0., V0Tu0, tag0, ctx, runtime);
  gemm_reduce(1., V1->Hmat, b1, ru1, 0., V1Tu1, tag1, ctx, runtime);
  gemm_reduce(1., V0->Hmat, b0, rd0, 0., V0Td0, tag0, ctx, runtime);
  gemm_reduce(1., V1->Hmat, b1, rd1, 0., V1Td1, tag1, ctx, runtime);


  /*
  const char *save_file1 = "Ufinish.txt";
  if (remove(save_file1) == 0)
    std::cout << "Remove file: " << save_file1 << std::endl;
  save_Htree(unode, save_file1, ctx, runtime);
  */

  
    
  // V0Td0 and V1Td1 contain the solution on output.
  // eta0 = V1Td1, eta1 = V0Td0.
  solve_node_matrix(V0Tu0, V1Tu1, V0Td0, V1Td1, tag, ctx, runtime);

  // This step requires a broadcast of V0Td0 and V1Td1
  // from root to leaves.
  // Assemble x from d0 and d1: merge two trees
  gemm_broadcast(-1., b0, ru0, V1Td1, 1., b0, rd0, tag0, ctx, runtime);
  gemm_broadcast(-1., b1, ru1, V0Td0, 1., b1, rd1, tag1, ctx, runtime);
}


void
FastSolver::solve_bfs(LR_Matrix &lr_mat, int tag_size,
		      Context ctx, HighLevelRuntime *runtime)
{
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
    if ((*uit)->isLegionLeaf == false) {
      assert((*vit)->isLegionLeaf == false);
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
		       Range mappingTag,
		       Context ctx, HighLevelRuntime *runtime) {
  
  if (unode->isLegionLeaf) {
    assert(vnode->isLegionLeaf);
    solve_legion_leaf(unode, vnode, mappingTag, ctx, runtime);
    return;
  }

  FSTreeNode * b0 = unode->lchild;
  FSTreeNode * b1 = unode->rchild;  
  FSTreeNode * V0 = vnode->lchild;
  FSTreeNode * V1 = vnode->rchild;

  Range mappingTag0 = mappingTag.lchild();
  Range mappingTag1 = mappingTag.rchild();

  assert(unode->isLegionLeaf == false);
  assert(V0->Hmat != NULL);
  assert(V1->Hmat != NULL);

  // This involves a reduction for V0Tu0, V0Td0, V1Tu1, V1Td1
  // from leaves to root in the H tree.
  LogicalRegion V0Tu0, V0Td0, V1Tu1, V1Td1;
  range ru0 = {b0->col_beg, b0->ncol};
  range ru1 = {b1->col_beg, b1->ncol};
  range rd0 = {0,           b0->col_beg};
  range rd1 = {0,           b1->col_beg};
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
