#include <algorithm>
#include <assert.h>
#include <list>
#include <iomanip>

#include "fastSolver.h"
#include "solverTasks.h"
#include "gemm.h"
#include "lapack_blas.h"
#include "macros.h"

void register_solver_tasks() {
  register_solver_operators();  
  register_gemm_tasks();
  register_Htree_tasks();
}


FastSolver::FastSolver(Context ctx, HighLevelRuntime *runtime) {
  this -> ctx     = ctx;
  this -> runtime = runtime;
}


void FastSolver::recLU_solve(LR_Matrix &lr_mat, int tag_size) {
  Range tag = {0, tag_size};
  recLU_solve(lr_mat.uroot, lr_mat.vroot, tag);
  //recLU_solve_bfs(lr_mat.uroot, lr_mat.vroot, tag);
}


void FastSolver::recLU_solve_bfs(FSTreeNode * uroot, FSTreeNode *
vroot, Range mappingTag) {

  std::list<FSTreeNode *> ulist;
  std::list<FSTreeNode *> vlist;
  ulist.push_back(uroot);
  vlist.push_back(vroot);
  
  std::list<Range> rangeList;
  rangeList.push_back(mappingTag);

  typedef std::list<FSTreeNode *>::iterator ITER;
  typedef std::list<FSTreeNode *>::reverse_iterator RITER;
  typedef std::list<Range>::reverse_iterator RRITER;
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
    visit(*ruit, *rvit, *rrit);

}

void FastSolver::visit(FSTreeNode *unode, FSTreeNode *vnode, Range mappingTag) {
  
  if (unode->isLegionLeaf) {
    assert(vnode->isLegionLeaf);
    solve_legion_leaf(unode, vnode, mappingTag);
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
  gemm(1., V0->Hmat, b0, ru0, 0., V0Tu0, mappingTag0, ctx, runtime);
  gemm(1., V1->Hmat, b1, ru1, 0., V1Tu1, mappingTag1, ctx, runtime);
  gemm(1., V0->Hmat, b0, rd0, 0., V0Td0, mappingTag0, ctx, runtime);
  gemm(1., V1->Hmat, b1, rd1, 0., V1Td1, mappingTag1, ctx, runtime);

  // V0Td0 and V1Td1 contain the solution on output.
  // eta0 = V1Td1
  // eta1 = V0Td0
  solve_node_matrix(V0Tu0, V1Tu1, V0Td0, V1Td1, mappingTag0, ctx, runtime);  

  // This step requires a broadcast of V0Td0 and V1Td1 from root to leaves.
  // Assemble x from d0 and d1: merge two trees
  gemm2(-1., b0, ru0, V1Td1, 1., b0, rd0, mappingTag0, ctx, runtime);
  gemm2(-1., b1, ru1, V0Td0, 1., b1, rd1, mappingTag1, ctx, runtime);
}


void FastSolver::recLU_solve(FSTreeNode * unode, FSTreeNode * vnode,
			     Range tag) {

  if (unode->isLegionLeaf) {

    assert(vnode->isLegionLeaf);

    //save_region(unode, "UUmat.txt", ctx, runtime);
	
    // pick a task tag id from tag_beg to tag_end.
    // here the first tag is picked.
    solve_legion_leaf(unode, vnode, tag); 

    //save_region(unode, "Umat.txt", ctx, runtime);
    
    return;
  }

  int   half = tag.size/2;
  Range tag0 = {tag.begin,      half};
  Range tag1 = {tag.begin+half, half};
  
  FSTreeNode * b0 = unode->lchild;
  FSTreeNode * b1 = unode->rchild;
  FSTreeNode * V0 = vnode->lchild;
  FSTreeNode * V1 = vnode->rchild;

  recLU_solve(b0, V0, tag0);
  recLU_solve(b1, V1, tag1);

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
  gemm(1., V0->Hmat, b0, ru0, 0., V0Tu0, tag0, ctx, runtime);
  gemm(1., V1->Hmat, b1, ru1, 0., V1Tu1, tag1, ctx, runtime);
  gemm(1., V0->Hmat, b0, rd0, 0., V0Td0, tag0, ctx, runtime);
  gemm(1., V1->Hmat, b1, rd1, 0., V1Td1, tag1, ctx, runtime);

  // V0Td0 and V1Td1 contain the solution on output.
  // eta0 = V1Td1, eta1 = V0Td0.
  solve_node_matrix(V0Tu0, V1Tu1, V0Td0, V1Td1, tag, ctx, runtime);

  // This step requires a broadcast of V0Td0 and V1Td1 from root to leaves.
  // Assemble x from d0 and d1: merge two trees
  gemm2(-1., b0, ru0, V1Td1, 1., b0, rd0, tag0, ctx, runtime);
  gemm2(-1., b1, ru1, V0Td0, 1., b1, rd1, tag1, ctx, runtime);
}


// this function launches leaf tasks
void FastSolver::solve_legion_leaf(FSTreeNode * uleaf, FSTreeNode * vleaf) {

  int nleaf = count_leaf(uleaf);
  //assert(nleaf == nleaf_per_node);
  //int max_tree_size = nleaf_per_node * 2;
  int max_tree_size = nleaf * 2;
  FSTreeNode arg[max_tree_size*2];

  arg[0] = *vleaf;
  int tree_size = tree_to_array(vleaf, arg, 0);
  //std::cout << "Tree size: " << tree_size << std::endl;
  assert(tree_size < max_tree_size);

  arg[max_tree_size] = *uleaf;
  tree_to_array(uleaf, arg, 0, max_tree_size);

  // encode the array size
  arg[0].col_beg = max_tree_size;

  LeafSolveTask launcher(
		  TaskArgument(
		    &arg[0],
		    sizeof(FSTreeNode)*(max_tree_size*2))
		  );

  launcher.add_region_requirement(
    RegionRequirement(uleaf->matrix->data,
		      READ_WRITE,
		      EXCLUSIVE,
		      uleaf->matrix->data)); // u region
  launcher.add_region_requirement(
    RegionRequirement(vleaf->matrix->data,
		      READ_ONLY,
		      EXCLUSIVE,
		      vleaf->matrix->data)); // v region
  launcher.add_region_requirement(
    RegionRequirement(vleaf->kmat->data,
		      READ_ONLY,
		      EXCLUSIVE,
		      vleaf->kmat->data)); // k region
  launcher.region_requirements[0].add_field(FID_X);
  launcher.region_requirements[1].add_field(FID_X);
  launcher.region_requirements[2].add_field(FID_X);    
  runtime->execute_task(ctx, launcher);
}



// this function launches leaf tasks
void FastSolver::solve_legion_leaf(FSTreeNode * uleaf, FSTreeNode *
				   vleaf, Range task_tag) {
  
  int nleaf = count_leaf(uleaf);
  //assert(nleaf == nleaf_per_node);
  //int max_tree_size = nleaf_per_node * 2;
  int max_tree_size = nleaf * 2;
  FSTreeNode arg[max_tree_size*2+2];

  arg[0] = *vleaf;
  int tree_size = tree_to_array(vleaf, arg, 0);
  //std::cout << "Tree size: " << tree_size << std::endl;
  assert(tree_size < max_tree_size);

  arg[max_tree_size] = *uleaf;
  tree_to_array(uleaf, arg, 0, max_tree_size);

  // encode the array size
  arg[0].col_beg = max_tree_size;

  
  LeafSolveTask launcher(
		  TaskArgument(
		    &arg[0],
		    sizeof(FSTreeNode)*(max_tree_size*2)),
		  Predicate::TRUE_PRED,
		  0,
		  task_tag.begin
		  );

  launcher.add_region_requirement(
    RegionRequirement(uleaf->matrix->data,
		      READ_WRITE,
		      EXCLUSIVE,
		      uleaf->matrix->data)); // u region
  launcher.add_region_requirement(
    RegionRequirement(vleaf->matrix->data,
		      READ_ONLY,
		      EXCLUSIVE,
		      vleaf->matrix->data)); // v region
  launcher.add_region_requirement(
    RegionRequirement(vleaf->kmat->data,
		      READ_ONLY,
		      EXCLUSIVE,
		      vleaf->kmat->data)); // k region
  launcher.region_requirements[0].add_field(FID_X);
  launcher.region_requirements[1].add_field(FID_X);
  launcher.region_requirements[2].add_field(FID_X);    
  runtime->execute_task(ctx, launcher);
}


void save_matrix(double *A, int nRows, int nCols, int LD, std::string filename) {

  std::ofstream outputFile(filename.c_str(), std::ios_base::app);
  if (outputFile.is_open()){
    outputFile<<nRows<<std::endl;
    outputFile<<nCols<<std::endl;
    for (int i = 0; i < nRows ;i++) {
      for (int j = 0; j< nCols ;j++) {
	//outputFile<<A[i+j*nRows]<<'\t';
	outputFile<<A[i+j*LD]<<'\t';
      }
      outputFile<<std::endl;
    }
  }
  outputFile.close();
}


void solve_node_matrix(LogicalRegion & V0Tu0, LogicalRegion & V1Tu1,
		       LogicalRegion & V0Td0, LogicalRegion & V1Td1,
		       Context ctx, HighLevelRuntime *runtime) {

  // this task can be indexed by any tag in the range.
  // the first tag is picked here.
  LUSolveTask launcher(TaskArgument(NULL, 0));
    
  launcher.add_region_requirement(RegionRequirement(V0Tu0, READ_ONLY,
						    EXCLUSIVE, V0Tu0));
  launcher.add_region_requirement(RegionRequirement(V1Tu1, READ_ONLY,
						    EXCLUSIVE, V1Tu1));
  launcher.add_region_requirement(RegionRequirement(V0Td0, READ_WRITE,
						    EXCLUSIVE, V0Td0));
  launcher.add_region_requirement(RegionRequirement(V1Td1, READ_WRITE,
						    EXCLUSIVE, V1Td1));
  
  launcher.region_requirements[0].add_field(FID_X);
  launcher.region_requirements[1].add_field(FID_X);
  launcher.region_requirements[2].add_field(FID_X);
  launcher.region_requirements[3].add_field(FID_X);
  
  runtime->execute_task(ctx, launcher);
}



void
solve_node_matrix(LogicalRegion & V0Tu0, LogicalRegion & V1Tu1,
		  LogicalRegion & V0Td0, LogicalRegion & V1Td1,
		  Range task_tag, Context ctx,
		  HighLevelRuntime *runtime) {

  // this task can be indexed by any tag in the range.
  // the first tag is picked here.
  LUSolveTask launcher(TaskArgument(NULL, 0),
		       Predicate::TRUE_PRED,
		       0,
		       task_tag.begin);
    
  launcher.add_region_requirement(RegionRequirement(V0Tu0,
						    READ_ONLY,
						    EXCLUSIVE,
						    V0Tu0));
  launcher.add_region_requirement(RegionRequirement(V1Tu1,
						    READ_ONLY,
						    EXCLUSIVE,
						    V1Tu1));
  launcher.add_region_requirement(RegionRequirement(V0Td0,
						    READ_WRITE,
						    EXCLUSIVE,
						    V0Td0));
  launcher.add_region_requirement(RegionRequirement(V1Td1,
						    READ_WRITE,
						    EXCLUSIVE,
						    V1Td1));
  
  launcher.region_requirements[0].add_field(FID_X);
  launcher.region_requirements[1].add_field(FID_X);
  launcher.region_requirements[2].add_field(FID_X);
  launcher.region_requirements[3].add_field(FID_X);

  runtime->execute_task(ctx, launcher);
}

