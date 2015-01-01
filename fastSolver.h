#ifndef _FAST_SOLVER
#define _FAST_SOLVER

#include <string>

#include "legion.h"
#include "Htree.h"


void register_solver_tasks();

class FastSolver {

 public:
  FastSolver(Context, HighLevelRuntime *);
  void initialize();

  void recLU_solve();
  void recLU_solve(LR_Matrix &lr_mat, int tag_size);
  void recLU_solve(FSTreeNode * unode, FSTreeNode * vnode, Range tag);
  void recLU_solve_bfs(FSTreeNode * uroot, FSTreeNode * vroot, Range mappingTag);
  
 private:

  void visit(FSTreeNode *unode, FSTreeNode *vnode);
  void visit(FSTreeNode *unode, FSTreeNode *vnode, Range mappingTag);


  /*--- private attributes ---*/
  //Eigen::VectorXd soln;
    
  //FSTreeNode *uroot;
  //FSTreeNode *vroot;
  //std::queue<FSTreeNode *> knodes;


  /*--- Legion runtime ---*/
  Context ctx;
  HighLevelRuntime *runtime;
};


void leaf_task(const Task *task, const std::vector<PhysicalRegion> &regions,
	       Context ctx, HighLevelRuntime *runtime);

void recLU_leaf_solve(FSTreeNode * uroot, FSTreeNode * vroot, double * u_ptr, double * v_ptr, double * k_ptr, int LD);


void save_matrix(double *A, int nRows, int nCols, int LD, std::string filename);


#endif // _FAST_SOLVER
