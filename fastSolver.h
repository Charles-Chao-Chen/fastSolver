#ifndef __FASTSOLVER_
#define __FASTSOLVER_


#include "legion.h"
#include "Htree.h"
#include "gemm.h"
#include "utility.h"

#include <string>
#include <fstream>
#include <vector>


using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


enum {
  LEAF_TASK_ID = 1,
  LU_SOLVE_TASK_ID = 3,
};


class FastSolver {

 public:
  FastSolver(Context, HighLevelRuntime *);
  void initialize();

  
  // void init_LR_data(HODLR_Tree::node *, FSTreeNode *, FSTreeNode *, Eigen::MatrixXd &, Eigen::MatrixXd &);

  void recLU_solve();
  void recLU_solve(LR_Matrix &);
  void recLU_solve(FSTreeNode *, FSTreeNode *);
  void recLU_solve(LR_Matrix &lr_mat, int tag_size);
  void recLU_solve(FSTreeNode * unode, FSTreeNode * vnode, Range tag);
  void recLU_solve_bfs(FSTreeNode * uroot, FSTreeNode * vroot);
  void recLU_solve_bfs(FSTreeNode * uroot, FSTreeNode * vroot, Range mappingTag);
  
  //void save_soln_from_region(FSTreeNode *);
  
 private:

  void visit(FSTreeNode *unode, FSTreeNode *vnode);
  void visit(FSTreeNode *unode, FSTreeNode *vnode, Range mappingTag);
  void solve_legion_leaf(FSTreeNode *, FSTreeNode *);

  void solve_legion_leaf(FSTreeNode * uleaf, FSTreeNode * vleaf, Range task_tag);
 
  //int  tree_to_array(FSTreeNode *, std::vector<FSTreeNode> &, int);
  //void tree_to_array(FSTreeNode *, std::vector<FSTreeNode> &, int, int);
  //int  tree_to_array(FSTreeNode *, FSTreeNode *, int);
  //void tree_to_array(FSTreeNode *, FSTreeNode *, int, int);


  
  /*--- private attributes ---*/
  //Eigen::VectorXd soln;
    
  //FSTreeNode *uroot;
  //FSTreeNode *vroot;
  //std::queue<FSTreeNode *> knodes;


  /*--- Legion runtime ---*/
  Context ctx;
  HighLevelRuntime *runtime;
};



class LUSolveTask : public TaskLauncher {
public:

  LUSolveTask(TaskArgument arg,
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





void register_solver_task();


void leaf_task(const Task *task, const std::vector<PhysicalRegion> &regions,
	       Context ctx, HighLevelRuntime *runtime);

void recLU_leaf_solve(FSTreeNode * uroot, FSTreeNode * vroot, double * u_ptr, double * v_ptr, double * k_ptr, int LD);


void save_matrix(double *A, int nRows, int nCols, int LD, std::string filename);

void solve_node_matrix(LogicalRegion &, LogicalRegion &, LogicalRegion &, LogicalRegion &, Context, HighLevelRuntime *);

void solve_node_matrix(LogicalRegion & V0Tu0, LogicalRegion & V1Tu1,
		       LogicalRegion & V0Td0, LogicalRegion & V1Td1,
		       Range task_tag, Context ctx, HighLevelRuntime
		       *runtime);


void lu_solve_task(const Task *, const std::vector<PhysicalRegion> &,
		   Context, HighLevelRuntime *);

//void saveVectorToText(const std::string outputFileName, Eigen::VectorXd & inputVector);



#endif // __FASTSOLVER_
