#ifndef __LEGIONTREE_
#define __LEGIONTREE_

#include <string>
#include <fstream>

#include "legion.h"


using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

#define MAX_TREE_SIZE 15 // used in InitCirculantKmatTask::TaskArgs


enum {
  SAVE_REGION_TASK_ID = 10,
  ZERO_MATRIX_TASK_ID = 20,
  CIRCULANT_MATRIX_TASK_ID = 30,
  CIRCULANT_KMAT_TASK_ID = 40,
};

enum MatrixType {
  UMatrix,
  VMatrix,
};



struct range {
  int col_beg;
  int ncol;
};


class Range {
 public:
  Range lchild();
  Range rchild();
 public:
  int begin;
  int size;
};


struct ColRange {
  int col_beg;
  int ncol;
};


struct CirArg {
  int col_beg;
  int row_beg;
  int r;
};


struct CirKmatArg {
  int row_beg;
  int r;
  double diag;
};


struct LeafData {

  //friend void set_element(double x, LogicalRegion &matrix, Context ctx, HighLevelRuntime *runtime);
  
  //public:
  LeafData() {cols=rows=0; data=LogicalRegion::NO_REGION;}
  
  //LeafData(int nrow_, int ncol_, Context ctx, HighLevelRuntime *runtime);
  //void set_matrix(double x, Context ctx, HighLevelRuntime *runtime);

  //void set_matrix_data(Eigen::MatrixXd &, Context, HighLevelRuntime *, int col_beg = 0);
  void set_matrix_data(double *mat, int rhs_rows, int rhs_cols, Context ctx, HighLevelRuntime *runtime, int row_beg = 0);

  void set_circulant_matrix_data(int col_beg, int row_beg, int r,
				 Context ctx, HighLevelRuntime *runtime);
  void set_circulant_matrix_data(int col_beg, int row_beg, int
				 r, Range tag, Context ctx, HighLevelRuntime *runtime);

  
  void set_circulant_kmat(CirKmatArg arg,
			  Context ctx, HighLevelRuntime *runtime);
  void set_circulant_kmat(CirKmatArg arg, Range tag, Context ctx,
			  HighLevelRuntime *runtime);
  
  
  //int col_beg; // begin index in the region
  //int row_beg;
  int cols;
  int rows;
  LogicalRegion data; // Region storing the actual data
  
  // Data at leaf nodes is stored in column major fashion.
  // This allows extracting a given column range.
};


// U and V have the same row structure
struct FSTreeNode {

  FSTreeNode();
  //FSTreeNode * copy_node();
  
  int row_beg; // begin index in the region
  int col_beg;
  int nrow;    
  int ncol;

  FSTreeNode * lchild;
  FSTreeNode * rchild;
  FSTreeNode * Hmat;
  
  LeafData *matrix;
  LeafData *kmat;
  
  //LogicalRegion matrix;
  //LogicalRegion Kmat;
  
  bool isLegionLeaf;
};


class LR_Matrix {

 public:
  LR_Matrix(int, Context ctx_, HighLevelRuntime *runtime_);
  LR_Matrix(int N, int threshold, int rhs_cols, int r, Context ctx_, HighLevelRuntime *runtime_);
  
  void create_legion_leaf(int nleaf_per_legion_node);

  //void init_RHS(Eigen::MatrixXd &RHS);
  void init_RHS(double *);
  void init_RHS(int, bool wait = false);
  void init_RHS(int, int, bool wait = false);
  
  void init_circulant_matrix(double diag);
  void init_circulant_matrix(double diag, int num_node);
  

  /* --- save solution --- */
  void save_solution(std::string);
  
  /* --- save solution to matrix --- */
  void get_soln_from_region(double *);
  void get_soln_from_region(double *soln, FSTreeNode *node, int row_beg = 0);

  /* --- output V --- */
  void print_Vmat(FSTreeNode *, std::string);

  
  /* --- tree root --- */
  int nleaf_per_node;
  FSTreeNode *uroot;
  FSTreeNode *vroot;

 private:

  /* --- create tree --- */
  int  create_legion_leaf(FSTreeNode *, int, int &);
  void create_matrix_region(FSTreeNode *);
  void create_vnode_from_unode(FSTreeNode *, FSTreeNode *);

  /* --- populate data --- */
 
  void init_RHS(FSTreeNode*, double *, int row_beg = 0);
  void init_RHS(FSTreeNode *, int, bool, int row_beg = 0);
  void init_RHS(FSTreeNode *node, int rand_seed, Range tag,
		bool wait, int row_beg = 0);
  
  void init_Umat(FSTreeNode *node, int row_beg = 0);
  void init_Vmat(FSTreeNode *node, double diag, int row_beg = 0);

  void init_Umat(FSTreeNode *node, Range tag, int row_beg = 0);
  void init_Vmat(FSTreeNode *node, double diag, Range tag, int row_beg = 0);

    
  /*--- helper functions ---*/

  //void create_legion_tree(HODLR_Tree::node *, FSTreeNode *);
  void create_legion_matrix(FSTreeNode *node);
  
  //void create_matrix(LogicalRegion &, int, int);
  void create_Hmatrix(FSTreeNode *, FSTreeNode *, int);
  void set_circulant_Hmatrix_data(FSTreeNode * Hmat, int nrow);
  void set_circulant_Hmatrix_data(FSTreeNode * Hmat, Range
				  tag, int row_beg);
  
  //void init_kmat(HODLR_Tree::node *, FSTreeNode *);
  //void fill_kmat(HODLR_Tree::node *, FSTreeNode *, Eigen::MatrixXd &);

  
  /* --- private attributes --- */
  int r; // only if every block has the same rank
  int rhs_rows;
  int rhs_cols;


  /*--- Legion runtime ---*/
  Context ctx;
  HighLevelRuntime *runtime;
};


//void create_matrix(LogicalRegion &matrix, int nrow, int ncol,
//		   Context ctx, HighLevelRuntime *runtime);

void create_matrix(LogicalRegion &, int, int, Context,
		   HighLevelRuntime *);

void zero_matrix(LogicalRegion &matrix, Context ctx, HighLevelRuntime
		 *runtime);
void zero_matrix(LogicalRegion &matrix, Range tag, Context ctx,
		 HighLevelRuntime *runtime);

//void set_element(double x, LogicalRegion &matrix, Context ctx, HighLevelRuntime *runtime);

void scale_matrix(double beta, LogicalRegion &matrix, Context ctx, HighLevelRuntime *runtime);


void save_region(LogicalRegion & matrix, int col_beg, int ncol, std::string filename, Context ctx, HighLevelRuntime *runtime);

void save_region(LogicalRegion & matrix, std::string filename, Context ctx, HighLevelRuntime *runtime);

void save_region(FSTreeNode * node, ColRange rg, std::string filename,
		 Context ctx, HighLevelRuntime *runtime, bool wait = false);

void save_region(FSTreeNode * node, std::string filename, Context ctx, HighLevelRuntime *runtime);

void register_save_task();
void register_zero_matrix_task();
void register_circulant_matrix_task();
void register_circulant_kmat_task();

void save_task(const Task *task, const std::vector<PhysicalRegion> &regions,
	       Context ctx, HighLevelRuntime *runtime);


void create_default_tree(FSTreeNode *node, int r, int threshold);

void set_row_begin_index(FSTreeNode *, int);
int  count_column_size(FSTreeNode *, int);
int  max_row_size(FSTreeNode *);


/*--- for debugging purpose ---*/
void print_legion_tree(FSTreeNode *);

void save_kmat(FSTreeNode * node, std::string filename, Context ctx, HighLevelRuntime *runtime);

/* --- save solution to matrix --- */
//void get_soln_from_region(Eigen::MatrixXd &, FSTreeNode *, Context ctx, HighLevelRuntime *runtime, int row_beg = 0);


void zero_matrix_task(const Task *task, const std::vector<PhysicalRegion> &regions,
	       Context ctx, HighLevelRuntime *runtime);


void circulant_matrix_task(const Task *task, const std::vector<PhysicalRegion> &regions,
	       Context ctx, HighLevelRuntime *runtime);

void circulant_kmat_task(const Task *task, const std::vector<PhysicalRegion> &regions,
	       Context ctx, HighLevelRuntime *runtime);


int  tree_to_array(FSTreeNode *, FSTreeNode *, int);
void tree_to_array(FSTreeNode *, FSTreeNode *, int, int);
void array_to_tree(FSTreeNode *arg, int idx);
void array_to_tree(FSTreeNode *arg, int idx, int shift);


class SaveRegionTask : public TaskLauncher {
public:
  struct TaskArgs {
    ColRange col_range;
    char filename[25];
  };

  SaveRegionTask(TaskArgument arg,
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


class InitRHSTask : public TaskLauncher {
public:
  struct TaskArgs {
    int rand_seed;
    //char filename[25];
  };

  InitRHSTask(TaskArgument arg,
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


class InitCirculantKmatTask : public TaskLauncher {
public:
  template <int N>
  struct TaskArgs {
    //int treeSize;
    int row_beg_global;
    int rank;
    //int LD; // leading dimension
    double diag;
    FSTreeNode treeArray[N];
  };
  
  InitCirculantKmatTask(TaskArgument arg,
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

void init_circulant_Kmat(FSTreeNode *V_legion_leaf, int row_beg_glo, int rank,
			 double diag, Range mapping_tag, Context ctx,
			 HighLevelRuntime *runtime);

int count_leaf(FSTreeNode *node);

void fill_circulant_Kmat(FSTreeNode * vnode, int, int r, double diag,
double *Kmat, int LD);

#endif // __LEGIONTREE_
