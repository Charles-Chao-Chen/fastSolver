#ifndef __LEGIONTREE_
#define __LEGIONTREE_


#include "legion.h"
#include "utility.h"

//#include "HODLR_Files/helperFunctions.hpp"
//#include <Eigen/Dense>

#include <string>
#include <fstream>


using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;


enum {
  FID_X,
};

enum {
  SAVE_REGION_TASK_ID = 10,
};

enum MatrixType {
  UMatrix,
  VMatrix,
};


struct range {
  int col_beg;
  int ncol;
};


struct LeafData {

  //friend void set_element(double x, LogicalRegion &matrix, Context ctx, HighLevelRuntime *runtime);
  
  //public:
  LeafData() {cols=rows=0; data=LogicalRegion::NO_REGION;}
  
  //LeafData(int nrow_, int ncol_, Context ctx, HighLevelRuntime *runtime);
  //void set_matrix(double x, Context ctx, HighLevelRuntime *runtime);

  //void set_matrix_data(Eigen::MatrixXd &, Context, HighLevelRuntime *, int col_beg = 0);
  void set_matrix_data(double *mat, int rhs_rows, int rhs_cols, Context ctx, HighLevelRuntime *runtime, int row_beg = 0);

  void set_circulant_matrix_data(int col_beg, int row_beg, int r, Context ctx, HighLevelRuntime *runtime);

    
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

  //void initialize(HODLR_Tree::node *, Eigen::VectorXd &);

  
  void create_legion_leaf(int nleaf_per_legion_node);

  //void init_RHS(Eigen::MatrixXd &RHS);
  void init_RHS(double *);
  
  void init_circulant_matrix(double diag);
  
  

  /* --- save solution --- */
  void save_solution(std::string);
  
  /* --- save solution to matrix --- */
  void get_soln_from_region(double *);
  void get_soln_from_region(double *soln, FSTreeNode *node, int row_beg = 0);
  
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
  /*
  void init_Umat(HODLR_Tree::node *, FSTreeNode *, Eigen::MatrixXd &);
  void init_Vmat(HODLR_Tree::node *, FSTreeNode *, Eigen::MatrixXd &);
  void init_RHS(FSTreeNode*, Eigen::MatrixXd &RHS);
  */
    
  void init_RHS(FSTreeNode*, double *, int row_beg = 0);
 
  void init_Umat(FSTreeNode *node, int row_beg = 0);
  void init_Vmat(FSTreeNode *node, double, int row_beg = 0);
    
  /*--- helper functions ---*/

  //void create_legion_tree(HODLR_Tree::node *, FSTreeNode *);
  void create_legion_matrix(FSTreeNode *node);
  
  //void create_matrix(LogicalRegion &, int, int);
  void create_Hmatrix(FSTreeNode *, FSTreeNode *, int);
  //void set_Hmatrix_data(FSTreeNode *, Eigen::MatrixXd &);
  void set_circulant_Hmatrix_data(FSTreeNode * Hmat, int nrow);
  
  //void accumulate_matrix(Eigen::MatrixXd &, Eigen::MatrixXd &, Eigen::MatrixXd &, Eigen::MatrixXd &, FSTreeNode *, Eigen::MatrixXd &, FSTreeNode *);
  

  //void init_kmat(HODLR_Tree::node *, FSTreeNode *);
  //void fill_kmat(HODLR_Tree::node *, FSTreeNode *, Eigen::MatrixXd &);
  void fill_circulant_kmat(FSTreeNode * vnode, int, int r, double diag, double *Kmat, int LD);

  /* --- output V --- */
  void print_Vmat(FSTreeNode *, std::string);
  
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

void create_matrix(LogicalRegion &, int, int, Context, HighLevelRuntime *);

void set_element(double x, LogicalRegion &matrix, Context ctx, HighLevelRuntime *runtime);

void scale_matrix(double beta, LogicalRegion &matrix, Context ctx, HighLevelRuntime *runtime);


void save_region(LogicalRegion & matrix, int col_beg, int ncol, std::string filename, Context ctx, HighLevelRuntime *runtime);

void save_region(LogicalRegion & matrix, std::string filename, Context ctx, HighLevelRuntime *runtime);

void save_region(FSTreeNode * node, range rg, std::string filename, Context ctx, HighLevelRuntime *runtime);

void save_region(FSTreeNode * node, std::string filename, Context ctx, HighLevelRuntime *runtime);

void register_save_task();

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







#endif // __LEGIONTREE_
