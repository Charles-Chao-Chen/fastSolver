#ifndef _LEGION_TREE_
#define _LEGION_TREE_

#include <string>
#include <fstream>

#include "legion.h"


using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

#define MAX_TREE_SIZE 15 // used in InitCirculantKmatTask::TaskArgs


void register_Htree_tasks();


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

/*
struct CirKmatArg {
  int row_beg;
  int r;
  double diag;
};
*/

class LeafData {

public:
  LeafData(): cols(0), rows(0), data(LogicalRegion::NO_REGION) {}
  //~LeafData();

  void
    set_circulant_matrix_data(int col_beg, int row_beg, int r, Range tag,
			      Context ctx, HighLevelRuntime *runtime);

  //void set_circulant_kmat(CirKmatArg arg, Range tag,
  //		       Context ctx, HighLevelRuntime *runtime);

  int cols;
  int rows;  
  //IndexSpace iSpace;
  //FieldSpace fSpace;
  LogicalRegion data; // Region storing the actual data
    
  // Data at leaf nodes is stored in column major fashion.
  // This allows extracting a given column range.
};


// U and V have the same row structure
struct FSTreeNode {

  FSTreeNode(int nrow=0,
	     int ncol=0,
	     int row_beg=0,
	     int col_beg=0,
	     FSTreeNode *lchild=NULL,
	     FSTreeNode *rchild=NULL,
	     FSTreeNode *Hmat=NULL,
	     LeafData *matrix=NULL,
	     LeafData *kmat=NULL,
	     bool isLegionLeaf=false);
  
  int row_beg; // begin index in the region
  int col_beg;
  int nrow;    
  int ncol;

  FSTreeNode *lchild;
  FSTreeNode *rchild;
  FSTreeNode *Hmat;
  
  LeafData *matrix; // low rank blocks
  LeafData *kmat;   // dense blocks
    
  bool isLegionLeaf;
};


class LR_Matrix {

 public:
  // LR_Matrix() {}

  void create_tree(int, int, int, int, int,
		   Context, HighLevelRuntime *);
  void init_right_hand_side(int, int, Context, HighLevelRuntime *);
  void init_circulant_matrix(double, int, Context, HighLevelRuntime *);

  int get_num_rhs() {return rhs_cols;}
  
  /* --- tree root --- */
  int nleaf_per_node;
  FSTreeNode *uroot;
  FSTreeNode *vroot;

 private:

  /* --- create tree --- */
  void create_legion_leaf(int nleaf_per_legion_node,
			  Context, HighLevelRuntime *);  
  int  create_legion_leaf(FSTreeNode *, int, int &,
			  Context, HighLevelRuntime *);
  void create_matrix_region(FSTreeNode *,
			    Context, HighLevelRuntime *);
  void create_vnode_from_unode(FSTreeNode *, FSTreeNode *,
			       Context, HighLevelRuntime *);

  /* --- populate data --- */

  void init_RHS(FSTreeNode *node, int rand_seed, Range tag,
		Context, HighLevelRuntime *,
		int row_beg = 0);
  void init_Umat(FSTreeNode *node, Range tag,
		 Context, HighLevelRuntime *, int row_beg = 0);
  void init_Vmat(FSTreeNode *node, double diag, Range tag,
		 Context, HighLevelRuntime *, int row_beg = 0);

    
  /*--- helper functions ---*/

  void create_legion_matrix(FSTreeNode *node,
			    Context, HighLevelRuntime *);
  void create_Hmatrix(FSTreeNode *, FSTreeNode *, int,
		      Context, HighLevelRuntime *);
  void set_circulant_Hmatrix_data(FSTreeNode * Hmat,
				  Range tag,
				  Context, HighLevelRuntime *,
				  int row_beg);
  
  
  /* --- private attributes --- */
  int r; // only if every block has the same rank
  int rhs_rows;
  int rhs_cols;
};


void create_matrix(LogicalRegion &, int, int, Context,
		   HighLevelRuntime *);


void register_save_task();
void register_circulant_matrix_task();
void register_circulant_kmat_task();

void create_balanced_tree(FSTreeNode *node, int r, int threshold);

void set_row_begin_index(FSTreeNode *, int);
int  count_column_size(FSTreeNode *, int);
int  max_row_size(FSTreeNode *);


/*--- for debugging purpose ---*/
void print_legion_tree(FSTreeNode *);


void circulant_matrix_task(const Task *task, const std::vector<PhysicalRegion> &regions,
	       Context ctx, HighLevelRuntime *runtime);

void circulant_kmat_task(const Task *task, const std::vector<PhysicalRegion> &regions,
	       Context ctx, HighLevelRuntime *runtime);


int  tree_to_array(FSTreeNode *, FSTreeNode *, int);
void tree_to_array(FSTreeNode *, FSTreeNode *, int, int);
void array_to_tree(FSTreeNode *arg, int idx);
void array_to_tree(FSTreeNode *arg, int idx, int shift);


void init_circulant_Kmat(FSTreeNode *V_legion_leaf, int row_beg_glo, int rank,
			 double diag, Range mapping_tag, Context ctx,
			 HighLevelRuntime *runtime);

int count_leaf(FSTreeNode *node);

void fill_circulant_Kmat(FSTreeNode * vnode, int, int r, double diag,
			 double *Kmat, int LD);

#endif // _LEGION_TREE_
