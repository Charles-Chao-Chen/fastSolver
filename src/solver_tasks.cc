#include "solver_tasks.h"
#include "htree_helper.h"
#include "lapack_blas.h"
#include "macros.h"

using namespace LegionRuntime::Accessor;


// for debugging purpose
void save_matrix(double *A, int nRows, int nCols, int LD,
		 std::string filename);


namespace {
  
  class LUSolveTask : public TaskLauncher {
  public:

    LUSolveTask(TaskArgument arg,
		Predicate pred = Predicate::TRUE_PRED,
		MapperID id = 0,
		MappingTagID tag = 0);
  
    static int TASKID;

    static void register_tasks(void);

  public:
    static void
    cpu_task(const Task *task,
	     const std::vector<PhysicalRegion> &regions,
	     Context ctx, HighLevelRuntime *runtime);
  };


  class LeafSolveTask : public TaskLauncher {
  public:

    LeafSolveTask(TaskArgument arg,
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
}


// legion task wrapper
void solve_node_matrix
  (LMatrix *(&V0Tu0), LMatrix *(&V1Tu1),
   LMatrix *(&V0Td0), LMatrix *(&V1Td1),
   Range task_tag, Context ctx,
   HighLevelRuntime *runtime) {

  // this task can be indexed by any tag in the range.
  // the first tag is picked here.
  LUSolveTask launcher(TaskArgument(NULL, 0),
		       Predicate::TRUE_PRED,
		       0,
		       task_tag.begin);
    
  launcher.add_region_requirement(RegionRequirement
				  (V0Tu0->data,
				   READ_ONLY,
				   EXCLUSIVE,
				   V0Tu0->data)
				  );
  launcher.add_region_requirement(RegionRequirement
				  (V1Tu1->data,
				   READ_ONLY,
				   EXCLUSIVE,
				   V1Tu1->data)
				  );
  launcher.add_region_requirement(RegionRequirement
				  (V0Td0->data,
				   READ_WRITE,
				   EXCLUSIVE,
				   V0Td0->data)
				  );
  launcher.add_region_requirement(RegionRequirement
				  (V1Td1->data,
				   READ_WRITE,
				   EXCLUSIVE,
				   V1Td1->data)
				  );
  
  launcher.region_requirements[0].add_field(FID_X);
  launcher.region_requirements[1].add_field(FID_X);
  launcher.region_requirements[2].add_field(FID_X);
  launcher.region_requirements[3].add_field(FID_X);

  Future f = runtime->execute_task(ctx, launcher);

#ifdef SERIAL
  std::cout << "Waiting for node_solve task ..." << std::endl;
  f.get_void_result();
#endif
}


/* ---- LUSolveTask implementation ---- */

/*static*/
int LUSolveTask::TASKID;

LUSolveTask::LUSolveTask(
  TaskArgument arg,
  Predicate pred /*= Predicate::TRUE_PRED*/,
  MapperID id /*= 0*/,
  MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

/*static*/
void LUSolveTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <LUSolveTask::cpu_task>(
			    AUTO_GENERATE_ID,
			    Processor::LOC_PROC, 
			    true,
			    true,
			    AUTO_GENERATE_ID,
			    TaskConfigOptions(true/*leaf*/),
			    "LU_Solve");

#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : LU_Solve\n", TASKID);
#endif
}

void
LUSolveTask::cpu_task(const Task *task,
		      const std::vector<PhysicalRegion> &regions,
		      Context ctx, HighLevelRuntime *runtime) {
  
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  assert(task->arglen == 0);
  
  IndexSpace is_V0Tu0 = task->regions[0].region.get_index_space();
  IndexSpace is_V1Tu1 = task->regions[1].region.get_index_space();
  IndexSpace is_V0Td0 = task->regions[2].region.get_index_space();
  IndexSpace is_V1Td1 = task->regions[3].region.get_index_space();

  Domain dom_V0Tu0 = runtime->get_index_space_domain(ctx, is_V0Tu0);
  Domain dom_V1Tu1 = runtime->get_index_space_domain(ctx, is_V1Tu1);
  Domain dom_V0Td0 = runtime->get_index_space_domain(ctx, is_V0Td0);
  Domain dom_V1Td1 = runtime->get_index_space_domain(ctx, is_V1Td1);

  Rect<2> rect_V0Tu0 = dom_V0Tu0.get_rect<2>();
  Rect<2> rect_V1Tu1 = dom_V1Tu1.get_rect<2>();
  Rect<2> rect_V0Td0 = dom_V0Td0.get_rect<2>();
  Rect<2> rect_V1Td1 = dom_V1Td1.get_rect<2>();

  RegionAccessor<AccessorType::Generic, double> acc_V0Tu0 =
    regions[0].get_field_accessor(FID_X).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> acc_V1Tu1 =
    regions[1].get_field_accessor(FID_X).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> acc_V0Td0 =
    regions[2].get_field_accessor(FID_X).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> acc_V1Td1 =
    regions[3].get_field_accessor(FID_X).typeify<double>();
    
  Rect<2> subrect;
  ByteOffset offsets[2];
  
  double *V0Tu0 = acc_V0Tu0.raw_rect_ptr<2>
    (rect_V0Tu0, subrect, offsets);
  assert(rect_V0Tu0 == subrect);

  double *V1Tu1 = acc_V1Tu1.raw_rect_ptr<2>
    (rect_V1Tu1, subrect, offsets);
  assert(rect_V1Tu1 == subrect);
  
  double *V0Td0 = acc_V0Td0.raw_rect_ptr<2>
    (rect_V0Td0, subrect, offsets);
  assert(rect_V0Td0 == subrect);

  double *V1Td1 = acc_V1Td1.raw_rect_ptr<2>
    (rect_V1Td1, subrect, offsets);
  assert(rect_V1Td1 == subrect);


  int V0Tu0_rows = rect_V0Tu0.dim_size(0);
  int V0Tu0_cols = rect_V0Tu0.dim_size(1);
  int V1Tu1_rows = rect_V1Tu1.dim_size(0);
  int V1Tu1_cols = rect_V1Tu1.dim_size(1);
  int V0Td0_rows = rect_V0Td0.dim_size(0);
  int V0Td0_cols = rect_V0Td0.dim_size(1);
  int V1Td1_rows = rect_V1Td1.dim_size(0);
  int V1Td1_cols = rect_V1Td1.dim_size(1);

  assert(V0Td0_cols == V1Td1_cols);
  assert(V0Tu0_rows + V1Tu1_rows == V0Tu0_cols + V1Tu1_cols);
  assert(V0Tu0_rows + V1Tu1_rows == V0Td0_rows + V1Td1_rows);


  /* form the Shur complement:
     --            --
     |  I    V0Tu0  | 
     | V1Tu1  I     |
     --            --
     and the solutions eta0 and eta1 overwrite
     V1Td1 and V0Td0. (Note the reversed order)
  */

  
  // Solve: S * eta0 = V1Td1 - V1Tu1 * V0Td0
  // where S = I - V1Tu1 * V0Tu0
  // Note:  eta0 overwrites V1Td1
  
  char transa  = 'n';
  char transb  = 'n';
  double alpha = -1.;
  double beta  =  1.;


  /*
  const char *save_file0 = "V1Td1.txt";
  if (remove(save_file0) == 0)
    std::cout << "Remove file: " << save_file0 << std::endl;
  save_data(V1Td1, V1Td1_rows, 0, V1Td1_cols, save_file0);
*/
  
  
  assert(V1Tu1_cols == V0Td0_rows);
  assert(V1Td1_rows == V1Tu1_rows);
  blas::dgemm_(&transa, &transb,
	       &V1Tu1_rows, &V0Td0_cols, &V1Tu1_cols,
	       &alpha,      V1Tu1,       &V1Tu1_rows,
	                    V0Td0,       &V0Td0_rows,
	       &beta,       V1Td1,       &V1Td1_rows);



  /*
  const char *save_file3 = "V1Td1_finish.txt";
  if (remove(save_file3) == 0)
    std::cout << "Remove file: " << save_file3 << std::endl;
  save_data(V1Td1, V1Td1_rows, 0, V1Td1_cols, save_file3);
  */



  
  int N = V1Tu1_rows;
  double *S = (double*) calloc( N*N, sizeof(double) );
  // initialize the indentity matrix
  for (int i=0; i<N; i++)
    S[i*(N+1)] = 1.;

  assert(V1Tu1_cols == V0Tu0_rows);
  blas::dgemm_(&transa, &transb,
	       &V1Tu1_rows, &V0Tu0_cols, &V1Tu1_cols,
	       &alpha,      V1Tu1,       &V1Tu1_rows,
	                    V0Tu0,       &V0Tu0_rows,
	       &beta,       S,           &N);



  
  /*
  const char *save_file = "V1Td1.txt";
  if (remove(save_file) == 0)
    std::cout << "Remove file: " << save_file << std::endl;
  save_data(V1Td1, V1Td1_rows, 0, V1Td1_cols, save_file);
  */

  
  int INFO;
  int IPIV[N];
  assert(V0Td0_cols == V1Td1_cols);
  lapack::dgesv_(&N,    &V1Td1_cols,
		 S,     &N,          IPIV,
		 V1Td1, &V1Td1_rows, &INFO);
  assert(INFO == 0);


  /*
  const char *save_file2 = "V1Td1_finish.txt";
  if (remove(save_file2) == 0)
    std::cout << "Remove file: " << save_file2 << std::endl;
  save_data(V1Td1, V1Td1_rows, 0, V1Td1_cols, save_file2);
  */
  

  // Solve: I * eta1 = V0Td0 - V0Tu0 * eta0
  // where no solve happens because of the indenty coefficience
  // Note:  eta1 overwrites V0Td0
  assert(V0Tu0_cols == V1Td1_rows);
  blas::dgemm_(&transa, &transb, &V0Tu0_rows, &V1Td1_cols, &V0Tu0_cols,
	       &alpha,   V0Tu0,  &V0Tu0_rows,
	                 V1Td1,  &V1Td1_rows,
	       &beta,    V0Td0,  &V0Td0_rows);   
  
  free(S);
}


/* ---- LeafSolveTask implementation ---- */

/*static*/
int LeafSolveTask::TASKID;

LeafSolveTask::LeafSolveTask(
  TaskArgument arg,
  Predicate pred /*= Predicate::TRUE_PRED*/,
  MapperID id /*= 0*/,
  MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

/*static*/
void LeafSolveTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <LeafSolveTask::cpu_task>(
			      AUTO_GENERATE_ID,
			      Processor::LOC_PROC, 
			      true,
			      true,
			      AUTO_GENERATE_ID,
			      TaskConfigOptions(true/*leaf*/),
			      "Leaf_Solve");
#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : Leaf_Solve\n", TASKID);
#endif
}

static void serial_leaf_solve
  (FSTreeNode * unode, FSTreeNode * vnode,
   double * u_ptr, double * v_ptr, double * k_ptr, int LD);

void LeafSolveTask::cpu_task
  (const Task *task,
   const std::vector<PhysicalRegion> &regions,
   Context ctx, HighLevelRuntime *runtime) {

  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  FSTreeNode *arg = (FSTreeNode *)task->args;
  
  // decode tree size
  int tree_size = arg[0].col_beg;
  //std::cout << "Tree size: " << tree_size << std::endl;
  arg[0].col_beg = 0;
  assert(task->arglen == sizeof(FSTreeNode)*(tree_size*2));

  FSTreeNode *vroot = arg;
  array_to_tree(arg, 0);

  FSTreeNode *uroot = &arg[tree_size];
  //array_to_tree(arg, 0, tree_size);
  array_to_tree(arg+tree_size, 0);
  
  //print_legion_tree(vroot);
  //print_legion_tree(uroot);
  
   RegionAccessor<AccessorType::Generic, double> acc_u = 
     regions[0].get_field_accessor(FID_X).typeify<double>();
   RegionAccessor<AccessorType::Generic, double> acc_v = 
     regions[1].get_field_accessor(FID_X).typeify<double>();
   RegionAccessor<AccessorType::Generic, double> acc_k = 
     regions[2].get_field_accessor(FID_X).typeify<double>();

   IndexSpace is_u = task->regions[0].region.get_index_space();
   IndexSpace is_v = task->regions[1].region.get_index_space();
   IndexSpace is_k = task->regions[2].region.get_index_space();

   Domain dom_u = runtime->get_index_space_domain(ctx, is_u);
   Domain dom_v = runtime->get_index_space_domain(ctx, is_v);
   Domain dom_k = runtime->get_index_space_domain(ctx, is_k);

   Rect<2> rect_u = dom_u.get_rect<2>();
   Rect<2> rect_v = dom_v.get_rect<2>();
   Rect<2> rect_k = dom_k.get_rect<2>();

   Rect<2> subrect;
   ByteOffset offsets[2];

   double *u_ptr = acc_u.raw_rect_ptr<2>(rect_u, subrect, offsets);
   assert(u_ptr != NULL);
   assert(rect_u == subrect);

   double *v_ptr = NULL;
   if (rect_v.volume() != 0) {
     // if legion leaf coinsides real leaf, no V is needed
     v_ptr = acc_v.raw_rect_ptr<2>(rect_v, subrect, offsets);
     assert(v_ptr != NULL);
     assert(rect_v == subrect);
     //std::cout << "V Offset: " << offsets[0].offset
     //	     << ", "         << offsets[1].offset << std::endl;
   }
   
   double *k_ptr = acc_k.raw_rect_ptr<2>(rect_k, subrect, offsets);
   assert(k_ptr != NULL);
   assert(rect_k == subrect);
   //std::cout << "K Offset: " << offsets[0].offset
   //	     << ", "         << offsets[1].offset << std::endl;

   int l_dim  = offsets[1].offset / sizeof(double);
   int u_nrow = rect_u.dim_size(0);
   assert( l_dim == u_nrow );   
   serial_leaf_solve(uroot, vroot, u_ptr, v_ptr, k_ptr, l_dim);
}


static void serial_leaf_solve
  (FSTreeNode * unode, FSTreeNode * vnode,
   double * u_ptr, double * v_ptr, double * k_ptr, int LD)
{
  /*
  printf("vlchild: %p, vrchild: %p\n", vnode->lchild, vnode->rchild);
  printf("ulchild: %p, urchild: %p\n", unode->lchild, unode->rchild);
  */
    
  if (unode->is_real_leaf()) {
    //printf("u nrow: %d, v nrow: %d\n", unode->nrow, vnode->nrow);
    assert(unode->nrow == vnode->nrow);
    int N     = unode->nrow;
    int NRHS  = unode->col_beg + unode->ncol;
    int LDA   = LD;
    int LDB   = LD;
    double *A = k_ptr + vnode->row_beg;
    double *B = u_ptr + vnode->row_beg;
      
    int INFO;
    int IPIV[N];
      
    lapack::dgesv_(&N, &NRHS, A, &LDA, IPIV, B, &LDB, &INFO);
    assert(INFO == 0);
    
    //lapack::dgetrf_(&N, &N, A, &LDA, IPIV, &INFO);
    /*
    //assume no pivoting
    for (int i=0; i<N; i++)
      IPIV[i] = i+1;
    */
    //char TRANS = 'n';
    //lapack::dgetrs_(&TRANS, &N, &NRHS, A, &LDA, IPIV, B, &LDB, &INFO);

    return;
  }

  serial_leaf_solve(unode->lchild, vnode->lchild, u_ptr, v_ptr, k_ptr, LD);
  serial_leaf_solve(unode->rchild, vnode->rchild, u_ptr, v_ptr, k_ptr, LD);
  
  char   transa = 't';
  char   transb = 'n';
  double alpha  = 1.0;
  double beta   = 0.0;
  
  int V0_rows = vnode->lchild->nrow;
  int V0_cols = vnode->lchild->ncol;
  int V1_rows = vnode->rchild->nrow;
  int V1_cols = vnode->rchild->ncol;

  int u0_rows = unode->lchild->nrow;
  int u0_cols = unode->lchild->ncol;
  int u1_rows = unode->rchild->nrow;
  int u1_cols = unode->rchild->ncol;
  
  int d0_rows = unode->lchild->nrow;
  int d0_cols = unode->lchild->col_beg;
  int d1_rows = unode->rchild->nrow;
  int d1_cols = unode->rchild->col_beg;
  
  double *V0 = v_ptr + vnode->lchild->row_beg + vnode->lchild->col_beg*LD;
  double *V1 = v_ptr + vnode->rchild->row_beg + vnode->rchild->col_beg*LD;
  double *u0 = u_ptr + unode->lchild->row_beg + unode->lchild->col_beg*LD;
  double *u1 = u_ptr + unode->rchild->row_beg + unode->rchild->col_beg*LD;
  double *d0 = u_ptr + unode->lchild->row_beg;
  double *d1 = u_ptr + unode->rchild->row_beg;


  // Shur complement
  assert(V0_cols + V1_cols == u0_cols + u1_cols);
  int    S_size = V0_cols + V1_cols;
  double *S = (double *) calloc( S_size*S_size, sizeof(double) );

  assert(d0_cols == d1_cols);
  double *S_RHS = (double *) malloc( S_size*d0_cols * sizeof(double) );
  
  // initialize the off-diagonal blocks to identity
  for (int i=0; i<S_size; i++)
    S[ (V0_cols + i)%S_size + i*S_size ] = 1.0;

  assert(V0_rows == u0_rows);
  assert(V1_rows == u1_rows);
  assert(V0_rows == d0_rows);
  assert(V1_rows == d1_rows);

  double *V0Tu0 = S;
  double *V1Tu1 = S + (V0_cols + u0_cols*S_size);
  double *V0Td0 = S_RHS;
  double *V1Td1 = S_RHS + V0_cols;
  
  blas::dgemm_(&transa, &transb, &V0_cols, &u0_cols, &V0_rows, &alpha, V0, &LD, u0, &LD, &beta, V0Tu0, &S_size);
  blas::dgemm_(&transa, &transb, &V1_cols, &u1_cols, &V1_rows, &alpha, V1, &LD, u1, &LD, &beta, V1Tu1, &S_size);
  blas::dgemm_(&transa, &transb, &V0_cols, &d0_cols, &V0_rows, &alpha, V0, &LD, d0, &LD, &beta, V0Td0, &S_size);
  blas::dgemm_(&transa, &transb, &V1_cols, &d1_cols, &V1_rows, &alpha, V1, &LD, d1, &LD, &beta, V1Td1, &S_size);

  
  int INFO;
  int IPIV[S_size];
  assert(d0_cols == d1_cols);

  lapack::dgesv_(&S_size, &d0_cols, S, &S_size, IPIV, S_RHS, &S_size, &INFO);
  assert(INFO == 0);


  //save_matrix(S_RHS, S_size, d1_cols, "S_RHS.txt");  
  transa =  'n';
  alpha  = -1.0;
  beta   =  1.0;
  
  double * eta0 = S_RHS;          
  double * eta1 = S_RHS + V1_cols;

  int eta0_rows = V1_cols;
  int eta0_cols = d0_cols;
  int eta1_rows = V0_cols;
  int eta1_cols = d0_cols;
  
  assert(u0_cols == eta0_rows);
  assert(u1_cols == eta1_rows);
  blas::dgemm_(&transa, &transb, &u0_rows, &eta0_cols, &u0_cols, &alpha, u0, &LD, eta0, &S_size, &beta, d0, &LD);
  blas::dgemm_(&transa, &transb, &u1_rows, &eta1_cols, &u1_cols, &alpha, u1, &LD, eta1, &S_size, &beta, d1, &LD);

  
  //printf("d0 2x2: %f, %f, %f, %f.\n", d0[0], d0[1], d0[LD], d0[LD+1]);
  
  //save_matrix(V0Tu0, V0_cols, u0_cols, "V0Tu0.txt");
  //save_matrix(V1Tu1, V1_cols, u1_cols, "V1Tu1.txt");
  //save_matrix(V0Td0, V0_cols, d0_cols, "V0Td0.txt");
  //save_matrix(V1Td1, V1_cols, d1_cols, "V1Td1.txt");
  //save_matrix(S,     S_size, S_size,  "Shur.txt");
  //save_matrix(S_RHS, S_size, d1_cols, "S_RHS.txt");
  //save_matrix(d0, unode->nrow, unode->col_beg+unode->ncol, LD, "result.txt");

  free(S);
  free(S_RHS);
}


// this function wrapper launches leaf tasks
void solve_legion_leaf
  (FSTreeNode * uleaf, FSTreeNode * vleaf, Range task_tag,
   Context ctx, HighLevelRuntime *runtime) {
  
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

  
  LeafSolveTask launcher(TaskArgument(
			   &arg[0],
			   sizeof(FSTreeNode)*(max_tree_size*2)),
			 Predicate::TRUE_PRED,
			 0,
			 task_tag.begin
			 );

  launcher.add_region_requirement(
    RegionRequirement(uleaf->lowrank_matrix->data,
		      READ_WRITE,
		      EXCLUSIVE,
		      uleaf->lowrank_matrix->data)); // u region
  launcher.add_region_requirement(
    RegionRequirement(vleaf->lowrank_matrix->data,
		      READ_ONLY,
		      EXCLUSIVE,
		      vleaf->lowrank_matrix->data)); // v region
  launcher.add_region_requirement(
    RegionRequirement(vleaf->dense_matrix->data,
		      READ_ONLY,
		      EXCLUSIVE,
		      vleaf->dense_matrix->data)); // k region
  launcher.region_requirements[0].add_field(FID_X);
  launcher.region_requirements[1].add_field(FID_X);
  launcher.region_requirements[2].add_field(FID_X);    
  Future ft = runtime->execute_task(ctx, launcher);
  
#ifdef SERIAL
  std::cout << "Waiting for leaf_solve task ..." << std::endl;
  ft.get_void_result();
#endif
}


void register_solver_operators() {
  LeafSolveTask::register_tasks();
  LUSolveTask::register_tasks();
}

