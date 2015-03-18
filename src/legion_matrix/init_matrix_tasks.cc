#include "init_matrix_tasks.h"
#include "node.h"
#include "lapack_blas.h"
#include "macros.h"

#include <stdlib.h> // for srand48_r() and drand48_r()
#include <assert.h>

void register_init_tasks() {
  RandomMatrixTask::register_tasks();
  DenseMatrixTask::register_tasks();
  CirculantMatrixTask::register_tasks();
}

/* ---- RandomMatrixTask implementation ---- */

/*static*/
int RandomMatrixTask::TASKID;

RandomMatrixTask::
RandomMatrixTask(TaskArgument arg,
		 Predicate pred /*= Predicate::TRUE_PRED*/,
		 MapperID id /*= 0*/,
		 MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

/*static*/
void RandomMatrixTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <RandomMatrixTask::cpu_task>(AUTO_GENERATE_ID,
				 Processor::LOC_PROC, 
				 true,
				 true,
				 AUTO_GENERATE_ID,
				 TaskConfigOptions(true/*leaf*/),
				 "random_matrix");
#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : randomize_matrix\n", TASKID);
#endif
}

void RandomMatrixTask::
cpu_task(const Task *task,
	 const std::vector<PhysicalRegion> &regions,
	 Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);

  TaskArgs* args = (TaskArgs *)task->args;
  long seed      = args->seed;
  Range columns  = args->columns;

  Rect<2> subrect;
  ByteOffset offsets[2];  
  IndexSpace is   = task->regions[0].region.get_index_space();
  Domain     dom  = runtime->get_index_space_domain(ctx, is);
  Rect<2>    rect = dom.get_rect<2>();
  double    *ptr  = regions[0].get_field_accessor(FID_X).
    typeify<double>().raw_rect_ptr<2>(rect, subrect, offsets);

  assert(rect == subrect);
  assert(ptr  != NULL);
    
  int nrow = rect.dim_size(0);
  struct drand48_data buffer;
  assert( srand48_r( seed, &buffer ) == 0 );
  for (int i=0; i<nrow; i++) {
    for (int j=0; j<columns.size(); j++) {
      int row = i;
      int col = j + columns.begin();
      int count = row + col*nrow;
      assert( drand48_r( &buffer, &ptr[count]) == 0 );
    }
  }
}


/* ---- CirculantKmatTask implementation ---- */

/*static*/
int DenseMatrixTask::TASKID;

DenseMatrixTask::
DenseMatrixTask(TaskArgument arg,
		      Predicate pred /*= Predicate::TRUE_PRED*/,
		      MapperID id /*= 0*/,
		      MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

/*static*/
void DenseMatrixTask::register_tasks(void)
{
  TASKID =
    HighLevelRuntime::register_legion_task
    <DenseMatrixTask::cpu_task>(AUTO_GENERATE_ID,
				      Processor::LOC_PROC, 
				      true,
				      true,
				      AUTO_GENERATE_ID,
				      TaskConfigOptions(true),
				      "init_Kmat");
#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : init_Dense_Block\n", TASKID);
#endif
}

void DenseMatrixTask::cpu_task
(const Task *task, const std::vector<PhysicalRegion> &regions,
 Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);

  TaskArgs<MAX_TREE_SIZE> *args = (TaskArgs<MAX_TREE_SIZE> *)task->args;
  int row_beg_global = args->row;
  int rank = args->rank;
  double diag = args->diag;
  Node *treeArray = args->treeArray;
  assert(task->arglen == sizeof(TaskArgs<MAX_TREE_SIZE>));

  Node *vroot = treeArray;
  array_to_tree(treeArray, 0);
  
  RegionAccessor<AccessorType::Generic, double> acc_k = 
    regions[0].get_field_accessor(FID_X).typeify<double>();
  IndexSpace is_k = task->regions[0].region.get_index_space();
  Domain dom_k = runtime->get_index_space_domain(ctx, is_k);
  Rect<2> rect_k = dom_k.get_rect<2>();

  Rect<2> subrect;
  ByteOffset offsets[2];

  double *k_ptr = acc_k.raw_rect_ptr<2>(rect_k, subrect, offsets);
  assert(k_ptr != NULL);
  assert(rect_k == subrect);

  int LD  = offsets[1].offset / sizeof(double);
  int krow = rect_k.dim_size(0);
  assert( LD == krow );

  // initialize Kmat
  memset(k_ptr, 0, rect_k.dim_size(0)*rect_k.dim_size(1)*sizeof(double));
  fill_circulant_Kmat(vroot, row_beg_global, rank, diag, k_ptr, LD);
}

/* ---- CirculantMatrixTask implementation ---- */

/*static*/
int CirculantMatrixTask::TASKID;

CirculantMatrixTask::
CirculantMatrixTask(TaskArgument arg,
			Predicate pred /*= Predicate::TRUE_PRED*/,
			MapperID id /*= 0*/,
			MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

/*static*/
void CirculantMatrixTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <CirculantMatrixTask::cpu_task>(AUTO_GENERATE_ID,
					Processor::LOC_PROC, 
					true,
					true,
					AUTO_GENERATE_ID,
					TaskConfigOptions(true),
					"init_low_rank_block");
#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : Init_Low_Rank_Block\n", TASKID);
#endif
}

void CirculantMatrixTask::
cpu_task(const Task *task,
	 const std::vector<PhysicalRegion> &regions,
	 Context ctx, HighLevelRuntime *runtime)
{

  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->arglen == sizeof(TaskArgs));

  const TaskArgs cir_arg = *((const TaskArgs*)task->args);
  int col_beg = cir_arg.col_beg;
  int row_beg = cir_arg.row_beg;
  int r       = cir_arg.rank;
  
  IndexSpace is = task->regions[0].region.get_index_space();
  Domain dom = runtime->get_index_space_domain(ctx, is);
  Rect<2> rect = dom.get_rect<2>();

  Rect<2> subrect;
  ByteOffset offsets[2];

  double *ptr = regions[0].get_field_accessor(FID_X).
    typeify<double>().raw_rect_ptr<2>(rect, subrect, offsets);
  assert(rect == subrect);
  assert(ptr  != NULL);
  
  int nrow = rect.dim_size(0);
  int ncol = rect.dim_size(1);
  int vol  = rect.volume();
  assert( (ncol - col_beg) % r == 0 );
    
  for (int j=0; j<ncol - col_beg; j++) {
    for (int i=0; i<nrow; i++) {
      int value = (j+i+row_beg)%r;

      int irow = i;
      int icol = j+col_beg;

      assert(irow + icol*nrow < vol);
      ptr[irow + icol*nrow] = value;
    }
  }
}

/*
void fill_circulant_Kmat
(Node * vnode, int row_beg_glo, int r, double diag, double *Kmat, int LD) {

  if (vnode->is_real_leaf()) {

    int ksize = vnode->nrow;
    
    // init U as a circulant matrix
    double *U = (double *) malloc(ksize*r * sizeof(double));
    for (int j=0; j<r; j++) {
      for (int i=0; i<ksize; i++) {
	U[i+j*ksize] = (vnode->row_beg+row_beg_glo+i+j) % r;
      }
    }

    // init the diagonal entries
    for (int i=0; i<ksize; i++)
      Kmat[vnode->row_beg + i + LD*i] = diag;
    
    char transa = 'n';
    char transb = 't';
    int  m = ksize;
    int  n = ksize;
    int  k = r;
    int  lda = ksize;
    int  ldb = ksize;
    int  ldc = LD;
    double alpha = 1.0;
    double beta  = 1.0;
    double *A = U;
    double *B = U;
    double *C = Kmat + vnode->row_beg;
    blas::dgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
    
    free(U);
    return;
  }

  fill_circulant_Kmat(vnode->lchild, row_beg_glo, r, diag, Kmat, LD);
  fill_circulant_Kmat(vnode->rchild, row_beg_glo, r, diag, Kmat, LD);
}
*/
