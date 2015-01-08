#include "initMatrixTasks.h"
#include "htreeHelper.h"
#include "lapack_blas.h"
#include "macros.h"

using namespace LegionRuntime::Accessor;


void register_init_tasks() {
  InitRHSTask::register_tasks();
  InitCirculantKmatTask::register_tasks();
  InitCirculantMatrixTask::register_tasks();
}


/* ---- InitRHSTask implementation ---- */

/*static*/
int InitRHSTask::TASKID;

InitRHSTask::
InitRHSTask(TaskArgument arg,
	    Predicate pred /*= Predicate::TRUE_PRED*/,
	    MapperID id /*= 0*/,
	    MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag)
{
}

/*static*/
void InitRHSTask::register_tasks(void)
{
  TASKID =
    HighLevelRuntime::register_legion_task
    <InitRHSTask::cpu_task>(AUTO_GENERATE_ID,
			    Processor::LOC_PROC, 
			    true,
			    true,
			    AUTO_GENERATE_ID,
			    TaskConfigOptions(true/*leaf*/),
			    "init_RHS");
  printf("Register task %d : init_RHS\n", TASKID);
}

void InitRHSTask::
cpu_task(const Task *task,
	 const std::vector<PhysicalRegion> &regions,
	 Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);

  TaskArgs* args = (TaskArgs *)task->args;
  int rand_seed = args->rand_seed;
  int ncol      = args->ncol;
  
  IndexSpace is   = task->regions[0].region.get_index_space();
  Domain     dom  = runtime->get_index_space_domain(ctx, is);
  Rect<2>    rect = dom.get_rect<2>();

  RegionAccessor<AccessorType::Generic, double> acc =
    regions[0].get_field_accessor(FID_X).typeify<double>();
  
  Rect<2> subrect;
  ByteOffset offsets[2];  

  double *ptr =  acc.raw_rect_ptr<2>(rect, subrect, offsets);
  assert(rect == subrect);
  assert(ptr  != NULL);

  int nrow = rect.dim_size(0);
  //printf("Start init_RHS task with %d rows.\n", nrow);
  
  srand( rand_seed );
  for (int j=0; j<ncol; j++) {
    for (int i=0; i<nrow; i++) {
      int row_idx = i;
      int col_idx = j;
      ptr[ row_idx + col_idx*nrow ] = frand(0, 1);
    }
  }
}


/* ---- InitCirculantKmatTask implementation ---- */

/*static*/
int InitCirculantKmatTask::TASKID;

InitCirculantKmatTask::
InitCirculantKmatTask(TaskArgument arg,
		      Predicate pred /*= Predicate::TRUE_PRED*/,
		      MapperID id /*= 0*/,
		      MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

/*static*/
void InitCirculantKmatTask::register_tasks(void)
{
  TASKID =
    HighLevelRuntime::register_legion_task
    <InitCirculantKmatTask::cpu_task>(AUTO_GENERATE_ID,
				      Processor::LOC_PROC, 
				      true,
				      true,
				      AUTO_GENERATE_ID,
				      TaskConfigOptions(true/*leaf*/),
				      "init_Kmat");
  printf("Register task %d : Init_Dense_Block\n", TASKID);
}

void InitCirculantKmatTask::
cpu_task(const Task *task,
	 const std::vector<PhysicalRegion> &regions,
	 Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);

  TaskArgs<MAX_TREE_SIZE> *args = (TaskArgs<MAX_TREE_SIZE> *)task->args;
  int row_beg_global = args->row_beg_global;
  int rank = args->rank;
  double diag = args->diag;
  FSTreeNode *treeArray = args->treeArray;
  assert(task->arglen == sizeof(TaskArgs<MAX_TREE_SIZE>));

  FSTreeNode *vroot = treeArray;
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

  int leading_dimension  = offsets[1].offset / sizeof(double);
  int k_nrow = rect_k.dim_size(0);
  assert( leading_dimension == k_nrow );
  // initialize Kmat
  memset(k_ptr, 0, rect_k.dim_size(0)*rect_k.dim_size(1)*sizeof(double));
  fill_circulant_Kmat(vroot, row_beg_global, rank, diag, k_ptr, leading_dimension);
}


/* ---- InitCirculantMatrixTask implementation ---- */

/*static*/
int InitCirculantMatrixTask::TASKID;

InitCirculantMatrixTask::
InitCirculantMatrixTask(TaskArgument arg,
			Predicate pred /*= Predicate::TRUE_PRED*/,
			MapperID id /*= 0*/,
			MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

/*static*/
void InitCirculantMatrixTask::register_tasks(void)
{
  TASKID =
    HighLevelRuntime::register_legion_task
    <InitCirculantMatrixTask::cpu_task>(AUTO_GENERATE_ID,
					Processor::LOC_PROC, 
					true,
					true,
					AUTO_GENERATE_ID,
					TaskConfigOptions(true/*leaf*/),
					"init_low_rank_block");
  printf("Register task %d : Init_Low_Rank_Block\n", TASKID);
}

void InitCirculantMatrixTask::
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

