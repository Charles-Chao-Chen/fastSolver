#include <assert.h>

#include "zero_matrix_task.h"


void register_zero_matrix_task() {
  ZeroMatrixTask::register_task();
}

/*
static void
zero_matrix(LogicalRegion &matrix, Range tag,
	    Context ctx, HighLevelRuntime *runtime) {
  
  assert(matrix != LogicalRegion::NO_REGION);
  ZeroMatrixTask launcher(TaskArgument(NULL, 0),
			  Predicate::TRUE_PRED,
			  0,
			  tag.begin);
  launcher.add_region_requirement(
	     RegionRequirement(matrix,
			       WRITE_DISCARD,
			       EXCLUSIVE,
			       matrix));
  launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_task(ctx, launcher);
}


static void
scale_matrix(double beta, LogicalRegion &matrix,
	     Context ctx, HighLevelRuntime *runtime) {
  assert(false);
}
*/

/* ---- zero_matrix_task implementation ---- */

/*static*/
int ZeroMatrixTask::TASKID;

ZeroMatrixTask::ZeroMatrixTask(
  TaskArgument arg,
  Predicate pred /*= Predicate::TRUE_PRED*/,
  MapperID id /*= 0*/,
  MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

/*static*/
void ZeroMatrixTask::register_task(void)
{
  TASKID =
    HighLevelRuntime::register_legion_task<ZeroMatrixTask::cpu_task>(
    AUTO_GENERATE_ID,
    Processor::LOC_PROC, 
    true,
    true,
    AUTO_GENERATE_ID,
    TaskConfigOptions(true/*leaf*/),
    "Init_Zero_Matrix");
  
  printf("Register task %d : ZeroMatrix\n", TASKID);
}

void
ZeroMatrixTask::cpu_task(const Task *task,
			 const std::vector<PhysicalRegion> &regions,
			 Context ctx, HighLevelRuntime *runtime) {

  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  assert(task->arglen == 0);

  IndexSpace is = task->regions[0].region.get_index_space();
  Domain dom = runtime->get_index_space_domain(ctx, is);
  Rect<2> rect = dom.get_rect<2>();

  Rect<2> subrect;
  ByteOffset offsets[2];

  double *ptr = regions[0].get_field_accessor(FID_X).typeify<double>().raw_rect_ptr<2>(rect, subrect, offsets);
  assert(rect == subrect);
  assert(ptr  != NULL);
  
  int nrow = rect.dim_size(0);
  int ncol = rect.dim_size(1);
  int size = nrow * ncol;

  memset(ptr, 0, size*sizeof(double));
}
