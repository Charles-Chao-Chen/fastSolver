#include "legion_matrix.h"
#include "init_matrix_tasks.h"
#include "zero_matrix_task.h"
#include "macros.h"

/* ---- Range class methods ---- */

Range Range::lchild () const
{
  int half_size = size/2;
  return (Range){begin, half_size};
}


Range Range::rchild () const
{
  int half_size = size/2;
  return (Range){begin+half_size, half_size};
}


/* ---- LMatrix class methods ---- */

void LMatrix::rand
  (const int randSeed, const Range &range, const Range &taskTag,
   Context ctx, HighLevelRuntime *runtime) {

  RandomMatrixTask::TaskArgs args = {randSeed, range.size};
  RandomMatrixTask launcher(TaskArgument(&args, sizeof(args)),
			    Predicate::TRUE_PRED,
			    0,
			    taskTag.begin);
    
  launcher.add_region_requirement(RegionRequirement
				  (data,
				   READ_WRITE,
				   EXCLUSIVE,
				   data).
				  add_field(FID_X)
				  );
  Future ft = runtime->execute_task(ctx, launcher);
  ft.get_void_result();
  //  std::cout << "Waiting for initializing rhs ..." << std::endl;
}


void LMatrix::zero
(const Range &taskTag, Context ctx, HighLevelRuntime *runtime) {
  
  assert(data != LogicalRegion::NO_REGION);
  ZeroMatrixTask launcher(TaskArgument(NULL, 0),
			  Predicate::TRUE_PRED,
			  0,
			  taskTag.begin);
  launcher.add_region_requirement(
	     RegionRequirement(data,
			       WRITE_DISCARD,
			       EXCLUSIVE,
			       data));
  launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_task(ctx, launcher);
}


void LMatrix::circulant
  (int col_beg, int row_beg, int rank, Range taskTag,
   Context ctx, HighLevelRuntime *runtime) {    

  typedef InitCirculantMatrixTask ICMT;
  ICMT::TaskArgs args = {col_beg, row_beg, rank};
  ICMT launcher(TaskArgument(&args,
			     sizeof(args)),
		Predicate::TRUE_PRED,
		0,
		taskTag.begin);
  launcher.add_region_requirement(RegionRequirement
				  (data,
				   READ_WRITE,
				   EXCLUSIVE,
				   data)
				  );
  launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_task(ctx, launcher);
}
