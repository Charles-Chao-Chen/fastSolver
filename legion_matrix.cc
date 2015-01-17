#include "legion_matrix.h"
#include "init_matrix_tasks.h"
#include "zero_matrix_task.h"
#include "save_region_task.h"
#include "macros.h"


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


void LMatrix::save
(const std::string filename,
 Context ctx, HighLevelRuntime *runtime, const Range rg) {

  SaveRegionTask::TaskArgs args;
  args.filename  = filename;
  args.col_range = rg;
    
  SaveRegionTask launcher(TaskArgument(&args, sizeof(args)));  
  launcher.add_region_requirement(RegionRequirement
				  (this->data,
				   READ_ONLY,
				   EXCLUSIVE,
				   this->data).
				  add_field(FID_X)
				  );
  Future fm = runtime->execute_task(ctx, launcher);
  fm.get_void_result(); // wait until finish
}

