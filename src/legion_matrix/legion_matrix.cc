#include "legion_matrix.h"
#include "init_matrix_tasks.h"
#include "zero_matrix_task.h"
#include "save_region_task.h"
#include "macros.h"


/* ---- LMatrix class methods ---- */

void LMatrix::rand
  (long int seed, const Range &range, const Range &taskTag,
   Context ctx, HighLevelRuntime *runtime) {

  this->seed = seed;
  RandomMatrixTask::TaskArgs args = {seed, range.size};
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
#ifdef SERIAL
  Future ft = runtime->execute_task(ctx, launcher);
  std::cout << "Waiting for init rhs ..." << std::endl;
  ft.get_void_result();
#endif
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
  Future f = runtime->execute_task(ctx, launcher);
#ifdef SERIAL
  std::cout << "Waiting for zero ..." << std::endl;
  f.get_void_result();
#endif
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
  Future f = runtime->execute_task(ctx, launcher);
#ifdef SERIAL
  std::cout << "Waiting for init low rank block ..."
	    << std::endl;
#endif

}


void LMatrix::save
(const std::string filename, const Range rg, 
 Context ctx, HighLevelRuntime *runtime, bool print_seed) {

  SaveRegionTask::TaskArgs args;
  strcpy(args.filename, filename.c_str());
  args.col_range = rg;
  args.seed = seed;
  args.print_seed = print_seed;
    
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
#ifdef SERIAL
  std::cout << "Waiting for writing into file ..." << std::endl;
#endif

}

