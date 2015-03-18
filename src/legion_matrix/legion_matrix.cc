#include "legion_matrix.h"
#include "init_matrix_tasks.h"
#include "zero_matrix_task.h"
#include "save_region_task.h"
#include "macros.h"

LMatrix::LMatrix(const int rows_, const int cols_,
		 const LogicalRegion lr)
  : rows(rows_), cols(cols_), data(lr) {}

void LMatrix::rand
(const long seed, const Range &columns, const int taskTag,
 Context ctx, HighLevelRuntime *runtime) {

  this->seed = seed;
  RandomMatrixTask::TaskArgs args = {seed, columns};
  RandomMatrixTask launcher(TaskArgument(&args, sizeof(args)),
			    Predicate::TRUE_PRED,
			    0,
			    taskTag);
    
  launcher.add_region_requirement(RegionRequirement
				  (data,
				   READ_WRITE,
				   EXCLUSIVE,
				   data).
				  add_field(FID_X)
				  );
  Future f = runtime->execute_task(ctx, launcher);
#ifdef SERIAL
  std::cout << "Waiting for init rhs ..." << std::endl;
  f.get_void_result();
#endif
}

void LMatrix::zero
(const int taskTag, Context ctx, HighLevelRuntime *runtime) {
  
  assert(data != LogicalRegion::NO_REGION);
  ZeroMatrixTask launcher(TaskArgument(NULL, 0),
			  Predicate::TRUE_PRED,
			  0,
			  taskTag);
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
  (const int col, const int row, const int rank, const int taskTag,
   Context ctx, HighLevelRuntime *runtime) {

  typedef CirculantMatrixTask ICMT;
  ICMT::TaskArgs args = {col, row, rank};
  ICMT launcher(TaskArgument(&args,
			     sizeof(args)),
		Predicate::TRUE_PRED,
		0,
		taskTag);
  launcher.add_region_requirement(RegionRequirement
				  (data,
				   READ_WRITE,
				   EXCLUSIVE,
				   data)
				  );
  launcher.region_requirements[0].add_field(FID_X);
  Future f = runtime->execute_task(ctx, launcher);
#ifdef SERIAL
  std::cout << "Waiting for init low rank block ..." << std::endl;
  f.get_void_result();
#endif
}

void LMatrix::save
(const std::string& filename, const Range& columns,
 Context ctx, HighLevelRuntime *runtime, bool print_seed) {

  SaveRegionTask::TaskArgs args;
  strcpy(args.filename, filename.c_str());
  args.columns = columns;
  args.seed    = seed;
  args.print_seed = print_seed;
    
  SaveRegionTask launcher(TaskArgument(&args, sizeof(args)));

  launcher.add_region_requirement(RegionRequirement
				  (this->data,
				   READ_ONLY,
				   EXCLUSIVE,
				   this->data).
				  add_field(FID_X)
				  );
  Future f = runtime->execute_task(ctx, launcher);

  // save matrix task has to wait, because multiple tasks may try
  //  to write into the same file.
  // note: some matrices are stored in seperate regions
  f.get_void_result(); 
#ifdef SERIAL
  std::cout << "Waiting for writing into file ..." << std::endl;
#endif
}

