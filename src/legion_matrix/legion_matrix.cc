#include "legion_matrix.h"
#include "init_matrix_tasks.h"
#include "zero_matrix_task.h"
#include "save_region_task.h"
#include "macros.h"

void create_matrix
(LMatrix *(&matrix), int nrow, int ncol,
 Context ctx, HighLevelRuntime *runtime) {

  // ncol can be 0 for the matrix below legion node
  // in v tree
  assert(nrow > 0);
  matrix = new LMatrix(nrow, ncol);

  int lower[2] = {0,      0};
  int upper[2] = {nrow-1, ncol-1}; // inclusive bound
  Rect<2> rect((Point<2>(lower)), (Point<2>(upper)));
  FieldSpace fs = runtime->create_field_space(ctx);
  IndexSpace is = runtime->
    create_index_space(ctx, Domain::from_rect<2>(rect));
  FieldAllocator allocator = runtime->
    create_field_allocator(ctx, fs);
  allocator.allocate_field(sizeof(double), FID_X);
  matrix->data = runtime->create_logical_region(ctx, is, fs);
  assert(matrix->data != LogicalRegion::NO_REGION);
}

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

