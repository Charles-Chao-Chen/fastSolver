#include "legion_matrix.h"
#include "initMatrixTasks.h"
#include "macros.h"

/* ---- Range class methods ---- */

Range Range::lchild ()
{
  int half_size = size/2;
  return (Range){begin, half_size};
}


Range Range::rchild ()
{
  int half_size = size/2;
  return (Range){begin+half_size, half_size};
}


/* ---- LMatrix class methods ---- */

void LMatrix::init_circulant_matrix
(int col_beg, int row_beg, int r, Range tag,
 Context ctx, HighLevelRuntime *runtime) {    

  typedef InitCirculantMatrixTask ICMT;
  ICMT::TaskArgs args = {col_beg, row_beg, r};
  ICMT launcher(TaskArgument(&args,
			     sizeof(args)),
		Predicate::TRUE_PRED,
		0,
		tag.begin);
  launcher.add_region_requirement(RegionRequirement
				  (data,
				   READ_WRITE,
				   EXCLUSIVE,
				   data)
				  );
  launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_task(ctx, launcher);
}
