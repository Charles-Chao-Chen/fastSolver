#include <iomanip>
#include <fstream>

#include "save_region_task.h"
#include "macros.h"

void register_save_region_task() {
  SaveRegionTask::register_tasks();
}


/* ---- SaveRegionTask implementation ---- */

/*static*/
int SaveRegionTask::TASKID;

SaveRegionTask::SaveRegionTask(TaskArgument arg,
			       Predicate pred /*=TRUE_PRED*/,
			       MapperID id /*= 0*/,
			       MappingTagID tag /*= 0*/)
  : TaskLauncher(TASKID, arg, pred, id, tag) {}

/*static*/
void SaveRegionTask::register_tasks(void)
{
  TASKID = HighLevelRuntime::register_legion_task
    <SaveRegionTask::cpu_task>(
			       AUTO_GENERATE_ID,
			       Processor::LOC_PROC, 
			       true,
			       true,
			       AUTO_GENERATE_ID,
			       TaskConfigOptions(true/*leaf*/),
			       "save_rhs");
#ifdef SHOW_REGISTER_TASKS
  printf("Register task %d : save_rhs\n", TASKID);
#endif
}

void save_data
(double *ptr, int nrow, int col_beg, int ncol,
 std::string filename, long int seed, bool print_seed) {

#ifdef DEBUG
  std::cout << " writing into "
	    << filename.c_str() << std::endl;
#endif
  std::ofstream outputFile;
  outputFile.exceptions ( std::ifstream::failbit
			| std::ifstream::badbit );
  try {
    outputFile.open(filename.c_str(), std::ios_base::app);
    if (print_seed)
      outputFile << seed << std::endl;
    for (int i=0; i<nrow; i++) {
      for (int j=0; j<ncol; j++) {
	int row_idx = i;
	int col_idx = j+col_beg;
	double x = ptr[ row_idx + col_idx*nrow ];
	outputFile << std::setprecision(20) << x << '\t';
      }
      outputFile << std::endl;
    }
    outputFile.close();
  }
  catch (std::ifstream::failure e) {
    std::cerr << "Exception writing into "
	      << filename.c_str() << std::endl;
  }
}

void SaveRegionTask::cpu_task
(const Task *task,
 const std::vector<PhysicalRegion> &regions,
 Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  const TaskArgs* task_args = (TaskArgs *)task->args;
  const char* filename = task_args->filename;
  int col_beg          = task_args->col_range.begin;
  int ncol             = task_args->col_range.size;
  long int seed        = task_args->seed;
  bool print_seed      = task_args->print_seed;
  
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
  if (ncol == -1)
    ncol = rect.dim_size(1);
  assert(col_beg+ncol <= rect.dim_size(1));
  save_data(ptr, nrow, col_beg, ncol, filename, seed, print_seed);
}

