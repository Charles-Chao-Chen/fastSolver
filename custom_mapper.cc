#include "custom_mapper.h"

void register_custom_mapper() {
  HighLevelRuntime::set_registration_callback(mapper_registration);
}

// Here we override the DefaultMapper ID so that
// all tasks that normally would have used the
// DefaultMapper will now use our AdversarialMapper.
// We create one new mapper for each processor
// and register it with the runtime.
void mapper_registration(Machine machine, HighLevelRuntime *rt,
			 const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
       it != local_procs.end(); it++) {
    rt->replace_default_mapper(
	new AdversarialMapper(machine, rt, *it), *it);
  }
}

// Here is the constructor for our adversial mapper.
// We'll use the constructor to illustrate how mappers can
// get access to information regarding the current machine.
AdversarialMapper::AdversarialMapper(Machine m, 
                                     HighLevelRuntime *rt, Processor p)
  : DefaultMapper(m, rt, p) // pass arguments through to DefaultMapper
{
  typedef std::set<Memory>::const_iterator    SMCI;

  std::set<Memory> all_mems;
  machine.get_all_memories(all_mems);
  for (SMCI it = all_mems.begin(); it != all_mems.end(); it++) {
    Memory::Kind kind = it->kind();
    if (kind == Memory::SYSTEM_MEM) {
      valid_mems.push_back(*it);
    }
  }
  assert( ! valid_mems.empty() );
}

// The first mapper call that we override is the 
// select_task_options call.  This mapper call is
// performed on every task launch immediately after
// it is made.  The call asks the mapper to select 
// set properities of the task:
//
//  inline_task - whether the task should be directly
//    inlined into its parent task, using the parent
//    task's physical regions.  
//  spawn_task - whether the task is eligible for 
//    stealing (based on Cilk-style semantics)
//  map_locally - whether the task should be mapped
//    by the processor on which it was launched or
//    whether it should be mapped by the processor
//    where it will run.
//  profile_task - should the runtime collect profiling
//    information about the task while it is executing
//  target_proc - which processor should the task be
//    sent to once all of its mapping dependences have
//    been satisifed.
//
//  Note that these properties are all set on the Task
//  object declared in legion.h.  The Task object is
//  the representation of a task that the Legion runtime
//  provides to the mapper for specifying mapping
//  decisions.  Note that there are similar objects
//  for inline mappings as well as other operations.
//
//  For our adversarial mapper, we perform the default
//  choices for all options except the last one.  Here
//  we choose a random processor in our system to 
//  send the task to.

void AdversarialMapper::select_task_options(Task *task)
{
  task->inline_task   = false;
  task->spawn_task    = false;
  task->map_locally   = false; // turn on remote mapping
  task->profile_task  = false;
  task->task_priority = 0;

  // pick the target memory idexed by task->tag
  // note launch node tasks have negative tags
  unsigned taskTag = abs(task->tag);
  assert(taskTag < valid_mems.size());
  Memory mem = valid_mems[taskTag];
  assert(mem != Memory::NO_MEMORY);
  
  // select valid processors
  typedef std::set<Processor>::const_iterator SPCI;
  std::vector<Processor> valid_options;
  std::set<Processor> options;
  machine.get_shared_processors(mem, options);
  for (SPCI it = options.begin(); it != options.end(); ) {
    Processor::Kind kind = it->kind();
    if (kind == Processor::LOC_PROC)
      valid_options.push_back(*it);
    it++;
  }
  
  if ( !valid_options.empty() ) {
    task->target_proc = valid_options[0];
    task->additional_procs.insert(valid_options.begin(),
				  valid_options.end());
  } else {
    // no valid processor available
    task->target_proc = Processor::NO_PROC;
    assert(false);
  }
}


// The next mapping call that we override is the map_task
// mapper method. Once a task has been assigned to map on
// a specific processor (the target_proc) then this method
// is invoked by the runtime to select the memories in 
// which to create physical instances for each logical region.
// The mapper communicates this information to the runtime
// via the mapping fields on RegionRequirements. The memories
// containing currently valid physical instances for each
// logical region is provided by the runtime in the 
// 'current_instances' field. The mapper must specify an
// ordered list of memories for the runtime to try when
// creating a physical instance in the 'target_ranking'
// vector field of each RegionRequirement. The runtime
// attempts to find or make a physical instance in each 
// memory until it succeeds. If the runtime fails to find
// or make a physical instance in any of the memories, then
// the mapping fails and the mapper will be notified that
// the task failed to map using the 'notify_mapping_failed'
// mapper call. If the mapper does nothing, then the task
// is placed back on the list of tasks eligible to be mapped.
// There are other fields that the mapper can set in the
// process of the map_task call that we do not cover here.
//
// In this example, the mapper finds the set of all visible
// memories from the target processor and then puts them
// in a random order as the target set of memories, thereby
// challenging the Legion runtime to maintain correctness
// of data moved through random sets of memories.

bool AdversarialMapper::map_task(Task *task)
{    
  // Put everything in the system memory
  Memory sys_mem = 
    machine_interface.find_memory_kind(task->target_proc,
				       Memory::SYSTEM_MEM);
  assert(sys_mem.exists());
  for (unsigned idx = 0; idx < task->regions.size(); idx++)
    {
      task->regions[idx].target_ranking.push_back(sys_mem);

      // special mapping ID for launch node tasks
      //  the regions will be virtually mapped
      task->regions[idx].virtual_map = task->tag < 0 ? true : false;
      task->regions[idx].enable_WAR_optimization = war_enabled;
      task->regions[idx].reduction_list = false;
      		
      // make everything SOA
      task->regions[idx].blocking_factor = 1;
      //task->regions[idx].max_blocking_factor;
    } 
  return true;
}

void AdversarialMapper::notify_mapping_failed(const Mappable *mappable)
{
  printf("WARNING: MAPPING FAILED!  Retrying...\n");
}


// The last mapper call we override is the notify_mapping_result
// call which is invoked by the runtime if the mapper indicated
// that it wanted to know the result of the mapping following
// the map_task call by returning true. The runtime invokes
// this call and the chosen memories for each RegionRequirement
// are set in the 'selected_memory' field. We use this call in
// this example to record the memories in which physical instances
// were mapped for each logical region of each task so we can
// see that the assignment truly is random.
/*
  void AdversarialMapper::notify_mapping_result(const Mappable *mappable)
  {
  if (mappable->get_mappable_kind() == Mappable::TASK_MAPPABLE)
  {
  const Task *task = mappable->as_mappable_task();
  assert(task != NULL);
  for (unsigned idx = 0; idx < task->regions.size(); idx++)
  {
  printf("Mapped region %d of task %s (ID %lld) to memory %x\n",
  idx, task->variants->name, 
  task->get_unique_task_id(),
  task->regions[idx].selected_memory.id);
  }
  }
  }
*/
