#include "custom_mapper.h"

void register_custom_mapper() {
  HighLevelRuntime::set_registration_callback(mapper_registration);
}

// Here we override the DefaultMapper ID so that
// all tasks that normally would have used the
// DefaultMapper will now use our AdversarialMapper.
// We create one new mapper for each processor
// and register it with the runtime.
void mapper_registration(Machine *machine, HighLevelRuntime *rt,
			 const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
       it != local_procs.end(); it++)
    {
      rt->replace_default_mapper(new AdversarialMapper(machine, rt, *it), *it);
    }
}

// Here is the constructor for our adversial mapper.
// We'll use the constructor to illustrate how mappers can
// get access to information regarding the current machine.
AdversarialMapper::AdversarialMapper(Machine *m, 
                                     HighLevelRuntime *rt, Processor p)
  : DefaultMapper(m, rt, p) // pass arguments through to DefaultMapper
{
  
  // The machine object is a singleton object that can be
  // used to get information about the underlying hardware.
  // The machine pointer will always be provided to all
  // mappers, but can be accessed anywhere by the static
  // member method Machine::get_machine().  Here we get
  // a reference to the set of all processors in the machine
  // from the machine object.  Note that the Machine object
  // actually comes from the Legion low-level runtime, most
  // of which is not directly needed by application code.
  // Typedefs in legion_types.h ensure that all necessary
  // types for querying the machine object are in scope
  // in the Legion HighLevel namespace.
  const std::set<Processor> &all_procs = machine->get_all_processors();
  // Recall that we create one mapper for every processor.  We
  // only want to print out this information one time, so only
  // do it if we are the mapper for the first processor in the
  // list of all processors in the machine.
  if ((*(all_procs.begin())) == local_proc)
    {
      // Print out how many processors there are and each
      // of their kinds.
      printf("There are %ld processors:\n", all_procs.size());
      for (std::set<Processor>::const_iterator it = all_procs.begin();
	   it != all_procs.end(); it++)
	{
	  // For every processor there is an associated kind
	  Processor::Kind kind = machine->get_processor_kind(*it);
	  switch (kind)
	    {
	      // Latency-optimized cores (LOCs) are CPUs
	    case Processor::LOC_PROC:
	      {
		printf("  Processor ID %x is CPU\n", it->id); 
		break;
	      }
	      // Throughput-optimized cores (TOCs) are GPUs
	    case Processor::TOC_PROC:
	      {
		printf("  Processor ID %x is GPU\n", it->id);
		break;
	      }
	      // Utility processors are helper processors for
	      // running Legion runtime meta-level tasks and 
	      // should not be used for running application tasks
	    case Processor::UTIL_PROC:
	      {
		printf("  Processor ID %x is utility\n", it->id);
		break;
	      }
	    default:
	      assert(false);
	    }
	}
      // We can also get the list of all the memories available
      // on the target architecture and print out their info.
      const std::set<Memory> &all_mems = machine->get_all_memories();
      printf("There are %ld memories:\n", all_mems.size());
      for (std::set<Memory>::const_iterator it = all_mems.begin();
	   it != all_mems.end(); it++)
	{
	  Memory::Kind kind = machine->get_memory_kind(*it);
	  size_t memory_size_in_kb = machine->get_memory_size(*it) >> 10;
	  switch (kind)
	    {
	      // RDMA addressable memory when running with GASNet
	    case Memory::GLOBAL_MEM:
	      {
		printf("  GASNet Global Memory ID %x has %ld KB\n", 
		       it->id, memory_size_in_kb);
		break;
	      }
	      // DRAM on a single node
	    case Memory::SYSTEM_MEM:
	      {
		printf("  System Memory ID %x has %ld KB\n",
		       it->id, memory_size_in_kb);
		break;
	      }
	      // Pinned memory on a single node
	    case Memory::REGDMA_MEM:
	      {
		printf("  Pinned Memory ID %x has %ld KB\n",
		       it->id, memory_size_in_kb);
		break;
	      }
	      // A memory associated with a single socket
	    case Memory::SOCKET_MEM:
	      {
		printf("  Socket Memory ID %x has %ld KB\n",
		       it->id, memory_size_in_kb);
		break;
	      }
	      // Zero-copy memory betweeen CPU DRAM and
	      // all GPUs on a single node
	    case Memory::Z_COPY_MEM:
	      {
		printf("  Zero-Copy Memory ID %x has %ld KB\n",
		       it->id, memory_size_in_kb);
		break;
	      }
	      // GPU framebuffer memory for a single GPU
	    case Memory::GPU_FB_MEM:
	      {
		printf("  GPU Frame Buffer Memory ID %x has %ld KB\n",
		       it->id, memory_size_in_kb);
		break;
	      }
	      // Block of memory sized for L3 cache
	    case Memory::LEVEL3_CACHE:
	      {
		printf("  Level 3 Cache ID %x has %ld KB\n",
		       it->id, memory_size_in_kb);
		break;
	      }
	      // Block of memory sized for L2 cache
	    case Memory::LEVEL2_CACHE:
	      {
		printf("  Level 2 Cache ID %x has %ld KB\n",
		       it->id, memory_size_in_kb);
		break;
	      }
	      // Block of memory sized for L1 cache
	    case Memory::LEVEL1_CACHE:
	      {
		printf("  Level 1 Cache ID %x has %ld KB\n",
		       it->id, memory_size_in_kb);
		break;
	      }
	    default:
	      assert(false);
	    }
	}

    
      // The Legion machine model represented by the machine object
      // can be thought of as a graph with processors and memories
      // as the two kinds of nodes.  There are two kinds of edges
      // in this graph: processor-memory edges and memory-memory
      // edges.  An edge between a processor and a memory indicates
      // that the processor can directly perform load and store
      // operations to that memory.  Memory-memory edges indicate
      // that data movement can be directly performed between the
      // two memories.  To illustrate how this works we examine
      // all the memories visible to our local processor in 
      // this mapper.  We can get our set of visible memories
      // using the 'get_visible_memories' method on the machine.
      const std::set<Memory> vis_mems = machine->get_visible_memories(local_proc);
      printf("There are %ld memories visible from processor %x\n",
	     vis_mems.size(), local_proc.id);
      for (std::set<Memory>::const_iterator it = vis_mems.begin();
	   it != vis_mems.end(); it++)
	{
	  // Edges between nodes are called affinities in the
	  // machine model.  Affinities also come with approximate
	  // indications of the latency and bandwidth between the 
	  // two nodes.  Right now these are unit-less measurements,
	  // but our plan is to teach the Legion runtime to profile
	  // these values on start-up to give them real values
	  // and further increase the portability of Legion applications.
	  std::vector<ProcessorMemoryAffinity> affinities;
	  int results = 
	    machine->get_proc_mem_affinity(affinities, local_proc, *it);
	  // We should only have found 1 results since we
	  // explicitly specified both values.
	  assert(results == 1);
	  printf("  Memory %x has bandwidth %d and latency %d\n",
		 it->id, affinities[0].bandwidth, affinities[0].latency);
	}
    }


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
  task->inline_task = false;
  task->spawn_task = false;
  task->map_locally = false; // turn on remote mapping
  task->profile_task = false;
  task->task_priority = 0;
  const std::set<Processor> &all_procs = machine->get_all_processors();
  
  //  assert(task->tag == 0);
  //printf("task tag: %d.\n", task->tag);
  
  const std::set<Memory> &all_mems = machine->get_all_memories();
  //printf("There are %ld memories:\n", all_mems.size());

  std::vector<Memory> valid_mems;
  for (std::set<Memory>::const_iterator it = all_mems.begin();
       it != all_mems.end(); it++) {
    
    Memory::Kind kind = machine->get_memory_kind(*it);
    if (kind == Memory::SYSTEM_MEM)
      valid_mems.push_back(*it);
  }

  //assert(task->tag < valid_mems.size());
  int mem_idx = task->tag % valid_mems.size();

  
  //mem_idx = 1- mem_idx;

  
  Memory mem = valid_mems[mem_idx];
  const std::set<Processor> & options = machine->get_shared_processors(mem);
  //printf("There are %d option processors.\n", options.size());
	
  std::vector<Processor> valid_options;
  for (std::set<Processor>::const_iterator it = options.begin();
       it != options.end(); it++) {
    if (machine->get_processor_kind(*it) == Processor::LOC_PROC)
      valid_options.push_back(*it);
  }
  
  //printf("There are %d valid processors.\n", valid_options.size());
  if (!valid_options.empty()) {
    task->target_proc = valid_options[0];
    task->additional_procs.insert(valid_options.begin(), valid_options.end());
  } else {
    task->target_proc = Processor::NO_PROC;
    assert(false);
  }
   
}



// The second call that we override is the slice_domain
// method. The slice_domain call is used by the runtime
// to query the mapper about the best way to distribute
// the points in an index space task launch throughout
// the machine. The maper is given the task and the domain
// to slice and then asked to generate sub-domains to be
// sent to different processors in the form of DomainSplit
// objects. DomainSplit objects describe the sub-domain,
// the target processor for the sub-domain, whether the
// generated slice can be stolen, and finally whether 
// slice_domain' should be recursively called on the
// slice when it arrives at its destination.
//
// In this example we use a utility method from the DefaultMapper
// called decompose_index_space to decompose our domain. We 
// recursively split the domain in half during each call of
// slice_domain and send each slice to a random processor.
// We continue doing this until the leaf domains have only
// a single point in them. This creates a tree of slices of
// depth log(N) in the number of points in the domain with
// each slice being sent to a random processor.
/*
  void AdversarialMapper::slice_domain(const Task *task, const Domain &domain,
  std::vector<DomainSplit> &slices)
  {
  const std::set<Processor> &all_procs = machine->get_all_processors();
  std::vector<Processor> split_set;
  for (unsigned idx = 0; idx < 2; idx++)
  {
  split_set.push_back(DefaultMapper::select_random_processor(
  all_procs, Processor::LOC_PROC, machine));
  }

  DefaultMapper::decompose_index_space(domain, split_set, 
  1, slices);
  for (std::vector<DomainSplit>::iterator it = slices.begin();
  it != slices.end(); it++)
  {
  Rect<1> rect = it->domain.get_rect<1>();
  if (rect.volume() == 1)
  it->recurse = false;
  else
  it->recurse = true;
  }
  }
*/
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
      task->regions[idx].virtual_map = false;
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

/*
  bool AdversarialMapper::map_task(Task *task) {

  // Put everything in the system memory
  Memory sys_mem = 
  machine_interface.find_memory_kind(task->target_proc,
  Memory::SYSTEM_MEM);
  assert(sys_mem.exists());
  for (unsigned idx = 0; idx < task->regions.size(); idx++)
  {
  task->regions[idx].target_ranking.push_back(sys_mem);
  task->regions[idx].virtual_map = false;
  task->regions[idx].enable_WAR_optimization = war_enabled;
  task->regions[idx].reduction_list = false;
  task->regions[idx].blocking_factor = 1;
  } 
  return true;
  }
*/

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
