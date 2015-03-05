#ifndef region_array_hpp
#define region_array_hpp

#include "legion_matrix.h"
//#include "hodlr_matrix.h"

/**
 * \class LMatrixArray
 * This is a decorator class that helps the Legion compiler
 * with returning an array of legion_matrix as the result
 * of legion tasks.
 */
class LMatrixArray {
  typedef std::vector<LMatrix> matArray;
public:
  LMatrixArray(void) { }
  LMatrixArray(const matArray&);
  //void get_regions(const HodlrMatrix&);
  
  /* void show(HighLevelRuntime *runtime, Context ctx);
  void destroy(HighLevelRuntime *runtime, Context ctx); */
  size_t arraySize() const {return array.size();}
  LMatrix operator[](size_t i) const;
  void operator+=(const LMatrixArray&);
public:
  size_t legion_buffer_size(void) const;
  size_t legion_serialize(void *buffer) const;
  size_t legion_deserialize(const void *buffer);
public:
  inline matArray& ref(void) { return array; }
private:
  matArray array;
};


#endif // region_array_hpp
