#include "matrix_array.hpp"


LMatrixArray::LMatrixArray
(const matArray& m) : array(m) {}

/*
void LMatrixArray::show(HighLevelRuntime *runtime, Context ctx) {
  for (size_t i=0; i<array.size(); i++)
    array[i].show(runtime, ctx, true); //wait
}

void LMatrixArray::destroy(HighLevelRuntime *runtime, Context ctx) {
  for (size_t i=0; i<array.size(); i++)
    array[i].destroy(runtime, ctx);
}
*/

LMatrix LMatrixArray::operator[](size_t i) const {
  assert(i>=0 && i<array.size());
  return array[i];
}

void LMatrixArray::operator+=(const LMatrixArray& rhs) {
  for (size_t i=0; i<rhs.arraySize(); i++)
    array.push_back(rhs[i]);
}

//--------------------------------------------
size_t LMatrixArray::legion_buffer_size(void) const
//--------------------------------------------
{
  size_t result = sizeof(size_t); // number of elements
  result += sizeof( LMatrix ) * array.size();
  return result;
}

//--------------------------------------------
size_t LMatrixArray::legion_serialize(void *buffer) const
//--------------------------------------------
{
  char *target = (char*)buffer; 
  *((size_t*)target) = array.size();
  target += sizeof(size_t);
  size_t numBytes = array.size() * sizeof(LMatrix);
  memcpy(target, &array[0], numBytes);
  target += numBytes;
  return (size_t(target) - size_t(buffer));
}

//--------------------------------------------
size_t LMatrixArray::legion_deserialize(const void *buffer)
//--------------------------------------------
{
  const char *source = (const char*)buffer;
  size_t numElems = *((const size_t*)source);
  source += sizeof(numElems);
  for (unsigned idx = 0; idx < numElems; idx++) {    
    LMatrix v = *((const LMatrix*)source);
    source += sizeof(v);
    array.push_back(v);
  }
  // Return the number of bytes consumed
  return (size_t(source) - size_t(buffer));
}

