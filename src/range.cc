#include "range.h"


/* ---- Range class methods ---- */

Range Range::lchild () const
// return the first half
{
  int half_size = size/2;
  return (Range){begin, half_size};
}


Range Range::rchild () const
// return the second half
{
  int half_size = size/2;
  return (Range){begin+half_size, half_size};
}

