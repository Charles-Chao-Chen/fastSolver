#include "range.h"


/* ---- Range class methods ---- */

Range Range::lchild () const
{
  int half_size = size/2;
  return (Range){begin, half_size};
}


Range Range::rchild () const
{
  int half_size = size/2;
  return (Range){begin+half_size, half_size};
}

