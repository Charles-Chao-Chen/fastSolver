#ifndef RANGE_H
#define RANGE_H

class Range {
 public:
 Range(                   ): begin(0),     size(0)    {}
 Range(           int size): begin(0),     size(size) {}
 Range(int begin, int size): begin(begin), size(size) {}
  //int begin() const {return mbegin;}
  //int size()  const {return msize;}
  Range lchild() const;
  Range rchild() const;
 public:
  int begin;
  int size;
};

// return the first half
inline Range Range::lchild () const
{
  int half_size = size/2;
  return (Range){begin, half_size};
}

// return the second half
inline Range Range::rchild () const
{
  int half_size = size/2;
  return (Range){begin+half_size, half_size};
}


#endif // RANGE_H
