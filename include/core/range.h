#ifndef RANGE_H
#define RANGE_H

class Range {
 public:
 Range(                   ): mbegin(0),     msize(0)    {}
 Range(           int size): mbegin(0),     msize(size) {}
 Range(int begin, int size): mbegin(begin), msize(size) {}
  int begin() const {return mbegin;}
  int size()  const {return msize;}
  Range lchild() const;
  Range rchild() const;
 public:
  int mbegin;
  int msize;
};

// return the first half
inline Range Range::lchild () const
{
  int half_size = msize/2;
  return (Range){mbegin, half_size};
}

// return the second half
inline Range Range::rchild () const
{
  int half_size = msize/2;
  return (Range){mbegin+half_size, half_size};
}


#endif // RANGE_H
