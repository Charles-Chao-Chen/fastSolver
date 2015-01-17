#ifndef RANGE_H
#define RANGE_H

class Range {
 public:
 Range(                   ): begin(0),     size(0)    {}
 Range(           int size): begin(0),     size(size) {}
 Range(int begin, int size): begin(begin), size(size) {}
  Range lchild() const;
  Range rchild() const;
 public:
  int begin;
  int size;
};

#endif // RANGE_H
