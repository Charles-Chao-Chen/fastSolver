#ifndef timer_h
#define timer_h

#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include <string>

// Return time in seconds since the Unix epoch
double timer(void);

class Timer {
public:
  void start();
  void stop();
  double get_elapsed_time();
  void   get_elapsed_time(const std::string);
private:
  double tStart;
  double tStop;
};

inline void Timer::start() {
  tStart = timer();
}

inline void Timer::stop() {
  tStop = timer();
}

inline double Timer::
get_elapsed_time() {
  return tStop-tStart;
}

inline void Timer::
get_elapsed_time(const std::string msg) {
  std::cout << "Time cost for " << msg
	    << "is : " << tStop-tStart
	    << std::endl;
}

inline double timer(void)
{
  double time;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  time = (double)tv.tv_sec + (double)tv.tv_usec/1.e6;
  return time;
}


#endif /* timer_h */
