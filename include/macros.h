#ifndef _MACROS_H
#define _MACROS_H

#include <sstream>
#include <stdexcept>

#include <stdlib.h>


// there is only one single field id
enum {
  FID_X,
};


// used in SaveRegionTask::TaskArgs
// for the length of the file name
//const int MAX_FILE_NAME_LENGTH = 50;

// used in InitCirculantKmatTask::TaskArgs
//  and NodeLaunchTask::TaskArgs
// for the subtree stored in array
const int MAX_TREE_SIZE = 150;



/* Uniform random number generator */
#define frand(xmin,xmax)			\
  ((double)xmin+(double)(xmax-xmin)*rand()/	\
   (double)RAND_MAX) 


/* Macros to throw exceptions */

#define ThrowException(msg) \
{std::stringstream s; s << __FILE__ << ":" << __LINE__ << ":" << __func__ << ": " << msg; \
throw std::runtime_error(s.str());}

#define CloseFileThrowException(f,msg) \
{fclose(f); std::stringstream s; s << __FILE__ << ":" << __LINE__ << ":" << __func__ << ": " << msg; \
throw std::runtime_error(s.str());}


#endif // _MACROS_H
