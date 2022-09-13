/**
 * \file real_type.h
 * \brief Define macros to switch single/double precision.
 *
 * \author P. Kestener
 * \date 25-03-2010
 * 
 */
#ifndef REAL_TYPE_H_
#define REAL_TYPE_H_

#include <math.h>

/**
 * \typedef real_t (alias to float or double)
 */
#ifdef DYABLO_USE_DOUBLE
using real_t = double;
#else
using real_t = float;
#endif // DYABLO_USE_DOUBLE

// math function
#if defined(DYABLO_USE_DOUBLE)
#define FMAX(x,y) ( x>y ? x : y )
#define FMIN(x,y) ( x<y ? x : y )
#define SQRT(x) sqrt(x)
#define FABS(x) fabs(x)
#define COPYSIGN(x,y) copysign(x,y)
#define ISNAN(x) isnan(x)
#define FMOD(x,y) fmod(x,y)
#define ZERO_F (0.0)
#define HALF_F (0.5)
#define ONE_FOURTH_F (0.25)
#define ONE_F  (1.0)
#define TWO_F  (2.0)
#else
#define FMAX(x,y) ( x>y ? x : y )
#define FMIN(x,y) ( x<y ? x : y )
#define SQRT(x) sqrtf(x)
#define FABS(x) fabsf(x)
#define COPYSIGN(x,y) copysignf(x,y)
#define ISNAN(x) isnanf(x)
#define FMOD(x,y) fmodf(x,y)
#define ZERO_F (0.0f)
#define HALF_F (0.5f)
#define ONE_FOURTH_F (0.25f)
#define ONE_F  (1.0f)
#define TWO_F  (2.0f)
#endif // DYABLO_USE_DOUBLE

// other usefull macros
#define SQR(x) ((x)*(x))

#endif // REAL_TYPE_H_
