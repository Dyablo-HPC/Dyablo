#ifndef MATH_OPERATORS_COMMON_H_
#define MATH_OPERATORS_COMMON_H_

template<class T> 
const T& mymin(const T& a, const T& b)
{
  return (b < a) ? b : a;
}

template<class T> 
const T& mymax(const T& a, const T& b)
{
    return (a < b) ? b : a;
}

#endif // MATH_OPERATORS_COMMON_H_
