/**
 * \file utils_block.h
 * \author Pierre Kestener
 */
#ifndef MUSCL_BLOCK_UTILS_H_
#define MUSCL_BLOCK_UTILS_H_

#include "shared/kokkos_shared.h"

KOKKOS_INLINE_FUNCTION
Kokkos::Array<int32_t,3> compute_cell_coord(int32_t index, 
                                            int32_t bx,
                                            int32_t by,
                                            int32_t bz)
{
  Kokkos::Array<int,3> res;

  res[IZ] = index / (bx*by);
  int32_t index2 = index - bx*by*res[IZ]; 
  res[IY] = index2 / bx;
  res[IX] = index2 - bx*res[IY];

  return res;

} // compute_cell_coord

#endif // MUSCL_BLOCK_UTILS_H_
