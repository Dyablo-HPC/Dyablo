/**
 * \file utils_block.h
 * \author Pierre Kestener
 */
#ifndef MUSCL_BLOCK_UTILS_H_
#define MUSCL_BLOCK_UTILS_H_

#include "shared/kokkos_shared.h"

namespace dyablo { namespace muscl_block {

using coord_t        = Kokkos::Array<uint32_t, 3>;
using blockSize_t    = Kokkos::Array<uint32_t, 3>;

// =======================================================
// =======================================================
/** 
 * Ordering of coordinates. Used in non-conformal interface update
 **/
struct CoordComparator {
  bool operator()(const coord_t &a, const coord_t &b) {
    if (a[0] < b[0])
      return true;
    else if (a[1] < b[1])
      return true;
    else
      return a[2] < b[2];
  }
};

// =======================================================
// =======================================================
/**
 * Flagging of faces, looking for non conformal neighbours
 * Storing result as bit mask on a 8 bits integer
 * If necessary to do the same with corners, then we will have
 * to switch to a 32 bits integer (26 possibilities in 3D)
 **/
enum ConformalStatus {
  TWO_TO_ONE_LEFT   = 0x01,
  TWO_TO_ONE_RIGHT  = 0x02,
  TWO_TO_ONE_DOWN   = 0x04,
  TWO_TO_ONE_UP     = 0x08,
  TWO_TO_ONE_FRONT  = 0x0F,
  TWO_TO_ONE_BACK   = 0x20
};

// =======================================================
// =======================================================
/**
 * input  : index to a given cell inside a block of data of size (bx,by,bz)
 * output : extract integer coordinates 
 */
template <int dim>
KOKKOS_INLINE_FUNCTION
coord_t index_to_coord(int32_t index, 
                       blockSize_t bSizes)
{
  const uint32_t& bx = bSizes[IX];
  const uint32_t& by = bSizes[IY];
  //const uint32_t& bz = bSizes[IZ];

  coord_t res;

  if (dim == 2) {

    res[IZ] = 0;
    res[IY] = (uint32_t)(index / bx);
    res[IX] = (uint32_t)(index - bx * res[IY]);    
  
  } else {

    res[IZ] = (uint32_t)(index / (bx * by));
    int32_t index2 = index - bx * by * res[IZ];
    res[IY] = (uint32_t)(index2 / bx);
    res[IX] = (uint32_t)(index2 - bx * res[IY]);

  }

  return res;

} // index_to_coord

// =======================================================
// =======================================================
/**
 * input  : index to a given cell inside a 2d block of data of size (bx,by)
 * output : extract integer coordinates 
 */
KOKKOS_INLINE_FUNCTION
coord_t index_to_coord(int32_t index, 
                       uint32_t bx,
                       uint32_t by)
{

  coord_t res;

  res[IZ] = 0;
  res[IY] = (uint32_t)(index / bx);
  res[IX] = (uint32_t)(index - bx * res[IY]);    
  
  return res;

} // index_to_coord

// =======================================================
// =======================================================
/**
 * input  : index to a given cell inside a 3d block of data of size (bx,by,bz)
 * output : extract integer coordinates 
 */
KOKKOS_INLINE_FUNCTION
coord_t index_to_coord(int32_t index, 
                       uint32_t bx,
                       uint32_t by,
                       uint32_t bz)
{

  coord_t res;

  res[IZ] = (uint32_t)(index / bx / by);

  uint32_t tmp = index - bx * by * res[IZ];

  res[IY] = (uint32_t)(tmp / bx);
  res[IX] = (uint32_t)(tmp - bx * res[IY]);    
  
  return res;

} // index_to_coord

// =======================================================
// =======================================================
/**
 * convert coordinates (i,j,k) of a given of a block of sizes (bx,by,bz)
 * into its corresponding linearized index in the same block but with 
 * ghost cells.
 */
template <int dim>
KOKKOS_INLINE_FUNCTION
uint32_t coord_to_index_g(coord_t     coords, 
                          blockSize_t bSizes,
                          uint32_t    ghostWidth)
{

  const uint32_t i = coords[IX] + ghostWidth;
  const uint32_t j = coords[IY] + ghostWidth;
  const uint32_t k = coords[IZ] + ghostWidth;

  const uint32_t& bx = bSizes[IX];
  const uint32_t& by = bSizes[IY];
  //const uint32_t& bz = bSizes[IZ];

  uint32_t res = dim == 2 ?
    i + (bx+2*ghostWidth)*j : 
    i + (bx+2*ghostWidth)*j + (bx+2*ghostWidth)*(by+2*ghostWidth)*k;

  return res;

} // coord_to_index_g

// =======================================================
// =======================================================
/**
 * convert coordinates (i,j) of a given of a block of sizes (bx,by)
 * into its corresponding linearized index in the same block but with 
 * ghost cells.
 */
KOKKOS_INLINE_FUNCTION
uint32_t coord_to_index_g(coord_t     coords, 
                          uint32_t    bx,
                          uint32_t    by,
                          uint32_t    ghostWidth)
{

  const uint32_t i = coords[IX] + ghostWidth;
  const uint32_t j = coords[IY] + ghostWidth;

  uint32_t res = i + (bx+2*ghostWidth)*j;

  return res;

} // coord_to_index_g

// =======================================================
// =======================================================
/**
 * convert coordinates (i,j,k) of a given of a block of sizes (bx,by,bz)
 * into its corresponding linearized index in the same block but with 
 * ghost cells.
 */
KOKKOS_INLINE_FUNCTION
uint32_t coord_to_index_g(coord_t     coords, 
                          uint32_t    bx,
                          uint32_t    by,
                          uint32_t    bz,
                          uint32_t    ghostWidth)
{

  const uint32_t i = coords[IX] + ghostWidth;
  const uint32_t j = coords[IY] + ghostWidth;
  const uint32_t k = coords[IZ] + ghostWidth;

  uint32_t res =  
    i + (bx+2*ghostWidth)*j + (bx+2*ghostWidth)*(by+2*ghostWidth)*k;

  return res;

} // coord_to_index_g

} // namespace muscl_block

} // namespace dyablo

#endif // MUSCL_BLOCK_UTILS_H_