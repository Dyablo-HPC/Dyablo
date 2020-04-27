/**
 * Some usefull routines to compute / handle Morton indexes.
 *
 * Morton curve (or Z-curve) is a space filling curve,
 *  i.e. a mapping from an n-dimensional space to 1d).
 *
 * \sa https://en.wikipedia.org/wiki/Z-order_curve
 */

/* Some of the following routines are adapted from:
 * http://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/
 *
 * This blog provides performances comparison between 3 methods:
 * - "For loop" method
 * - "Magic bits" method
 * - "Lookup table" method
 *
 * Here, we only use the "Magic bits" method, also used PABLO.
 * 
 * Other interesting references:
 * - libmorton, https://github.com/Forceflow/libmorton
 * - mortonlib, https://github.com/aavenel/mortonlib
 * - http://bitmath.blogspot.fr/2012/11/tesseral-arithmetic-useful-snippets.html
 *
 */
#ifndef SHARED_MORTON_UTILS_H
#define SHARED_MORTON_UTILS_H

#include <cstdint>

#include "shared/enums.h" // for ComponentIndex3D

#include <Kokkos_Macros.hpp> // for KOKKOS_INLINE_FUNCTION

namespace dyablo {

/**
 * Helper method to encode a Morton key from the cartesian coordinates.
 * Seperate bits from a given integer "dim" positions apart and 
 * inserting zeros in between.
 *
 * e.g. "100" becomes "01|00|00"    in 2D
 *      "100" becomes "001|000|000" in 3D
 */
template<int dim>
KOKKOS_INLINE_FUNCTION
uint64_t splitBy3(uint32_t a);

/* 1D version */
template<>
KOKKOS_INLINE_FUNCTION
uint64_t splitBy3<1>(uint32_t a)
{
  
  uint64_t x = a;

  return x;
  
} // splitBy3<1>

/* 2D version */
template<>
KOKKOS_INLINE_FUNCTION
uint64_t splitBy3<2>(uint32_t a)
{

  // we take all 32 bits (this is different in 3D)
  // because the result must hold in an uint64_t ( 2x21=63 bits --> uint64_t is enought )  
  uint64_t x = a & 0xffffffff;
  x = (x | x << 16) & 0xffff0000ffff;
  x = (x | x << 8) & 0xff00ff00ff00ff;
  x = (x | x << 4) & 0xf0f0f0f0f0f0f0f;
  x = (x | x << 2) & 0x3333333333333333;
  x = (x | x << 1) & 0x5555555555555555;

  return x;
  
} // splitBy3<2>

/* 3D version */
template<>
KOKKOS_INLINE_FUNCTION
uint64_t splitBy3<3>(uint32_t a)
{

  // we only look at the first 21 bits
  // because the result must hold in an uint64_t :
  // 3x21=63 bits --> uint64_t is enought )

  uint64_t x = a & 0x1fffff;
  
  x = (x | x << 32) & 0x1f00000000ffff;  // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
  x = (x | x << 16) & 0x1f0000ff0000ff;  // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
  x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
  x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
  x = (x | x << 2) & 0x1249249249249249;

  return x;
    
} // splitBy3<3>

/**
 * A simple class holding several integers.
 */
template <int dim>
struct index_t;

template<>
struct index_t<2>{

  static constexpr int ndim = 2;

  uint32_t data[ndim];

  KOKKOS_INLINE_FUNCTION
  index_t() {}

  KOKKOS_INLINE_FUNCTION
  index_t(int i, int j) { data[0]=i; data[1]=j; }

  KOKKOS_INLINE_FUNCTION
  uint32_t operator[](size_t i) const { return data[i]; }

  KOKKOS_INLINE_FUNCTION
  uint32_t& operator[](size_t i) { return data[i]; }

}; // index_t<2>

template<>
struct index_t<3>{

  static constexpr int ndim = 3;

  uint32_t data[ndim];

  KOKKOS_INLINE_FUNCTION
  index_t() {}

  KOKKOS_INLINE_FUNCTION
  index_t(int i, int j, int k) { data[0]=i; data[1]=j; data[2]=k; }

  KOKKOS_INLINE_FUNCTION
  uint32_t operator[](size_t i) const { return data[i]; }

  KOKKOS_INLINE_FUNCTION
  uint32_t& operator[](size_t i) { return data[i]; }

}; // index_t<3>


/**
 * Encode / compute Morton key from integer cartesian coordinate (i,j,k).
 *
 * In 2D, cartesian coordinates x,y can be as large as 2^32 (about 4e9)
 * In 3D, cartesian coordinates x,y,z must be lower than 2^21 = 2097152
 *
 * \param[in] index cartesian coordinates (x,y,z)
 * \return the Morton key.
 */
template<int dim>
KOKKOS_INLINE_FUNCTION
uint64_t compute_morton_key(const index_t<dim>& index)
{
  uint64_t key = 0;
  if (dim == 1) {
    key = index[IX];
  } else if (dim == 2) {
    key |= splitBy3<dim>(index[IX]) | splitBy3<dim>(index[IY]) << 1;
  } else if (dim == 3) {
    key |= splitBy3<dim>(index[IX]) | splitBy3<dim>(index[IY]) << 1 | splitBy3<dim>(index[IZ]) << 2;
  }
  
  return key;
  
} // compute_morton_key

/** another 2d version of morton key */
KOKKOS_INLINE_FUNCTION
uint64_t compute_morton_key(const uint32_t ix, const uint32_t iy)
{
  uint64_t key = 0;
  key |= splitBy3<2>(ix) | splitBy3<2>(iy) << 1;
  return key;
  
} // compute_morton_key - 2d

/** another 3d version of morton key */
KOKKOS_INLINE_FUNCTION
uint64_t compute_morton_key(const uint32_t ix, const uint32_t iy, const uint32_t iz)
{
  uint64_t key = 0;
  key |= splitBy3<3>(ix) | splitBy3<3>(iy) << 1 | splitBy3<3>(iz) << 2;
  return key;
  
} // compute_morton_key - 3d


/**
 * Helper method for Magicbits Morton key decoding.
 *
 * Returned value is encoded on 32 bits.
 * In 2D, all bits can be significant.
 * In 3D, only the last 21 bits are significant, i.e. the returned value 
 * must be lower than 2^21=2097152.
 *
 * Template parameter allows to extract either x,y or z coordinate.
 * \tparam dim should be 2 or 3
 * \tparam coord should be IX, IY or IZ (from enums)
 */
template<int dim, int coord>
KOKKOS_INLINE_FUNCTION
uint32_t morton_extract_bits(uint64_t key)
{
  // shift bit by dimension and thus select which coordinate to extract
  key = key >> coord;
  
  if (dim==2) {
    // 5 is "0101" in binary
    // just mask bits to zero out bits at odd positions
    // so that one can extract one out of 2 bits
    key &= 0x5555555555555555;
    
    key = (key ^ (key >> 1))  & 0x3333333333333333;
    key = (key ^ (key >> 2))  & 0x0f0f0f0f0f0f0f0f;
    key = (key ^ (key >> 4))  & 0x00ff00ff00ff00ff;
    key = (key ^ (key >> 8))  & 0x0000ffff0000ffff;
    key = (key ^ (key >> 16)) & 0x00000000ffffffff;

  } else if (dim==3) {
    // 249 pattern is |0010|0100|1001| in binary notation
    // so that on can extract one out of 3 bits
    key &= 0x1249249249249249;
    key = (key ^ (key >> 2))  & 0x30c30c30c30c30c3;
    key = (key ^ (key >> 4))  & 0xf00f00f00f00f00f;
    key = (key ^ (key >> 8))  & 0x00ff0000ff0000ff;
    key = (key ^ (key >> 16)) & 0x00ff00000000ffff;
    key = (key ^ (key >> 32)) & 0x1fffff;
  }
  
  return static_cast<uint32_t>(key);

} // morton_extract_bits

/**
 * Get morton key of a face neighbor at same level
 * given current octant Morton key.
 *
 * \param[in] key is the morton key of current octant
 * \param[in] level AMR level of current octant
 * \param[in] face is faceId (FACE_XMIN, FACE_XMAX, ....)
 *
 * \return neighbor morton key
 *
 * \todo maybe try to extract only the coordinate of interest not all 
 *       three coordinates.
 * \todo all the addition could probably be implemented more efficiently
 *       with bit manipulation
 *
 */
KOKKOS_INLINE_FUNCTION
uint64_t get_neighbor_morton(uint64_t key,
                             uint8_t level,
                             uint8_t face)
{
  auto x = morton_extract_bits<3,IX>(key);
  auto y = morton_extract_bits<3,IY>(key);
  auto z = morton_extract_bits<3,IZ>(key);

  constexpr int MAX_LEVEL = 20;
  auto length = uint32_t(1) << (MAX_LEVEL - level);

  // domain length
  auto total_length = uint32_t(1) << MAX_LEVEL;

  if (face == 0) 
  {
    if (x < length)
      x += total_length;
    x -= length;
  }

  if (face == 1)
  {
    x += length;
    if (x >= total_length)
      x -= total_length;
  }

  if (face == 2) 
  {
    if (y < length)
      y += total_length;
    y -= length;
  }

  if (face == 3)
  {
    y += length;
    if (y >= total_length)
      y -= total_length;
  }

  if (face == 4) 
  {
    if (z < length)
      z += total_length;
    z -= length;
  }

  if (face == 5)
  {
    z += length;
    if (z >= total_length)
      z -= total_length;
  }

  return compute_morton_key(x,y,z);

} // get_neighbor_morton

/**
 * Get morton key of a face neighbor at same level
 * given current octant Morton key.
 *
 * \param[in] key is the morton key of current octant
 * \param[in] level AMR level of current octant
 * \param[in] level_n AMR level of neigh octant
 * \param[in] face is faceId (FACE_XMIN, FACE_XMAX, ....)
 * \param[in] neigh_id is neighbor id
 *
 * \return neighbor morton key
 *
 * \note neigh_id should 0 or 1 in 2D
 * \note neigh_id should 0, 1, 2 or 3 in 3D
 *
 * \todo maybe try to extract only the coordinate of interest not all 
 *       three coordinates.
 * \todo all the addition could probably be implemented more efficiently
 *       with bit manipulation
 *
 */
KOKKOS_INLINE_FUNCTION
uint64_t get_neighbor_morton(uint64_t key,
                             uint8_t level,
                             uint8_t level_n,
                             uint8_t face,
                             uint8_t neigh_id)
{
  
  uint32_t xyz[3];
  xyz[IX] = morton_extract_bits<3,IX>(key);
  xyz[IY] = morton_extract_bits<3,IY>(key);
  xyz[IZ] = morton_extract_bits<3,IZ>(key);

  constexpr int MAX_LEVEL = 20;

  // length of current octant
  auto length = uint32_t(1) << (MAX_LEVEL - level);

  // length of neighbor
  auto length_n = uint32_t(1) << (MAX_LEVEL - level_n);

  // domain length
  auto total_length = uint32_t(1) << MAX_LEVEL;

  // get direction and left/right face
  auto dir = face >> 1;
  auto iface = face & 0x1;

  uint32_t b0 =  neigh_id     & 0x1 ;
  uint32_t b1 = (neigh_id>>1) & 0x1 ;
  
  // neighbor is smaller
  if (level_n > level)
  {

    if (dir == IX)
    {

      xyz[IX] = iface==0 ? xyz[IX]-length_n : xyz[IX]+length;
      if (xyz[IX]<0)
        xyz[IX] += total_length;
      if (xyz[IX]>=total_length)
        xyz[IX] -= total_length;

      xyz[IY] = b0   ==0 ? xyz[IY]          : xyz[IY]+length_n;
      xyz[IZ] = b1   ==0 ? xyz[IZ]          : xyz[IZ]+length_n;
    }

    if (dir == IY)
    {
      xyz[IX] = b0   ==0 ? xyz[IX]          : xyz[IX]+length_n;

      xyz[IY] = iface==0 ? xyz[IY]-length_n : xyz[IY]+length;
      if (xyz[IY]<0)
        xyz[IY] += total_length;
      if (xyz[IY]>=total_length)
        xyz[IY] -= total_length;

      xyz[IZ] = b1   ==0 ? xyz[IZ]          : xyz[IZ]+length_n;
    }

    if (dir == IZ)
    {
      xyz[IX] = b0   ==0 ? xyz[IX]          : xyz[IX]+length_n;
      xyz[IY] = b1   ==0 ? xyz[IY]          : xyz[IY]+length_n;
      
      xyz[IZ] = iface==0 ? xyz[IZ]-length_n : xyz[IZ]+length;
      if (xyz[IZ]<0)
        xyz[IZ] += total_length;
      if (xyz[IZ]>=total_length)
        xyz[IZ] -= total_length;

    }

    return compute_morton_key(xyz[IX],xyz[IY],xyz[IZ]);

  } // end neighbor is smaller

  // neighbor is larger
  if (level_n < level)
  {

    if (dir == IX)
    {
      xyz[IX] = iface == 0 ? xyz[IX]-length_n : xyz[IX]+length;
      if (xyz[IX]<0)
        xyz[IX] += total_length;
      if (xyz[IX]>=total_length)
        xyz[IX] -= total_length;

      xyz[IY] = b0   ==0 ? xyz[IY]          : xyz[IY]-length;
      xyz[IZ] = b1   ==0 ? xyz[IZ]          : xyz[IZ]-length;
    }

    if (dir == IY)
    {
      xyz[IX] = b0   ==0 ? xyz[IX]          : xyz[IX]-length;
      
      xyz[IY] = iface == 0 ? xyz[IY]-length_n : xyz[IY]+length;
      if (xyz[IY]<0)
        xyz[IY] += total_length;
      if (xyz[IY]>=total_length)
        xyz[IY] -= total_length;

      xyz[IZ] = b1   ==0 ? xyz[IZ]          : xyz[IZ]-length;
    }

    if (dir == IZ)
    {
      xyz[IX] = b0   ==0 ? xyz[IX]          : xyz[IX]-length;
      xyz[IY] = b1   ==0 ? xyz[IY]          : xyz[IY]-length;

      xyz[IZ] = iface == 0 ? xyz[IZ]-length_n : xyz[IZ]+length;
      if (xyz[IZ]<0)
        xyz[IZ] += total_length;
      if (xyz[IZ]>=total_length)
        xyz[IZ] -= total_length;

    }

    return compute_morton_key(xyz[IX],xyz[IY],xyz[IZ]);

  } // end neighbor is larger

  // should never reach here
  return compute_morton_key(0,0,0);

} // get_neighbor_morton

} // namespace dyablo

#endif // SHARED_MORTON_UTILS_H
