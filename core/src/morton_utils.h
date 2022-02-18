/**
 * \file morton_utils.h
 * Some usefull routines to compute / handle Morton indexes.
 *
 * Morton curve (or Z-curve) is a space filling curve,
 *  i.e. a mapping from an n-dimensional space to 1d).
 *
 * \sa https://en.wikipedia.org/wiki/Z-order_curve
 *
 *
 * Some of the following routines are adapted from:
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

#include <cstdint> // for uint32_t, uint64_t, etc...
#include <cstddef> // for std::size_t

#include "enums.h" // for ComponentIndex3D
#include "real_type.h"

#include <Kokkos_Macros.hpp> // for KOKKOS_INLINE_FUNCTION

namespace dyablo {

/**
 * Helper method to encode a Morton key from the cartesian coordinates.
 * Separate bits from a given integer "dim" positions apart and 
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
  uint32_t operator[](std::size_t i) const { return data[i]; }

  KOKKOS_INLINE_FUNCTION
  uint32_t& operator[](std::size_t i) { return data[i]; }

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
  uint32_t operator[](std::size_t i) const { return data[i]; }

  KOKKOS_INLINE_FUNCTION
  uint32_t& operator[](std::size_t i) { return data[i]; }

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

} // namespace dyablo

#endif // SHARED_MORTON_UTILS_H
