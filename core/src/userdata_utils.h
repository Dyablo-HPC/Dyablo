#pragma once

#include "kokkos_shared.h"

namespace userdata_utils{
  template< int N >
  struct integer_t{};

  template<int iOct_pos, typename DataArray_t, typename Nargs, typename... Args>
  struct get_U_t{
    KOKKOS_INLINE_FUNCTION
    static typename DataArray_t::value_type& get_U( const DataArray_t& U, uint32_t iOct, uint32_t elt_index, Args... is)
    {
      uint32_t current_dim_size = U.extent(sizeof...(Args));
      uint32_t rem = elt_index%current_dim_size;
      uint32_t div = elt_index/current_dim_size;
      return get_U_t<iOct_pos, DataArray_t, integer_t<sizeof...(Args)+1>, Args..., uint32_t>::get_U(U, iOct, div, is..., rem);
    }
  };

  template<int iOct_pos, typename DataArray_t, typename... Args>
  struct get_U_t<iOct_pos, DataArray_t, integer_t<iOct_pos>, Args...>{
    KOKKOS_INLINE_FUNCTION
    static typename DataArray_t::value_type& get_U( const DataArray_t& U, uint32_t iOct, uint32_t elt_index, Args... is)
    {
      return get_U_t<iOct_pos, DataArray_t, integer_t<sizeof...(Args)+1>, Args..., uint32_t>::get_U(U, iOct, elt_index, is..., iOct);
    }
  };

  template<int iOct_pos, typename DataArray_t, typename... Args>
  struct get_U_t<iOct_pos, DataArray_t, integer_t<DataArray_t::rank>, Args...>{
    KOKKOS_INLINE_FUNCTION
    static typename DataArray_t::value_type& get_U( const DataArray_t& U, uint32_t /*iOct*/, uint32_t /*elt_index*/, Args... is)
    {
      return U(is...);
    } 
  };

  /**
   * Get Value of U(...,iOct,...)
   * @tparam DataArray_t Kokkos::View containing user data
   * @tparam iOct_pos position of iOct parameter for DataArray_t::operator()
   * @param U Kokkos::View to acces
   * @param iOct octant index
   * @param elt_index linearized index for elements inside octant
   * NOTE : leftmost non-iOct index is contiguous (LayoutLeft)
   **/
  template<int iOct_pos, typename DataArray_t>
  KOKKOS_INLINE_FUNCTION
  typename DataArray_t::value_type& get_U( const DataArray_t& U, uint32_t iOct, uint32_t elt_index)
  {
    return get_U_t<iOct_pos, DataArray_t, integer_t<0>>::get_U(U, iOct, elt_index);
  }

  template < typename DataArray_t, int iOct_pos >
  uint32_t octant_size( const DataArray_t& U )
  {
    // Compute number of values per octants
    uint32_t elts_per_octs = 1;
    for(int i=0; i<DataArray_t::rank; i++)
      if( i != iOct_pos )
        elts_per_octs *= U.extent(i);
    return elts_per_octs;
  }
}