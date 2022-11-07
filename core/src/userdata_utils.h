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


  /**
   * Transfert values from U_right_iOct to U
   * When iOct is not rightmost index in U
   **/
  template <int iOct_pos, typename DataArray_t>
  std::enable_if_t< iOct_pos < DataArray_t::rank-1 , 
  void > transpose_from_right_iOct( const DataArray_t& U_right_iOct, DataArray_t& U )
  {
    // When iOct is not the rightmost index, a temportary MPI buffer 
    // with iOct rightmost index is used and has to be transposed

    // Realloc U with the correct number of octants
    auto layout_U = U.layout();
    for(int i=0; i<DataArray_t::rank; i++)
    {
      if( i < iOct_pos )
        layout_U.dimension[i] = U_right_iOct.extent(i);
      else if( i == iOct_pos )
        layout_U.dimension[i] = U_right_iOct.extent(DataArray_t::rank-1);
      else // (i > iOct_pos)
        layout_U.dimension[i] = U_right_iOct.extent(i-1);
    }
    Kokkos::realloc(U, layout_U);

    uint32_t elts_per_octs = octant_size<DataArray_t, iOct_pos>(U);

    // Transpose value from U_right_iOct to U
    Kokkos::parallel_for( "transpose_from_right_iOct", U_right_iOct.size(),
                          KOKKOS_LAMBDA(uint32_t index)
    {
      uint32_t iOct = index/elts_per_octs;
      uint32_t i = index%elts_per_octs;
      
      get_U<iOct_pos>(U, iOct, i) = get_U<DataArray_t::rank-1>(U_right_iOct, iOct, i);
    });
  }

  /**
   * Transfert values from U_right_iOct to U
   * When iOct is rightmost index un U there is nothing to transpose 
   **/
  template <int iOct_pos, typename DataArray_t>
  std::enable_if_t< iOct_pos == DataArray_t::rank-1 , 
  void > transpose_from_right_iOct( const DataArray_t& U_right_iOct, DataArray_t& U )
  {
    U = U_right_iOct;
  }

  /**
   * Transfert values from U to U_right_iOct
   * When iOct is not rightmost index in U
   **/
  template <int iOct_pos, typename DataArray_t>
  std::enable_if_t< iOct_pos < DataArray_t::rank-1 , 
  void > transpose_to_right_iOct( const DataArray_t& U, DataArray_t& U_right_iOct )
  {
    // When iOct is not the rightmost index, a temportary MPI buffer 
    // with iOct rightmost index is used and has to be transposed

    // Realloc U_right_iOct with the correct number of octants
    auto layout_right_iOct = U.layout();
    for(int i=0; i<DataArray_t::rank-1; i++)
    {
      if( i < iOct_pos )
        layout_right_iOct.dimension[i] = U.extent(i);
      else // (i >= iOct_pos)
        layout_right_iOct.dimension[i] = U.extent(i+1);
    }
    layout_right_iOct.dimension[DataArray_t::rank-1] = U.extent(iOct_pos);

    Kokkos::realloc(U_right_iOct, layout_right_iOct);

    uint32_t elts_per_octs = octant_size<DataArray_t, iOct_pos>(U);

    // Transpose value from U_right_iOct to U
    Kokkos::parallel_for( "transpose_to_right_iOct", U.size(),
                          KOKKOS_LAMBDA(uint32_t index)
    {
      uint32_t iOct = index/elts_per_octs;
      uint32_t i = index%elts_per_octs;
      
      get_U<DataArray_t::rank-1>(U_right_iOct, iOct, i) = get_U<iOct_pos>(U, iOct, i);
    });
  }

  template <int iOct_pos, typename DataArray_t>
  std::enable_if_t< iOct_pos == DataArray_t::rank-1 , 
  void > transpose_to_right_iOct( const DataArray_t& U, DataArray_t& U_right_iOct )
  {
    U_right_iOct = U;
  }


}