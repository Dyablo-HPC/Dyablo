#pragma once

#include "shared/kokkos_shared.h"
#include "shared/userdata_utils.h"

namespace bitpit{

template< typename T >
class DataLBInterface;

} // namespace bitpit

namespace dyablo {

/**
 * Class used by ParaTree::loadBalance to manage user data
 * @tparam DataArray_t Kokkos::View type to contain user data. This must be Host-accessible
 * @tparam iOct_pos position of octant index for DataArray_t::operator()
 **/
template< typename DataArray_t, int iOct_pos >
class UserDataLB : public bitpit::DataLBInterface<UserDataLB<DataArray_t, iOct_pos>>
{
static_assert( Kokkos::SpaceAccessibility< typename DataArray_t::memory_space, Kokkos::HostSpace >::accessible, 
                 "UserDataLB can only use Kokkos views accessible on Host" );

public:
  using Policy_t = Kokkos::RangePolicy<Kokkos::OpenMP>; //! Use OpenMP to iterate over data
  using Layout_t = typename DataArray_t::array_layout;

  DataArray_t& data;
  DataArray_t& ghostdata;

  Layout_t layout;
  uint32_t vals_per_oct = 1;

  UserDataLB( DataArray_t& data, DataArray_t& ghostdata )
    : data(data), ghostdata(ghostdata), layout(data.layout())
  {
    assert( data.size() != 0 );
    layout.dimension[iOct_pos] = 1;
    for(int i=0; i<DataArray_t::rank; i++)
    {
      vals_per_oct *= layout.dimension[i];
    }
  }

  size_t fixedSize() const 
  {
    return vals_per_oct*sizeof(typename DataArray_t::value_type);
  }

  size_t size(const uint32_t) const 
  {
    return fixedSize();
  }

  void resize(uint32_t newSize)
  {
    Layout_t new_layout = layout;
    new_layout.dimension[iOct_pos]=newSize;
    Kokkos::resize( data, new_layout );
  }

  void resizeGhost(uint32_t newSize)
  {
    Layout_t new_layout = layout;
    new_layout.dimension[iOct_pos]=newSize;
    Kokkos::resize( ghostdata, new_layout );
  }

  template<class Buffer>
  void gather(Buffer & buff, const uint32_t iOct)
  {
    using userdata_utils::get_U;
    for(uint32_t i=0; i<vals_per_oct; i++)
      buff << get_U<iOct_pos>(data, iOct, i);
  }

  template<class Buffer>
  void scatter(Buffer & buff, const uint32_t iOct)
  {
    using userdata_utils::get_U;
    for(uint32_t i=0; i<vals_per_oct; i++)
      buff >> get_U<iOct_pos>(data, iOct, i);
  }

  void move( uint32_t from, uint32_t to )
  {
    using userdata_utils::get_U;
    for(uint32_t i=0; i<vals_per_oct; i++)
    {
      get_U<iOct_pos>(data, to, i) = get_U<iOct_pos>(data, from, i);
    }
  }

  void assign( uint32_t stride, uint32_t length )
  {
    using userdata_utils::get_U;

    DataArray_t dataCopy("dataLBcopy");

    Kokkos::parallel_for("dyablo::muscl_block::UserDataLB::assign(1)",
      Policy_t{0, length}, [&](uint32_t iOct)
    {
      for(int i=0; i<vals_per_oct; i++)
        get_U<iOct_pos>(dataCopy, iOct, i) = get_U<iOct_pos>(data, iOct+stride, i);
    });

    Kokkos::parallel_for("dyablo::muscl_block::UserDataLB::assign(2)",
      Policy_t{0, length}, [&](uint32_t iOct)
    {
      for(int i=0; i<vals_per_oct; i++)
        get_U<iOct_pos>(data, iOct, i) = get_U<iOct_pos>(dataCopy, iOct, i);
    });
  }

  void shrink(){}
};

} // namespace dyablo