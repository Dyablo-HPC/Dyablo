#include "GhostCommunicator.h"

namespace dyablo{


namespace GhostCommunicator_pablo_impl{

/** 
 * @function gather_aux
 * Auxiliary function for UserDataComm::gather()
 * This is used to generate different gather() versions according to rank of view
 * Equivalent DataArray_t::rank == 3 :
 * ```
 *  void gather(Buffer & buff, const uint32_t iOct) const {
 *    for(uint32_t i1=0; i1<data.extent(1); i1++)
 *      for(uint32_t i0=0; i0<data.extent(0); i0++)
 *      {
 *        buff << data( i0, i1, iOct );
 *      }
 *  }
 * ```
* @tparam iOct_pos is the position of the iOct parameter in the array subscripts
 *         ex when iOct pos = 1 in a 3D array : buff << data( i0, iOct, i1 );
 **/
template<int N, int iOct_pos, typename Buffer, typename DataArray_t, typename... Args>
std::enable_if_t< N == -1, void >
gather_aux(Buffer& buff, const DataArray_t& data, uint32_t iOct, Args... is)
{
  buff << data( is... );
}
template<int N, int iOct_pos, typename Buffer, typename DataArray_t, typename... Args>
std::enable_if_t< N >= 0, void >
gather_aux(Buffer& buff, DataArray_t& data, uint32_t iOct, Args... is)
{
  if( N == iOct_pos )
  {
    gather_aux<N-1, iOct_pos>(buff, data, iOct, iOct, is...);
  }
  else
  {
    for(size_t i=0; i<data.extent(N); i++)
    {
      gather_aux<N-1, iOct_pos>(buff, data, iOct, i, is...);
    }
  }
}

/** 
 * @function scatter_aux
 * Auxiliary function for UserDataComm::scatter()
 * This is used to generate different scatter() versions according to rank of view
 * Equivalent DataArray_t::rank == 3 :
 * ```
 * void scatter(Buffer & buff, const uint32_t iOct) const {
 *   for(uint32_t i1=0; i1<data.extent(1); i1++)
 *     for(uint32_t i0=0; i0<data.extent(0); i0++)
 *     {
 *       buff >> ghostData( i0, i1, iOct );
 *     }
 * }
 * ```
 * @tparam iOct_pos is the position of the iOct parameter in the array subscripts
 *         ex when iOct pos = 1 in a 3D array : buff >> ghostData( i0, iOct, i1 );
 **/  
template<int N, int iOct_pos, typename Buffer, typename DataArray_t, typename... Args>
std::enable_if_t< N == -1, void >
scatter_aux(Buffer& buff, const DataArray_t& data, uint32_t iOct, Args... is)
{
  buff >> data( is... );
}
template<int N, int iOct_pos, typename Buffer, typename DataArray_t, typename... Args>
std::enable_if_t< N >= 0, void >
scatter_aux(Buffer& buff, DataArray_t& data, uint32_t iOct, Args... is)
{
  if( N == iOct_pos )
  {
    scatter_aux<N-1, iOct_pos>(buff, data, iOct, iOct, is...);
  }
  else
  {
    for(size_t i=0; i<data.extent(N); i++)
    {
      scatter_aux<N-1, iOct_pos>(buff, data, iOct, i, is...);
    }
  }
}

/**
 * Callback class to serialize/deserialize user data for PABLO MPI communication
 * (block based version)
 */
template<typename DataArray_t, int iOct_pos>
class UserDataComm : public bitpit::DataCommInterface<UserDataComm<DataArray_t, iOct_pos>>
{

public:
  //! bulk data owned by current MPI process
  DataArray_t data;

  //! ghost data, owned by a different MPI process
  DataArray_t ghostData;

  size_t elts_per_oct;

  UserDataComm(DataArray_t data_, DataArray_t ghostData_) :
    data(data_),
    ghostData(ghostData_)
  {
    elts_per_oct = 1;
    for( int i=0; i<DataArray_t::rank-1; i++ )
    {
      elts_per_oct *= data.extent(i);
    }
  }; // UserDataComm::UserDataComm
  

  //! number of bytes per cell to exchange
  size_t fixedSize() const
  {
    return sizeof(typename DataArray_t::value_type)*elts_per_oct;
  }


  template<class Buffer>
  void gather(Buffer & buff, const uint32_t iOct) const {
    gather_aux<DataArray_t::rank-1, iOct_pos>( buff, data, iOct );
  }

  /// Fill ghosts with data received from neighbor MPI processes.
  template<class Buffer>
  void scatter(Buffer & buff, const uint32_t iOct) const {
    scatter_aux<DataArray_t::rank-1, iOct_pos>( buff, ghostData, iOct );
  }

  ~UserDataComm(){};
  
}; // class UserDataComm

template< typename DataArray_t, int iOct_pos = DataArray_t::rank-1 >
void exchange_ghosts_aux( AMRmesh_pablo& amr_mesh, 
                          const DataArray_t& U, const DataArray_t& Ughost)
{
  assert(U.extent( iOct_pos ) == amr_mesh.getNumOctants()); // Specified index must be iOct

  assert( Ughost.extent( iOct_pos ) == amr_mesh.getNumGhosts()  );

  using DataArray_host_t = typename DataArray_t::HostMirror;

  // Copy Data to host for MPI communication 
  DataArray_host_t U_host = Kokkos::create_mirror_view(U);
  DataArray_host_t Ughost_host = Kokkos::create_mirror_view(Ughost);
  Kokkos::deep_copy(U_host, U);

  UserDataComm<DataArray_host_t, iOct_pos> data_comm(U_host, Ughost_host);
  amr_mesh.communicate(data_comm);

  // Copy back ghosts to Device
  Kokkos::deep_copy(Ughost, Ughost_host);
}

} //namespace

void GhostCommunicator_pablo::exchange_ghosts(const DataArrayBlock& U, const DataArrayBlock& Ughost) const
{
    GhostCommunicator_pablo_impl::exchange_ghosts_aux(amr_mesh, U, Ughost);
}

void GhostCommunicator_pablo::exchange_ghosts(const DataArray& U, const DataArray& Ughost) const
{
    GhostCommunicator_pablo_impl::exchange_ghosts_aux<DataArray, 0>(amr_mesh, U, Ughost);
}


}//namespace dyablo