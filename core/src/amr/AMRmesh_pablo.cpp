#ifdef DYABLO_COMPILE_PABLO

#include "AMRmesh_pablo.h"

#include "UserData.h"

namespace dyablo {

void AMRmesh_pablo::loadBalance_userdata( uint8_t compact_levels, UserData& U )
{
#ifdef DYABLO_USE_MPI
    uint32_t nbOcts_old = this->getNumOctants();
    
    ParaTree::loadBalance(compact_levels);
    pmesh_epoch++;

    ParaTree::ExchangeRanges pablo_send_ranges = ParaTree::getLoadBalanceRanges().sendRanges;
    
    int mpi_rank = GlobalMpiSession::get_comm_world().MPI_Comm_rank();;

    Kokkos::View< int* > target_domains( "target_domains", nbOcts_old );
    {
        auto target_domains_host = Kokkos::create_mirror_view(target_domains);

        for( int i=0; i<target_domains.size(); i++  )
            target_domains_host[i] = mpi_rank;
        for( const auto& [target_rank, interval] : pablo_send_ranges )
        {
            for( int i = interval[0]; i<interval[1]; i++ )
                target_domains_host[i] = target_rank;
        }

        Kokkos::deep_copy( target_domains, target_domains_host );
    }

    ViewCommunicator ghost_comm( target_domains );
    U.exchange_loadbalance( ghost_comm );
#endif // DYABLO_USE_MPI
}

} // namespace dyablo

#endif // DYABLO_COMPILE_PABLO