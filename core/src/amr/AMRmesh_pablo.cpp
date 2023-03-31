#ifdef DYABLO_COMPILE_PABLO

#include "AMRmesh_pablo.h"

#include "UserData.h"

namespace dyablo {

void AMRmesh_pablo::loadBalance_userdata( uint8_t compact_levels, UserData& U )
{
#ifdef DYABLO_USE_MPI
    auto field_names_set = U.getEnabledFields();
    std::vector<std::string> field_names(field_names_set.begin(), field_names_set.end());

    using GatheredView = Kokkos::View<real_t***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    
    GatheredView U_gathered("U_gathered", U.getShape().U.extent(0), U.getShape().U.extent(2), U.nbFields());
    for( int i=0; i<field_names.size(); i++ )
    {
        auto U_gathered_slice = Kokkos::subview( U_gathered, Kokkos::ALL(), Kokkos::ALL(), i );
        auto U_field_slice = Kokkos::subview( U.getField(field_names[i]).U, Kokkos::ALL(), 0, Kokkos::ALL());
        Kokkos::deep_copy( U_gathered_slice, U_field_slice );
        U.delete_field(field_names[i]);
    }

    GatheredView Ughost_host; // Dummy ghost array
    using UserDataLB_t = UserDataLB<GatheredView, 1> ;
    UserDataLB_t data_lb(U_gathered, Ughost_host);
    ParaTree::loadBalance<UserDataLB_t>(data_lb, compact_levels);
    pmesh_epoch++;

    for( int i=0; i<field_names.size(); i++ )
    {
        auto U_gathered_slice = Kokkos::subview( U_gathered, Kokkos::ALL(), Kokkos::ALL(), i );
        U.new_fields({field_names[i]});
        auto U_field_slice = Kokkos::subview( U.getField(field_names[i]).U, Kokkos::ALL(), 0, Kokkos::ALL() );
        Kokkos::deep_copy( U_field_slice, U_gathered_slice );
    }

#endif // DYABLO_USE_MPI
}

} // namespace dyablo

#endif // DYABLO_COMPILE_PABLO