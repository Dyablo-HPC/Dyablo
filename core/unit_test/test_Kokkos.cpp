/**
 * Test that some Kokkos features work as intended when used like in Dyablo
 **/

#include "gtest/gtest.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_UnorderedMap.hpp"

struct OctCoordinates{
    int i,j,k;
    int level;
};

void test_UnorderedMap_morton()
{
    int level = 4;
    int nx, ny, nz;
    nx = ny = nz = (1 << level);

    Kokkos::UnorderedMap< OctCoordinates, int > oct_map(nx*ny*nz);

    int inserted = 0;
    Kokkos::parallel_reduce( "Fill_UnorderedMap", nx*ny*nz,
        KOKKOS_LAMBDA( int index, int& inserted )
    {
        OctCoordinates key;
        key.i = index%nx;
        key.j = (index/nx)%ny;
        key.k = index/(nx*ny);
        key.level = level;

        auto insert_res = oct_map.insert( key, index );
        if( insert_res.success() ) inserted++;
    }, inserted);

    EXPECT_EQ(inserted, nx*ny*nz);
    EXPECT_EQ( oct_map.size(), nx*ny*nz );

    int count = 0;
    int count_fail = 0;
    Kokkos::parallel_reduce( "Read_UnorderedMap", oct_map.capacity(),
        KOKKOS_LAMBDA( int map_index, int& count, int& count_fail )
    {
        if( oct_map.valid_at(map_index) )
        {
            count++;
            OctCoordinates key = oct_map.key_at(map_index);
            int index = oct_map.value_at(map_index);
            if( index != key.i + key.j*nx + key.k*nx*ny )
                count_fail++;
        }
    }, count, count_fail);

    EXPECT_EQ(count_fail, 0);
    EXPECT_EQ(count, nx*ny*nz);
}

TEST( Test_Kokkos_dyablo, UnorderedMap_morton )
{
    test_UnorderedMap_morton();
}

#include "utils/misc/HostDeviceSingleton.h"

namespace dyablo{

using VarIndex = int;

std::map< std::string, VarIndex > varindex_names{
    {"rho",2},
    {"e_tot",3},
    {"rho_u",4}
};

struct VarIndex_Set{
    VarIndex ID, IE, IU;
};

template<>
void HostDeviceSingleton<VarIndex_Set>::set()
{
 HostDeviceSingleton<VarIndex_Set>::set(VarIndex_Set{varindex_names.at("rho"), varindex_names.at("e_tot"), varindex_names.at("rho_u") });
}

void test_HostDeviceSingleton()
{
    HostDeviceSingleton<VarIndex_Set>::set(VarIndex_Set{varindex_names.at("rho"), varindex_names.at("e_tot"), varindex_names.at("rho_u") });

    EXPECT_EQ( 2, HostDeviceSingleton<VarIndex_Set>::get().ID );

    Kokkos::View<int> ID_val_device("ID_val_device");
    Kokkos::parallel_for( "test_HostDeviceSingleton", 1, 
        KOKKOS_LAMBDA(int)
    {
        ID_val_device() = HostDeviceSingleton<VarIndex_Set>::get().ID;
    });
    auto ID_val_host = Kokkos::create_mirror_view(ID_val_device);
    Kokkos::deep_copy(ID_val_host,ID_val_device);

    EXPECT_EQ( 2, ID_val_host() );
}

} // namespace dyablo

TEST( Test_Kokkos_dyablo, HostDeviceSingleton )
{
    dyablo::test_HostDeviceSingleton();
}