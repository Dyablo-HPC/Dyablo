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