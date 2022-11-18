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

namespace dyablo{

KOKKOS_INLINE_FUNCTION
const Kokkos::Array<double,3>& constexpr_var()
{
    static constexpr Kokkos::Array<double,3> var_ = {1.5, 2.5, 3.5};
    return var_;
} 

KOKKOS_INLINE_FUNCTION
void set_value( const Kokkos::View<double[3]>& v )
{
    v(0) = constexpr_var()[0];
    v(1) = constexpr_var()[1];
    v(2) = constexpr_var()[2];
}


void test_constexpr_device()
{
    Kokkos::View<double[3]> var_view_device("var_view");
    Kokkos::parallel_for( "test_constexpr_device", 1, 
        KOKKOS_LAMBDA(int)
    {
        set_value( var_view_device );
    });
    auto var_view_host = Kokkos::create_mirror_view(var_view_device);
    Kokkos::deep_copy(var_view_host,var_view_device);

    EXPECT_EQ( 1.5, var_view_host(0) );
    EXPECT_EQ( 2.5, var_view_host(1) );
    EXPECT_EQ( 3.5, var_view_host(2) );
}

} // namespace dyablo

TEST( Test_Kokkos_dyablo, test_constexpr_device )
{
    dyablo::test_constexpr_device();
}