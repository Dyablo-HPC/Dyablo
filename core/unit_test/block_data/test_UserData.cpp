#include "gtest/gtest.h"

#include "UserData.h"

using namespace dyablo;

namespace dyablo{

void run_test()
{
  std::string configmap_str = "";
  ConfigMap configMap(configmap_str);

  // Fill default values
  constexpr int ndim = 3;
  configMap.getValue<int>("run", "ndim", ndim);
  uint32_t bx = configMap.getValue<uint32_t>("amr", "bx", 8);
  uint32_t by = configMap.getValue<uint32_t>("amr", "by", 4);
  uint32_t bz = configMap.getValue<uint32_t>("amr", "bz", (ndim==2)?1:4);
  uint32_t level_min = configMap.getValue<uint32_t>("amr", "level_min", 3);
  uint32_t level_max = configMap.getValue<uint32_t>("amr", "level_max", 5);
  configMap.getValue<real_t>("mesh", "xmin", -2);
  configMap.getValue<real_t>("mesh", "ymin", 0);
  configMap.getValue<real_t>("mesh", "zmin", 1);
  configMap.getValue<real_t>("mesh", "xmax", 2);
  configMap.getValue<real_t>("mesh", "ymax", 2);
  configMap.getValue<real_t>("mesh", "zmax", 4);

  AMRmesh pmesh( ndim, ndim, std::array<bool,3>{false,false,false}, level_min, level_max);
  ForeachCell foreach_cell( pmesh, configMap );

  UserData U(configMap, foreach_cell);

  U.new_fields({"px","py","pz"});

  EXPECT_TRUE( U.has_field("px") );
  EXPECT_TRUE( U.has_field("py") );
  EXPECT_TRUE( U.has_field("pz") );
  EXPECT_FALSE( U.has_field("xx") );

  EXPECT_EQ( U.getShape().nbOcts, pmesh.getNumOctants() );
  EXPECT_EQ( U.getShape().bx, bx );
  EXPECT_EQ( U.getShape().by, by );
  EXPECT_EQ( U.getShape().bz, bz );

  // Init px, py, pz
  {
    VarIndex PX = (VarIndex)2, PY = (VarIndex)7, PZ = (VarIndex)3;

    auto Uinit = U.getAccessor( { {"px",PX,0}, {"py",PY,0}, {"pz",PZ,0} } );

    auto cells = foreach_cell.getCellMetaData();

    foreach_cell.foreach_cell( "Init p", U.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      auto c = cells.getCellCenter(iCell);
      Uinit.at( iCell, PX ) = c[IX];
      Uinit.at( iCell, PY ) = c[IY];
      Uinit.at( iCell, PZ ) = c[IZ];
    });
  }

  U.new_fields({"p2x","p2y","p2z"});

  // Initialize p2x, p2y, p2z from px, py, pz using same Varindex in different accessors
  {
    VarIndex PX = (VarIndex)1, PY = (VarIndex)3, PZ = (VarIndex)5;

    auto U1 = U.getAccessor( { {"px",PX,0}, {"py",PY,0}, {"pz",PZ,0} } );
    auto U2 = U.getAccessor( { {"p2x",PX,0}, {"p2y",PY,0}, {"p2z",PZ,0} } );

    foreach_cell.foreach_cell( "Copy p2", U.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      U2.at( iCell, PX ) = U1.at( iCell, PX ) + 1.0;
      U2.at( iCell, PY ) = U1.at( iCell, PY ) + 1.0;
      U2.at( iCell, PZ ) = U1.at( iCell, PZ ) + 1.0;
    });
  }

  // Verify that p2xyz has the right position
  {
    VarIndex PX = (VarIndex)1, PY = (VarIndex)3, PZ = (VarIndex)5;

    auto U2 = U.getAccessor( { {"p2x",PX,0}, {"p2y",PY,0}, {"p2z",PZ,0} } );

    auto cells = foreach_cell.getCellMetaData();

    int error_count = 0;
    foreach_cell.reduce_cell( "Check p2", U.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell, int& error_count )
    {
      auto c = cells.getCellCenter(iCell);
      if( U2.at( iCell, PX ) != c[IX] + 1.0 )
        error_count++;
      if( U2.at( iCell, PY ) != c[IY] + 1.0 )
        error_count++;
      if( U2.at( iCell, PZ ) != c[IZ] + 1.0 )
        error_count++;
    }, error_count);

    EXPECT_EQ( 0, error_count );
  }

  U.move_field("px", "p2x");
  U.move_field("py", "p2y");
  U.move_field("pz", "p2z");

  EXPECT_TRUE( U.has_field("px") );
  EXPECT_TRUE( U.has_field("py") );
  EXPECT_TRUE( U.has_field("pz") );
  EXPECT_FALSE( U.has_field("p2x") );
  EXPECT_FALSE( U.has_field("p2y") );
  EXPECT_FALSE( U.has_field("p2z") );

  // Verify that pxyz has been overwritten
  {
    auto px = U.getField("px");
    auto py = U.getField("py");
    auto pz = U.getField("pz");

    auto cells = foreach_cell.getCellMetaData();

    int error_count = 0;
    foreach_cell.reduce_cell( "Check p overwrite", px,
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell, int& error_count )
    {
      auto c = cells.getCellCenter(iCell);
      if( px.at_ivar( iCell, 0) != c[IX] + 1.0 )
        error_count++;
      if( py.at_ivar( iCell, 0) != c[IY] + 1.0 )
        error_count++;
      if( pz.at_ivar( iCell, 0) != c[IZ] + 1.0 )
        error_count++;
    }, error_count);

    EXPECT_EQ( 0, error_count );
  }
}

} // namespace dyablo

TEST(Test_UserData, works)
{
  dyablo::run_test();
}