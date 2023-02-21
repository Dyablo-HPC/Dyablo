/**
 * \file test_CopyGhostBlockCellData_2D.cpp
 * \author Pierre Kestener, A. Durocher
 * Tests ghost cells copy for block based AMR from `CopyFaceBlockCellDataFunctor`
 * Initializes a mesh with `Implode` initial conditions and calls `CopyFaceBlockCellDataFunctor`
 * Ghost values in 3 octants are verified against the expected results from initial conditions
 * The 3 octants cover different cases (bigger, smaller, same size, boundary) (See declaration of variables iOct1, iOct2, iOct2)
 * \date September, 24th 2019
 */

#include <Kokkos_Core.hpp>

#include "real_type.h"    // choose between single and double precision
#include "FieldManager.h"


#include "legacy/CopyInnerBlockCellData.h"
#include "legacy/CopyGhostBlockCellData.h"
#include "legacy/ConvertToPrimitivesHydroFunctor.h"
#include "init/InitialConditions_analytical.h"
#include "foreach_cell/AMRBlockForeachCell.h"

using Device = Kokkos::DefaultExecutionSpace;

#include "gtest/gtest.h"

namespace dyablo
{

using CellArray = AMRBlockForeachCell_CellArray_impl::CellArray_global_ghosted;

enum NEIGH_SIZE : uint8_t
{
  NEIGH_IS_SMALLER   = 0,
  NEIGH_IS_LARGER    = 1,
  NEIGH_IS_SAME_SIZE = 2
};

struct AnalyticalFormula_implode_norefine : public AnalyticalFormula_base
{
  int ndim;
  real_t xmin, xmax;
  real_t ymin, ymax;
  real_t zmin, zmax;
  real_t gamma0;
  real_t rho_out, p_out, u_out, v_out, w_out;
  real_t rho_in, p_in, u_in, v_in, w_in;
  int shape;
  bool debug;

  AnalyticalFormula_implode_norefine(ConfigMap& configMap)
   : 
    ndim( configMap.getValue<int>("mesh", "ndim", 3) ),
    xmin( configMap.getValue<real_t>("mesh", "xmin", 0.0) ), xmax( configMap.getValue<real_t>("mesh", "xmax", 1.0) ),
    ymin( configMap.getValue<real_t>("mesh", "ymin", 0.0) ), ymax( configMap.getValue<real_t>("mesh", "ymax", 1.0) ),
    zmin( configMap.getValue<real_t>("mesh", "zmin", 0.0) ), zmax( configMap.getValue<real_t>("mesh", "zmax", 1.0) ),
    gamma0 ( configMap.getValue<real_t>("hydro","gamma0", 1.4) )
  {
    rho_out  = configMap.getValue<real_t>("implode","density_outer", 1.0);
    p_out  = configMap.getValue<real_t>("implode","pressure_outer", 1.0);
    u_out  = configMap.getValue<real_t>("implode","vx_outer", 0.0);
    v_out  = configMap.getValue<real_t>("implode","vy_outer", 0.0);
    w_out  = configMap.getValue<real_t>("implode","vz_outer", 0.0);

    rho_in  = configMap.getValue<real_t>("implode","density_inner", 0.125);
    p_in  = configMap.getValue<real_t>("implode","pressure_inner", 0.14);
    u_in  = configMap.getValue<real_t>("implode","vx_inner", 0.0);
    v_in  = configMap.getValue<real_t>("implode","vy_inner", 0.0);
    w_in  = configMap.getValue<real_t>("implode","vz_inner", 0.0);

    shape = configMap.getValue<int>("implode", "shape_region",0);
    
    debug = configMap.getValue<bool>("implode", "debug", false);
  }

  KOKKOS_INLINE_FUNCTION
  bool need_refine( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
  {
      return false;
  }

  KOKKOS_INLINE_FUNCTION
  ConsHydroState value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
  {
    ConsHydroState res;

    // initialize
    bool tmp;
    if (shape == 1) {
      if ( ndim==2 )
        tmp = x+y*y > 0.5 and x+y*y < 1.5;
      else
        tmp = x+y+z > 0.5 and x+y+z < 2.5;
    } else {
      if ( ndim==2 )
        tmp = x+y > (xmin+xmax)/2. + ymin;
      else
        tmp = x+y+z > (xmin+xmax)/2. + ymin + zmin;
    }

    if (tmp) {
      res.rho   = rho_out;
      res.e_tot = p_out/(gamma0-1.0) +
        0.5 * rho_out * (u_out*u_out + v_out*v_out);
      res.rho_u = u_out;
      res.rho_v = v_out;
    } else {
      res.rho   = rho_in;
      res.e_tot = p_in/(gamma0-1.0) +
        0.5 * rho_in * (u_in*u_in + v_in*v_in);
      res.rho_u = u_in;
      res.rho_v = v_in;
    }

    if ( ndim==3 ) {
      if (tmp) {
        res.rho_w = w_out;
        res.e_tot = p_out/(gamma0-1.0) +
          0.5 * rho_out * (u_out*u_out + v_out*v_out + w_out*w_out);
      } else {
        res.rho_w = w_in;
        res.e_tot = p_in/(gamma0-1.0) +
          0.5 * rho_in * (u_in*u_in + v_in*v_in + w_in*w_in);
      }
    }

    if (debug) {
      // just add a gradient to density
      real_t delta =  ndim==2  ?
        (x+y)*0.05 :
        (x+y+z)*0.05;
      res.rho += delta;
      res.e_tot += 2*delta;    
    }

    return res;
  }
};

using Init_implode = InitialConditions_analytical<AnalyticalFormula_implode_norefine>;

template<int ndim>
std::shared_ptr<AMRmesh> create_mesh();

template<>
std::shared_ptr<AMRmesh> create_mesh<2>()
{
  int ndim = 2;
  int level_min = 4; // Needed for initialization with pre-existing mesh
  int level_max = 5;
  std::shared_ptr<AMRmesh> amr_mesh = std::make_shared<AMRmesh>(ndim,ndim,std::array<bool,3>{false,false,false},level_min,level_max);

  EXPECT_GT( amr_mesh->getNumOctants(), 28 ) << "Internal test error : too few local octants";
  amr_mesh->setMarker(15,1);
  amr_mesh->setMarker(16,1);
  amr_mesh->setMarker(17,1);
  amr_mesh->setMarker(18,1);
  amr_mesh->setMarker(20,1);
  amr_mesh->setMarker(22,1);
  amr_mesh->setMarker(24,1);
  amr_mesh->setMarker(25,1);
  amr_mesh->setMarker(28,1);

  amr_mesh->adapt();
  amr_mesh->updateConnectivity();
  amr_mesh->loadBalance();

  return amr_mesh;
}

template<>
std::shared_ptr<AMRmesh> create_mesh<3>()
{
  int ndim = 3;
  int level_min = 3;
  int level_max = 4;
  std::shared_ptr<AMRmesh> amr_mesh = std::make_shared<AMRmesh>(ndim, ndim, std::array<bool,3>{false,false,false}, level_min, level_max);

  EXPECT_GT( amr_mesh->getNumOctants(), 121 ) << "Internal test error : too few local octants";
  // Refine initial 47 (final smaller 47..54)
  amr_mesh->setMarker(47,1);
  // Refine around initial 78
  amr_mesh->setMarker(65 ,1);
  amr_mesh->setMarker(72 ,1);
  amr_mesh->setMarker(73 ,1);
  amr_mesh->setMarker(67 ,1);
  amr_mesh->setMarker(74 ,1);
  amr_mesh->setMarker(75 ,1);
  amr_mesh->setMarker(76 ,1);
  amr_mesh->setMarker(77 ,1);
  amr_mesh->setMarker(71 ,1);
  amr_mesh->setMarker(69 ,1);
  //78
  amr_mesh->setMarker(81 ,1);
  amr_mesh->setMarker(88 ,1);
  amr_mesh->setMarker(89 ,1);
  amr_mesh->setMarker(79 ,1);
  amr_mesh->setMarker(85 ,1);
  amr_mesh->setMarker(92 ,1);
  amr_mesh->setMarker(93 ,1);
  amr_mesh->setMarker(97 ,1);
  amr_mesh->setMarker(104 ,1);
  amr_mesh->setMarker(105 ,1);
  amr_mesh->setMarker(99 ,1);
  amr_mesh->setMarker(106 ,1);
  amr_mesh->setMarker(107 ,1);
  amr_mesh->setMarker(113 ,1);
  amr_mesh->setMarker(120 ,1);
  amr_mesh->setMarker(121 ,1);

  amr_mesh->adapt();
  amr_mesh->updateConnectivity();
  amr_mesh->loadBalance();

  return amr_mesh;
}

template< typename CheckCell >
void check_2d( const CheckCell& check_cell, int iOct1, int iOct2, int iOct3 )
{
  NEIGH_SIZE LARGER = NEIGH_SIZE::NEIGH_IS_LARGER;
  NEIGH_SIZE SMALLER = NEIGH_SIZE::NEIGH_IS_SMALLER;
  uint32_t iOct = iOct1; // chose an octant which should have a "same size" neighbor in all direction
  uint32_t blocksize = 4;
  uint32_t ghostWidth = 2;
  for( uint32_t i1=0; i1<blocksize; i1++ )
    for( uint32_t ig=0; ig<ghostWidth; ig++ )
    {
      // Check Bottom border (print top)
      check_cell(iOct,ghostWidth+i1,ig,0);
      // Check Top border (print bottom)
      check_cell(iOct,ghostWidth+i1,ig+blocksize+ghostWidth,0);
      // Check Left border
      check_cell(iOct,ig,ghostWidth+i1,0);
      // Check Right
      check_cell(iOct,ig+blocksize+ghostWidth,ghostWidth+i1,0);
    }

  for( uint32_t ig1=0; ig1<ghostWidth; ig1++ )
    for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
    {
      //Check Bottom Left
      check_cell(iOct,ig1,ig2,0);
      //Check Bottom Right
      check_cell(iOct,ig1+blocksize+ghostWidth,ig2,0);
      //Check Upper Left
      check_cell(iOct,ig1,ig2+blocksize+ghostWidth,0);
      //Check Upper Right
      check_cell(iOct,ig1+blocksize+ghostWidth,ig2+blocksize+ghostWidth,0);
    }

  // chose an octant which should have at least
  // a "larger size" neighbor in one direction
  iOct = iOct2;
  for( uint32_t i1=0; i1<blocksize; i1++ )
    for( uint32_t ig=0; ig<ghostWidth; ig++ )
    {
      // Check Bottom border (print top)
      check_cell(iOct,ghostWidth+i1,ig,0,LARGER);
      // Check Top border (print bottom)
      check_cell(iOct,ghostWidth+i1,ig+blocksize+ghostWidth,0);
      // Check Left border
      check_cell(iOct,ig,ghostWidth+i1,0,LARGER);
      // Check Right
      check_cell(iOct,ig+blocksize+ghostWidth,ghostWidth+i1,0);
    }

  for( uint32_t ig1=0; ig1<ghostWidth; ig1++ )
    for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
    {
      //Check Bottom Left
      check_cell(iOct,ig1,ig2,0,LARGER);
      //Check Bottom Right
      check_cell(iOct,ig1+blocksize+ghostWidth,ig2,0,LARGER);
      //Check Upper Left
      check_cell(iOct,ig1,ig2+blocksize+ghostWidth,0,LARGER);
      //Check Upper Right
      check_cell(iOct,ig1+blocksize+ghostWidth,ig2+blocksize+ghostWidth,0);
    }

  // chose an octant which should have at least
  // an interface with "smaller size" neighbor in one direction and a periodic boundary
  iOct = iOct3; // chose an octant which should have a "same size" neighbor in all direction
  for( uint32_t i1=0; i1<blocksize; i1++ )
    for( uint32_t ig=0; ig<ghostWidth; ig++ )
    {
      // Check Bottom border (print top)
      check_cell(iOct,ghostWidth+i1,ig,0, SMALLER);
      // Check Top border (print bottom)
      check_cell(iOct,ghostWidth+i1,ig+blocksize+ghostWidth,0, SMALLER);
      // Check Left border
      check_cell(iOct,ig,ghostWidth+i1,0, SMALLER);
      // Check Right
      check_cell(iOct,ig+blocksize+ghostWidth,ghostWidth+i1,0, SMALLER);
    }

  for( uint32_t ig1=0; ig1<ghostWidth; ig1++ )
    for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
    {
      //Check Bottom Left
      check_cell(iOct,ig1,ig2,0, SMALLER);
      //Check Bottom Right
      check_cell(iOct,ig1+blocksize+ghostWidth,ig2,0, SMALLER);
      //Check Upper Left
      check_cell(iOct,ig1,ig2+blocksize+ghostWidth,0, SMALLER);
      //Check Upper Right
      check_cell(iOct,ig1+blocksize+ghostWidth,ig2+blocksize+ghostWidth,0, SMALLER);
    }

  // //Check (full) right border cells
  // for( uint32_t i1=0; i1<blocksize+2*ghostWidth; i1++ )
  //   for( uint32_t ig=0; ig<ghostWidth; ig++ )
  // {
  //   BOOST_CHECK_CLOSE(UgroupHost(ghostWidth+blocksize+ig + (blocksize+2*ghostWidth)*i1, fm[IP], iOct), 
  //                     UgroupHost(ghostWidth+blocksize-1  + (blocksize+2*ghostWidth)*i1, fm[IP], iOct), 0.001);
}

template< typename CheckCell >
void check_3d( const CheckCell& check_cell, int iOct1, int iOct2, int iOct3 )
{
  NEIGH_SIZE LARGER = NEIGH_SIZE::NEIGH_IS_LARGER;
  NEIGH_SIZE SMALLER = NEIGH_SIZE::NEIGH_IS_SMALLER;
  uint32_t blocksize = 4;
  uint32_t bx = blocksize;
  uint32_t by = blocksize;
  uint32_t bz = blocksize;
  uint32_t ghostWidth = 2;

  { // Test values for iOct1
      uint32_t iOct = iOct1;

      //Check faces
      for( uint32_t ig=0; ig<ghostWidth; ig++)
        for( uint32_t i1=ghostWidth; i1<ghostWidth+bx; i1++ )
          for( uint32_t i2=ghostWidth; i2<ghostWidth+bx; i2++ )
      {
        check_cell(iOct,ig,i1,i2);                 //Left
        check_cell(iOct,bx+ghostWidth+ig,i1,i2);   //Right
        check_cell(iOct,i1,ig,i2);                 // Bottom (print up)
        check_cell(iOct,i1,ig+ghostWidth+by,i2);   // Top (print down)
        check_cell(iOct,i1,i2,ig);                 // front 
        check_cell(iOct,i1,i2,ig+ghostWidth+bz);   // back
      }
      //Check edges
      for( uint32_t ig1=0; ig1<ghostWidth; ig1++)
        for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
          for( uint32_t i2=ghostWidth; i2<ghostWidth+bx; i2++ )
      {
        //Edges in X direction
        check_cell( iOct, i2, ig1, ig2 );
        check_cell( iOct, i2, ig1+bx+ghostWidth, ig2 );
        check_cell( iOct, i2, ig1, ig2+bx+ghostWidth );
        check_cell( iOct, i2, ig1+bx+ghostWidth, ig2+bx+ghostWidth);
        //Edges in Y direction
        check_cell( iOct, ig1, i2, ig2 );
        check_cell( iOct, ig1+bx+ghostWidth, i2, ig2 );
        check_cell( iOct, ig1, i2, ig2+bx+ghostWidth );
        check_cell( iOct, ig1+bx+ghostWidth, i2, ig2+bx+ghostWidth);
        //Edges in Z direction
        check_cell( iOct, ig1, ig2, i2 );
        check_cell( iOct, ig1+bx+ghostWidth, ig2, i2 );
        check_cell( iOct, ig1, ig2+bx+ghostWidth, i2 );
        check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, i2 );
      }

      //Check corners
      for( uint32_t ig1=0; ig1<ghostWidth; ig1++)
        for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
          for( uint32_t ig3=0; ig3<ghostWidth; ig3++ )
      {
          check_cell( iOct, ig1, ig2, ig3 );
          check_cell( iOct, ig1+bx+ghostWidth, ig2, ig3 );
          check_cell( iOct, ig1, ig2+bx+ghostWidth, ig3 );
          check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, ig3);
          check_cell( iOct, ig1, ig2, ig3+bx+ghostWidth );
          check_cell( iOct, ig1+bx+ghostWidth, ig2, ig3+bx+ghostWidth );
          check_cell( iOct, ig1, ig2+bx+ghostWidth, ig3+bx+ghostWidth );
          check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, ig3+bx+ghostWidth );
      }

    }

    { // Test values for iOct2
      uint32_t iOct = iOct2;

      //Check faces
      for( uint32_t ig=0; ig<ghostWidth; ig++)
        for( uint32_t i1=ghostWidth; i1<ghostWidth+bx; i1++ )
          for( uint32_t i2=ghostWidth; i2<ghostWidth+bx; i2++ )
      {
        check_cell(iOct,ig,i1,i2,LARGER);                 //Left
        check_cell(iOct,bx+ghostWidth+ig,i1,i2);          //Right
        check_cell(iOct,i1,ig,i2,LARGER);                 // Bottom (print up)
        check_cell(iOct,i1,ig+ghostWidth+by,i2);          // Top (print down)
        check_cell(iOct,i1,i2,ig,LARGER);                 // front (print last)
        check_cell(iOct,i1,i2,ig+ghostWidth+bz);          // back (print first)
      }
      //Check edges
      for( uint32_t ig1=0; ig1<ghostWidth; ig1++)
        for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
          for( uint32_t i2=ghostWidth; i2<ghostWidth+bx; i2++ )
      {
        //Edges in X direction
        check_cell( iOct, i2, ig1, ig2, LARGER );
        check_cell( iOct, i2, ig1+bx+ghostWidth, ig2,LARGER );
        check_cell( iOct, i2, ig1, ig2+bx+ghostWidth, LARGER);
        check_cell( iOct, i2, ig1+bx+ghostWidth, ig2+bx+ghostWidth );
        //Edges in Y direction
        check_cell( iOct, ig1, i2, ig2, LARGER );
        check_cell( iOct, ig1+bx+ghostWidth, i2, ig2,LARGER );
        check_cell( iOct, ig1, i2, ig2+bx+ghostWidth, LARGER );
        check_cell( iOct, ig1+bx+ghostWidth, i2, ig2+bx+ghostWidth);
        //Edges in Z direction
        check_cell( iOct, ig1, ig2, i2, LARGER );
        check_cell( iOct, ig1+bx+ghostWidth, ig2, i2,LARGER );
        check_cell( iOct, ig1, ig2+bx+ghostWidth, i2,LARGER );
        check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, i2);
      }

      //Check corners
      for( uint32_t ig1=0; ig1<ghostWidth; ig1++)
        for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
          for( uint32_t ig3=0; ig3<ghostWidth; ig3++ )
      {
          check_cell( iOct, ig1, ig2, ig3, LARGER );
          check_cell( iOct, ig1+bx+ghostWidth, ig2, ig3,LARGER );
          check_cell( iOct, ig1, ig2+bx+ghostWidth, ig3,LARGER );
          check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, ig3, LARGER );
          check_cell( iOct, ig1, ig2, ig3+bx+ghostWidth,LARGER );
          check_cell( iOct, ig1+bx+ghostWidth, ig2, ig3+bx+ghostWidth, LARGER );
          check_cell( iOct, ig1, ig2+bx+ghostWidth, ig3+bx+ghostWidth, LARGER );
          check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, ig3+bx+ghostWidth );
      }

    }

    { // Test values for iOct3
      uint32_t iOct = iOct3;

      //Check faces
      for( uint32_t ig=0; ig<ghostWidth; ig++)
        for( uint32_t i1=ghostWidth; i1<ghostWidth+bx; i1++ )
          for( uint32_t i2=ghostWidth; i2<ghostWidth+bx; i2++ )
      {
        check_cell(iOct,ig,i1,i2,SMALLER);                 //Left
        check_cell(iOct,bx+ghostWidth+ig,i1,i2,SMALLER);   //Right
        check_cell(iOct,i1,ig,i2,SMALLER);                 // Bottom (print up)
        check_cell(iOct,i1,ig+ghostWidth+by,i2,SMALLER);   // Top (print down)
        check_cell(iOct,i1,i2,ig,SMALLER);                 // front 
        check_cell(iOct,i1,i2,ig+ghostWidth+bz,SMALLER);   // back
      }
      //Check edges
      for( uint32_t ig1=0; ig1<ghostWidth; ig1++)
        for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
          for( uint32_t i2=ghostWidth; i2<ghostWidth+bx; i2++ )
      {
        //Edges in X direction
        check_cell( iOct, i2, ig1, ig2 ,SMALLER);
        check_cell( iOct, i2, ig1+bx+ghostWidth, ig2 ,SMALLER);
        check_cell( iOct, i2, ig1, ig2+bx+ghostWidth ,SMALLER);
        check_cell( iOct, i2, ig1+bx+ghostWidth, ig2+bx+ghostWidth ,SMALLER);
        //Edges in Y direction
        check_cell( iOct, ig1, i2, ig2 ,SMALLER);
        check_cell( iOct, ig1+bx+ghostWidth, i2, ig2 ,SMALLER);
        check_cell( iOct, ig1, i2, ig2+bx+ghostWidth ,SMALLER);
        check_cell( iOct, ig1+bx+ghostWidth, i2, ig2+bx+ghostWidth ,SMALLER);
        //Edges in Z direction
        check_cell( iOct, ig1, ig2, i2, SMALLER );
        check_cell( iOct, ig1+bx+ghostWidth, ig2, i2 ,SMALLER);
        check_cell( iOct, ig1, ig2+bx+ghostWidth, i2 ,SMALLER);
        check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, i2,SMALLER );
      }

      //Check corners
      for( uint32_t ig1=0; ig1<ghostWidth; ig1++)
        for( uint32_t ig2=0; ig2<ghostWidth; ig2++ )
          for( uint32_t ig3=0; ig3<ghostWidth; ig3++ )
      {
          check_cell( iOct, ig1, ig2, ig3,SMALLER );
          check_cell( iOct, ig1+bx+ghostWidth, ig2, ig3,SMALLER );
          check_cell( iOct, ig1, ig2+bx+ghostWidth, ig3,SMALLER );
          check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, ig3,SMALLER );
          check_cell( iOct, ig1, ig2, ig3+bx+ghostWidth,SMALLER );
          check_cell( iOct, ig1+bx+ghostWidth, ig2, ig3+bx+ghostWidth,SMALLER );
          check_cell( iOct, ig1, ig2+bx+ghostWidth, ig3+bx+ghostWidth,SMALLER );
          check_cell( iOct, ig1+bx+ghostWidth, ig2+bx+ghostWidth, ig3+bx+ghostWidth,SMALLER );
      }

    }
}

// =======================================================================
// =======================================================================
template<int ndim>
void run_test()
{
  /*
   * testing CopyGhostBlockCellDataFunctor
   */
  std::cout << "// =========================================\n";
  std::cout << "// Testing CopyGhostBlockCellDataFunctor " << ndim << "D ... "<< std::endl;
  std::cout << "// =========================================\n";

  /*
   * read parameter file and initialize a ConfigMap object
   */
  // only MPI rank 0 actually reads input file
  std::string input_file = (ndim == 2) ? "./block_data/test_blast_2D_block.ini"
                                        :"./block_data/test_blast_3D_block.ini";
  ConfigMap configMap = ConfigMap::broadcast_parameters(input_file);

  int ndim_ini = configMap.getValue<int>("mesh", "ndim", ndim);
  assert( ndim_ini == ndim );
  GravityType gravity_type = configMap.getValue<GravityType>("gravity", "gravity_type", GRAVITY_NONE);

  // Setup Fieldmanager
  FieldManager fieldMgr;
  {
    // always enable rho, energy and velocity components
    std::set< VarIndex > enabled_vars( {ID, IP, IE, IU, IV} );
    
    bool three_d = ndim == 3 ? 1 : 0;

    if( three_d ) enabled_vars.insert( IW );

    fieldMgr = FieldManager(enabled_vars);
  }
  
  auto fm = fieldMgr.get_id2index();

  /*
   * "geometry" setup
   */

  // block ghost width
  uint32_t ghostWidth = configMap.getValue<uint32_t>("amr", "ghostwidth", 2);

  BoundaryConditionType boundary_xmin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmin", BC_ABSORBING);
  BoundaryConditionType boundary_xmax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_xmax", BC_ABSORBING);
  BoundaryConditionType boundary_ymin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymin", BC_ABSORBING);
  BoundaryConditionType boundary_ymax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_ymax", BC_ABSORBING);
  BoundaryConditionType boundary_zmin  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmin", BC_ABSORBING);
  BoundaryConditionType boundary_zmax  = configMap.getValue<BoundaryConditionType>("mesh","boundary_type_zmax", BC_ABSORBING);

  // block sizes
  uint32_t bx = configMap.getValue<uint32_t>("amr", "bx", 4);
  uint32_t by = configMap.getValue<uint32_t>("amr", "by", 4);
  uint32_t bz = configMap.getValue<uint32_t>("amr", "bz", ndim==2 ? 1 : 4);

  // block sizes with ghosts
  uint32_t bx_g = bx + 2 * ghostWidth;
  uint32_t by_g = by + 2 * ghostWidth;
  uint32_t bz_g = ndim==2 ? 1 : bz + 2 * ghostWidth;;

  blockSize_t blockSizes, blockSizes_g;
  blockSizes[IX] = bx;
  blockSizes[IY] = by;
  blockSizes[IZ] = bz;

  blockSizes_g[IX] = bx_g;
  blockSizes_g[IY] = by_g;
  blockSizes_g[IZ] = bz_g;

  std::cout << "Using " 
            << "bx=" << bx << " "
            << "by=" << by << " "
            << "bz=" << bz << " "
            << "bx_g=" << bx_g << " "
            << "by_g=" << by_g << " "
            << "bz_g=" << bz_g << " "
            << "ghostwidth=" << ghostWidth << "\n";

  // create solver members
  //std::unique_ptr<SolverHydroMusclBlock> solver = std::make_unique<SolverHydroMusclBlock>(params, configMap);
  
  std::cout << "Create mesh..." << std::endl;
  std::shared_ptr<AMRmesh> amr_mesh = create_mesh<ndim>(  ); //solver->amr_mesh 

  uint32_t nbOctsPerGroup, iGroup, iOct1, iOct2, iOct3;
  // Choose representing octants according to dimension
  if( ndim == 2 )
  {
    nbOctsPerGroup = (ndim==2) ? 64 : 256;
    iGroup = 0;
    // Octant that have a "same size" neighbor in all direction
    iOct1 = 3;
    // Octant that have at least
    // a "larger size" neighbor in one direction
    iOct2 = 15;
    // Octant that have have at least
    // an interface with "smaller size" neighbor in one direction and a periodic boundary
    iOct3 = 31;
  }
  else
  {
    nbOctsPerGroup = 256;
    iGroup = 0;
    // Octant that have a "same size" neighbor in all direction
    iOct1 = 7;
    // Octant that have at least
    // a "larger size" neighbor in one direction
    iOct2 = 47;
    // Octant that have have at least
    // an interface with "smaller size" neighbor in one direction and a periodic boundary
    // (11 cells have been refined before index 78)
    iOct3 = 11*7+78;
  }

  std::cout << "Apply initial condition..." << std::endl;

  int nbfields = fieldMgr.nbfields();
  int nbOcts = amr_mesh->getNumOctants();

  
  Timers timers;
  ForeachCell foreach_cell(*amr_mesh, configMap);
  UserData U_(configMap, foreach_cell);

  Init_implode init_implode(configMap, foreach_cell, timers);
  init_implode.init( U_ ); 
  AnalyticalFormula_implode_norefine init_implode_formula( configMap );
  U_.exchange_ghosts( GhostCommunicator_kokkos( amr_mesh ) );

  LegacyDataArray U( U_ );

  // by now, init condition must have been called

  uint32_t nbCellsPerOct_g = bx_g * by_g * bz_g;

  /*
   * allocate/initialize Ugroup
   */

  std::cout << "Currently mesh has " << nbOcts << " octants\n";

  std::cout << "Using nbCellsPerOct_g (number of cells per octant with ghots) = " << nbCellsPerOct_g << "\n";
  std::cout << "Using nbOctsPerGroup (number of octant per group) = " << nbOctsPerGroup << "\n";
  

  DataArrayBlock Ugroup("Ugroup", nbCellsPerOct_g, nbfields, nbOctsPerGroup);
  DataArrayBlockHost UgroupHost("UgroupHost", nbCellsPerOct_g, nbfields, nbOctsPerGroup);

  std::cout << "Ugroup sizes = " 
              << Ugroup.extent(0) << " "
              << Ugroup.extent(1) << " "
              << Ugroup.extent(2) << "\n";

  // save solution, just for cross-checking
  //solver->save_solution();

  // Define print functions as lambdas
  // Print info about original local (iGroup set in main()) octant
  auto show_octant = [&](uint32_t iOct_local){  
    //#define DEBUG_PRINT
    #ifdef DEBUG_PRINT
    uint32_t iOct_global = iOct_local + iGroup * nbOctsPerGroup;

    std::cout << "Looking at octant id = " << iOct_global << "\n";
    // octant location
    double x = amr_mesh->getCoordinates(iOct_global)[IX];
    double y = amr_mesh->getCoordinates(iOct_global)[IY];
    std::cout << "Octant location : x=" << x << " y=" << y << "\n";
    auto print_neighbor_status = [&]( int codim, int iface)
    { 
      std::vector<uint32_t> iOct_neighbors;
      std::vector<bool> isghost_neighbors;
      amr_mesh->findNeighbours(iOct_global, iface, codim, iOct_neighbors, isghost_neighbors);
      if (iOct_neighbors.size() == 0)
      {
        std::cout << "( no neigh)";
      }
      else if(iOct_neighbors.size() == 1)
      {
        uint32_t neigh_level = isghost_neighbors[0] ? 
                    amr_mesh->getLevelGhost(iOct_neighbors[0]) : 
                    amr_mesh->getLevel(iOct_neighbors[0]);
        if ( amr_mesh->getLevel(iOct_global) > neigh_level )
          std::cout << "( bigger  )";
        else if ( amr_mesh->getLevel(iOct_global) < neigh_level )
          std::cout << "( smaller )";
        else
          std::cout << "(same size)";
      }
      else
      {
        std::cout << "( smaller )";
      }    
    };
    auto print_neighbor = [&]( int codim, int iface)
    { 
      std::vector<uint32_t> iOct_neighbors;
      std::vector<bool> isghost_neighbors;
      amr_mesh->findNeighbours(iOct_global, iface, codim, iOct_neighbors, isghost_neighbors);
      if (iOct_neighbors.size() == 0)
      {
        std::cout << "(no neigh)";
      }
      else
      {
        std::cout << std::setw(10) << iOct_neighbors[0];
      }
    };
    std::cout << "Octant neighbors :" << std::endl;
    print_neighbor_status(2,0) ; print_neighbor_status(1,2) ; print_neighbor_status(2,1); std::cout << std::endl;
    print_neighbor_status(1,0) ; std::cout << "(         )"; print_neighbor_status(1,1); std::cout << std::endl;
    print_neighbor_status(2,2) ; print_neighbor_status(1,3) ; print_neighbor_status(2,3); std::cout << std::endl;
    std::cout << "Octant neighbors :" << std::endl;
    print_neighbor(2,0) ; print_neighbor(1,2) ; print_neighbor(2,1); std::cout << std::endl;
    print_neighbor(1,0) ; std::cout << "          "; print_neighbor(1,1); std::cout << std::endl;
    print_neighbor(2,2) ; print_neighbor(1,3) ; print_neighbor(2,3); std::cout << std::endl;
    

    // std::cout << "  FACE_XMIN   "; print_neighbor_status(FACE_XMIN);
    // std::cout << "  FACE_XMAX  "; print_neighbor_status(FACE_XMAX);
    // std::cout << "  FACE_TOP    "; print_neighbor_status(FACE_TOP);
    // std::cout << "  FACE_BOTTOM "; print_neighbor_status(FACE_BOTTOM);

    std::cout << "Printing Uhost data from iOct = " << iOct_global << "\n";
    for (uint32_t iz=0; iz<bz; ++iz) 
    {
      for (uint32_t iy=0; iy<by; ++iy)
      {
        for (uint32_t ix=0; ix<bx; ++ix)
        {
          uint32_t index = ix + bx*(iy+by*iz);
          printf("%5f ",Uhost(index,fm[ID],iOct_global));
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
    #endif
  };

  // Print info about final local octant (with ghosts)
  auto show_octant_group = [&](uint32_t iOct_local){  
    uint32_t iOct_global = iOct_local + iGroup * nbOctsPerGroup;
    // print data from the chosen iGroup 
    std::cout << "Printing Ugroup data from iOct = " << iOct_global << " | iOctLocal = " << iOct_local << " and iGroup = " << iGroup << "\n";       
    for (uint32_t iz=0; iz < bz_g; ++iz) 
    {
      for (uint32_t iy = 0; iy < by_g; ++iy)
      {
        for (uint32_t ix = 0; ix < bx_g; ++ix)
        {
          uint32_t index = ix + bx_g*(iy+by_g*iz);
          printf("%5f ", UgroupHost(index, fm[IP], iOct_local));
        }
        std::cout << "\n";
      } 
      std::cout << "\n";
    }
  };

  show_octant(iOct1);
  show_octant(iOct2);
  show_octant(iOct3);  

  // first copy inner cells
  CopyInnerBlockCellDataFunctor::apply({ndim, gravity_type},
                                       fm, 
                                       blockSizes,
                                       ghostWidth, 
                                       nbOcts,
                                       nbOctsPerGroup,
                                       U, Ugroup, iGroup);

  std::cout << "==========================================";
  std::cout << "Testing CopyGhostBlockCellDataFunctor....\n";
  {
    InterfaceFlags interface_flags(nbOctsPerGroup); //solver->interface_flags
    CopyGhostBlockCellDataFunctor::apply(amr_mesh->getLightOctree(),
                                        {
                                          boundary_xmin, boundary_xmax,
                                          boundary_ymin, boundary_ymax,
                                          boundary_zmin, boundary_zmax,
                                          gravity_type
                                        }, 
                                        fm,
                                        blockSizes,
                                        ghostWidth,
                                        nbOctsPerGroup,
                                        U,  
                                        Ugroup, 
                                        iGroup,
                                        interface_flags);

    Kokkos::deep_copy(UgroupHost, Ugroup);
    
    
    show_octant_group(iOct1);
    show_octant_group(iOct2);
    show_octant_group(iOct3);  

    // CopyFaceBlockCellDataFunctor::apply(solver->amr_mesh,
    //                                     configMap,
    //                                     params, 
    //                                     fm,
    //                                     blockSizes,
    //                                     ghostWidth,
    //                                     nbOctsPerGroup,
    //                                     solver->U, 
    //                                     solver->Ughost, 
    //                                     Ugroup, 
    //                                     iGroup,
    //                                     Interface_flags);
    // CopyCornerBlockCellDataFunctor::apply(solver->amr_mesh,
    //                                     configMap,
    //                                     params, 
    //                                     fm,
    //                                     blockSizes,
    //                                     ghostWidth,
    //                                     nbOctsPerGroup,
    //                                     solver->U, 
    //                                     solver->Ughost, 
    //                                     Ugroup, 
    //                                     iGroup,
    //                                     Interface_flags);
    // show_octant_group(iOct1);
    
    NEIGH_SIZE LARGER = NEIGH_SIZE::NEIGH_IS_LARGER;
    NEIGH_SIZE SMALLER = NEIGH_SIZE::NEIGH_IS_SMALLER;
    // Fetch values from initial conditions and compare to actual value 
    auto check_cell = [&](uint32_t iOct_local, uint32_t ix, uint32_t iy, uint32_t iz, NEIGH_SIZE neighbor_size = NEIGH_SIZE::NEIGH_IS_SAME_SIZE )
    { 
      uint32_t iOct_global = iOct_local + iGroup * nbOctsPerGroup;
      const real_t octSize = amr_mesh->getSize(iOct_global)[0];
      const real_t cellSize = octSize/bx;
      const real_t x0 = amr_mesh->getCoordinates(iOct_global)[IX];
      const real_t y0 = amr_mesh->getCoordinates(iOct_global)[IY];
      const real_t z0 = (ndim==2)?0:amr_mesh->getCoordinates(iOct_global)[IZ];
      real_t x = x0 + ix*cellSize - ghostWidth*cellSize + cellSize/2;
      real_t y = y0 + iy*cellSize - ghostWidth*cellSize + cellSize/2;
      real_t z = (ndim==2)?0:z0 + iz*cellSize - ghostWidth*cellSize + cellSize/2;

      real_t expected;
      if( neighbor_size == LARGER )
      {
        // Works because ghostWidth and blockSizes are pair
        uint32_t ix_larger = (ix/2)*2 + 1;
        uint32_t iy_larger = (iy/2)*2 + 1;
        uint32_t iz_larger = (ndim==2)?0:(iz/2)*2 + 1;
        real_t x_larger = x0 + ix_larger*cellSize - ghostWidth*cellSize;
        real_t y_larger = y0 + iy_larger*cellSize - ghostWidth*cellSize;
        real_t z_larger = (ndim==2)?0:z0 + iz_larger*cellSize - ghostWidth*cellSize;
        expected = init_implode_formula.value( x_larger,y_larger,z_larger,0,0,0 ).e_tot;
      }
      else if( neighbor_size == SMALLER )
      {
        expected = 0;        
        for(int dz=0; dz<(ndim-1); dz++)
          for(int dy=0; dy<2; dy++)
            for(int dx=0; dx<2; dx++)
            {
              real_t x_smaller = x - cellSize/4 + dx*(cellSize/2);
              real_t y_smaller = y - cellSize/4 + dy*(cellSize/2);
              real_t z_smaller = (ndim==2)?0:z - cellSize/4 + dz*(cellSize/2);
              expected += init_implode_formula.value( x_smaller ,y_smaller ,z_smaller,0,0,0).e_tot;
            }
        expected = expected/(2*2*(ndim-1)); 
      }
      else
      {
        expected = init_implode_formula.value( x,y,z,0,0,0  ).e_tot;
      }
      
      uint32_t index = ix + bx_g * iy + bx_g * by_g * iz;
      real_t& actual = UgroupHost(index, fm[IP], iOct_local);

      //EXPECT_NEAR(actual, expected, 0.0001);      
      EXPECT_DOUBLE_EQ(actual, expected);

      //// ---------------------- Debug prints ------------------------
      if( actual != expected )
      {
        std::cout << "Value at iOct : " << iOct_global << " pos ghosted : "
                    "(" << ix << "," << iy << "," << iz << ") = "
                    "(" << x << "," << y << "," << z << ") :" << std::endl;
        std::cout << "  Should be (implode) : " <<  expected << std::endl;
        std::cout << "  Is                  : " <<  actual << std::endl;
      }
      //actual = 9.1111; //To visualize which cells have been tested
      //actual = (actual==expected) ? 9.1111 : 9.2222; //To visualize which tested cells are wrong
      //actual = actual - expected; //To visualize difference to expected
      //// ---------------------- Debug prints ------------------------
    };

    if( GlobalMpiSession::get_comm_world().MPI_Comm_rank() == 0 )
    {
      if(ndim==2)
        check_2d(check_cell, iOct1, iOct2, iOct3);
      else
        check_3d(check_cell, iOct1, iOct2, iOct3);
    }

    // }    
      
  } // end testing CopyFaceBlockCellDataFunctor

} // run_test



} // namespace dyablo


TEST(dyablo, test_CopyGhostBlockCellData_2D)
{
  dyablo::run_test<2>();
}

TEST(dyablo, test_CopyGhostBlockCellData_3D)
{
  dyablo::run_test<3>();
}