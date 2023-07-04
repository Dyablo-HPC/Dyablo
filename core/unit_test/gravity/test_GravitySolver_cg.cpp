/**
 * \author A. Durocher
 * Tests Conjuguate Gradient gravity solver
 * 
 * This file tests the self-gravity solver.
 * An AMR mesh is created and filled with a Herquist density profile 
 * The self-gravity solver is called to compute the gravity potential associated
 * to the density profile. Then the resulting gravity potential is compared* to 
 * the analytical Hernquist gravity potential. 
 * 
 * This test generates paraview output files. 
 * In test_GravitySolver_iter0000001.xmf
 *  - iphy : the computed gravity potential
 *  - igx : numerical gravity potential shifted to fit* the analytical potential
 *  - igy : the analytical Hernquist gravity potential ("periodic" with 2 repetitions)
 *  - igz : |igx-igy|/|igy| ()
 * 
 * Note : numerical values are only compared to analytical values at the center 
 * of the domain. This is because the analytical solution is not accurate on 
 * borders because of boundary conditions. 
 */

#include "gtest/gtest.h"
#include "amr/AMRmesh.h"
#include "mpi/GhostCommunicator.h"
#include "gravity/GravitySolver_cg.h"
#include "io/IOManager.h"

namespace dyablo {
namespace constants{
  constexpr double Pi = 3.14;
}

//https://arxiv.org/pdf/1712.07070.pdf , Appendix A
struct Hernquist{
  constexpr static real_t G = 1/(4*dyablo::constants::Pi);
  constexpr static real_t rho0 = 1;
  constexpr static real_t r0 = 0.01;
  constexpr static real_t L = 1;
  
  constexpr static real_t M = 2*dyablo::constants::Pi*r0*r0*r0*rho0;

  KOKKOS_INLINE_FUNCTION
  static real_t rho(real_t r2)
  {
    real_t r = std::sqrt(r2)*L;
    return rho0/( r/r0 * std::pow(1 + r/r0, 3) );
  }

  KOKKOS_INLINE_FUNCTION
  static real_t phi(real_t r2)
  {
    real_t r = std::sqrt(r2)*L;
    return -G*M / (r+r0);
  }

  KOKKOS_INLINE_FUNCTION
  static real_t phi_periodic( real_t x, real_t y, real_t z )
  {
    constexpr int dmax = 2;
    real_t res = 0;
    for( int ix = -dmax; ix <= +dmax; ix++ )
    for( int iy = -dmax; iy <= +dmax; iy++ )
    for( int iz = -dmax; iz <= +dmax; iz++ )
    {
      real_t r2 = (x+ix)*(x+ix) + (y+iy)*(y+iy) + (z+iz)*(z+iz);
      res += phi( r2 );
    }
    return res;    
  }

  // density trigger for refinement
  static real_t rhomax(int level)
  {
    int GAMER_level = level+2-6; // for 4³ blocks - level_min = 6
    return 1E-3 * std::pow( 4, GAMER_level );
  }
};


std::shared_ptr<AMRmesh> mesh_amrgrid_semiperiodic_sphere()
{
  int level_min = 4;
  int level_max = level_min+6;
  uint32_t bx=4, by=4, bz=4;


  std::cout << "// =========================================\n";
  std::cout << "// Testing GravitySolver_gc...\n";
  std::cout << "// Grid : amr - blocks " << bx << " -  levels " << level_min << " -> " << level_max << " \n";
  std::cout << "// Boundary conditions : (absorbing, absorbing, periodic) \n";
  std::cout << "// =========================================\n";

  std::cout << "Create mesh..." << std::endl;
  std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
  {
    int ndim = 3;   

    amr_mesh = std::make_shared<AMRmesh>(ndim, ndim, std::array<bool,3>{false,false,true}, level_min, level_max);

    for(int level=level_min+1; level<level_max; level++)
    {
      for( uint32_t iOct=0; iOct<amr_mesh->getNumOctants(); iOct++ )
      {
        auto oct_pos = amr_mesh->getCoordinates(iOct);
        real_t oct_size = amr_mesh->getSize(iOct)[0];
        
        for( uint32_t c=0; c<bx*by*bz; c++ )
        {
          uint32_t cz = c/(bx*by);
          uint32_t cy = (c - cz*bx*by)/bx;
          uint32_t cx = c - cz*bx*by - cy*bx;

          real_t x = oct_pos[IX] + (cx+0.5)*oct_size/bx - 0.5;
          real_t y = oct_pos[IY] + (cy+0.5)*oct_size/by - 0.5;
          real_t z = oct_pos[IZ] + (cz+0.5)*oct_size/bz - 0.5;

          real_t r2 = x*x + y*y + z*z;

          if( Hernquist::rho(r2) > Hernquist::rhomax(level) )
          {
            amr_mesh->setMarker(iOct, 1);
            break;
          }
        }
      }
      amr_mesh->adapt();
    }

    uint8_t levels = 6;
    amr_mesh->loadBalance(levels);
  }

  std::cout << "End Mesh creation." << std::endl;

  return amr_mesh;
}



/// Tests convergence with 
void test_GravitySolver( std::shared_ptr<AMRmesh> amr_mesh )
{ 
  // Content of .ini file used ton configure configmap and HydroParams
  std::string configmap_str = 
    "[run]\n"
    "solver_name=Hydro_Muscl_Block_3D \n"
    "[output]\n"
    "outputPrefix=test_GravitySolver\n"
    "write_variables=rho,gphi,gx,gy,gz\n"
    "[amr]\n"
    "use_block_data=yes\n"
    "bx=4\n"
    "by=4\n"
    "bz=4\n"
    "[mesh]\n"
    "ndim=3\n"
    "boundary_type_xmin=absorbing\n"
    "boundary_type_xmax=absorbing\n"
    "boundary_type_ymin=absorbing\n"
    "boundary_type_ymax=absorbing\n"
    "boundary_type_zmin=periodic\n"
    "boundary_type_zmax=periodic\n"
    "[gravity]\n"
    "gravity_type=field\n"
    "G=1\n"
    "\n";
  ConfigMap configMap(configmap_str);
  ForeachCell foreach_cell( *amr_mesh, configMap );

  std::cout << "Initialize User Data..." << std::endl;
  
  enum VarIndex_gravity{
    Irho,
    Igx,
    Igy,
    Igz,
    Iphi
  };

  UserData U_( configMap, foreach_cell );
  U_.new_fields({"rho", "gx", "gy", "gz", "gphi"});

  UserData::FieldAccessor U = U_.getAccessor({
    {"rho", Irho},
    {"gx", Igx},
    {"gy", Igy},
    {"gz", Igz},
    {"gphi", Iphi}
  });

  { 
    // Initialize U
    auto cells = foreach_cell.getCellMetaData();
    foreach_cell.foreach_cell( "Init", U.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      auto pos = cells.getCellCenter(iCell);
      real_t x = pos[IX]-0.5;
      real_t y = pos[IY]-0.5;
      real_t z = pos[IZ]-0.5;

      real_t r2 = x*x + y*y + z*z;

      U.at( iCell, Irho ) = Hernquist::rho(r2);
    });

    U_.exchange_ghosts(GhostCommunicator(amr_mesh));
  }

  Timers timers;

  // TODO use factory here instead?
  GravitySolver_cg gravitysolver( 
    configMap,
    foreach_cell,
    timers
  );
  std::unique_ptr<IOManager> iomanager = IOManagerFactory::make_instance( 
    "IOManager_hdf5",
    configMap,
    foreach_cell,
    timers
  );

  ScalarSimulationData scalar_data;
  
  gravitysolver.update_gravity_field( U_, scalar_data);

  int iter = 0;
  int time = 0;
  scalar_data.set<int>("iter", iter++);
  scalar_data.set<real_t>("time",time++);
  iomanager->save_snapshot(U_, scalar_data);
  
  real_t rcore = 2*Hernquist::r0;
  {
    real_t Phi_ana_mean = 0;
    real_t Phi_num_mean = 0;
    real_t Vcore = 0;
    auto cells = foreach_cell.getCellMetaData();
    foreach_cell.reduce_cell( "Init", U.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell, real_t& Phi_ana_mean, real_t& Phi_num_mean, real_t& Vcore )
    {
      auto pos = cells.getCellCenter(iCell);
      auto size = cells.getCellSize(iCell);

      real_t x = pos[IX]-0.5;
      real_t y = pos[IY]-0.5;
      real_t z = pos[IZ]-0.5;

      real_t r2 = x*x + y*y + z*z;
      real_t Vcell = size[IX]*size[IY]*size[IZ];
      if( r2 < rcore*rcore )
      {
        Phi_ana_mean += Hernquist::phi_periodic(x,y,z)*Vcell;
        Phi_num_mean += U.at(iCell, Iphi)*Vcell;
        Vcore += Vcell;
      } 
    }, Phi_ana_mean, Phi_num_mean, Vcore);

    Phi_ana_mean /= Vcore;
    Phi_num_mean /= Vcore;

    int err_count=0;
    foreach_cell.reduce_cell( "Init", U.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell, int err_count )
    {
      auto pos = cells.getCellCenter(iCell);
      real_t x = pos[IX]-0.5;
      real_t y = pos[IY]-0.5;
      real_t z = pos[IZ]-0.5;

      real_t r2 = x*x + y*y + z*z;

      real_t Phi_ana = Hernquist::phi_periodic(x,y,z);
      real_t Phi_num = U.at( iCell, Iphi ) + (Phi_ana_mean-Phi_num_mean);
      real_t Phi_err = std::abs(Phi_ana-Phi_num)/std::abs(Phi_ana);

      if( r2 < rcore*rcore )
        if( abs(Phi_ana - Phi_num) > 1E-2 )
          err_count ++;
      
      U.at(iCell, Igx) = Phi_ana;
      U.at(iCell, Igy) = Phi_num;
      U.at(iCell, Igz) = Phi_err;
    }, err_count);

    EXPECT_EQ(err_count, 0);
  }

  scalar_data.set<int>("iter", iter++);
  scalar_data.set<real_t>("time",time++);
  iomanager->save_snapshot(U_, scalar_data);  
}

} // namespace dyablo

TEST(Test_GravitySolver_cg, mesh_amrgrid_semiperiodic_sphere)
{
  dyablo::test_GravitySolver(dyablo::mesh_amrgrid_semiperiodic_sphere());
}


