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

  static real_t rho(real_t r2)
  {
    real_t r = std::sqrt(r2)*L;
    return rho0/( r/r0 * std::pow(1 + r/r0, 3) );
  }

  static real_t phi(real_t r2)
  {
    real_t r = std::sqrt(r2)*L;
    return -G*M / (r+r0);
  }

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
  uint32_t nbOcts = amr_mesh->getNumOctants();
  
  // Content of .ini file used ton configure configmap and HydroParams
  std::string configmap_str = 
    "[run]\n"
    "solver_name=Hydro_Muscl_Block_3D \n"
    "[output]\n"
    "outputPrefix=test_GravitySolver\n"
    "write_variables=rho,igphi,igx,igy,igz\n"
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

  FieldManager fieldMgr = FieldManager({ID,IGPHI,IGX,IGY,IGZ});

  ForeachCell foreach_cell( *amr_mesh, configMap );

  std::cout << "Initialize User Data..." << std::endl;
  ForeachCell::CellArray_global_ghosted U = foreach_cell.allocate_ghosted_array("U", fieldMgr);

  // TODO use ForeachCell and avoix extracting bx, by, bz
  int bx = U.bx;
  int by = U.by;
  int bz = U.bz;
  id2index_t fm = U.fm;
  uint32_t nbCellsPerOct = bx*by*bz;
  { 
    // Initialize U
    DataArrayBlock::HostMirror U_host = Kokkos::create_mirror_view(U.U);
    for( uint32_t iOct=0; iOct<nbOcts; iOct++ )
    {
      auto oct_pos = amr_mesh->getCoordinates(iOct);
      real_t oct_size = amr_mesh->getSize(iOct)[0];
      
      for( uint32_t c=0; c<nbCellsPerOct; c++ )
      {
        uint32_t cz = c/(bx*by);
        uint32_t cy = (c - cz*bx*by)/bx;
        uint32_t cx = c - cz*bx*by - cy*bx;

        real_t x = oct_pos[IX] + (cx+0.5)*oct_size/bx - 0.5;
        real_t y = oct_pos[IY] + (cy+0.5)*oct_size/by - 0.5;
        real_t z = oct_pos[IZ] + (cz+0.5)*oct_size/bz - 0.5;

        real_t r2 = x*x + y*y + z*z;

        U_host(c, fm[ID], iOct) = Hernquist::rho(r2);
      }
    }
    Kokkos::deep_copy( U.U, U_host );
    U.exchange_ghosts(GhostCommunicator(amr_mesh));
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

  ForeachCell::CellArray_global_ghosted Uout = foreach_cell.allocate_ghosted_array("Uout", fieldMgr);
  Kokkos::deep_copy(Uout.U, U.U);

  gravitysolver.update_gravity_field( U, Uout );

  int iter = 0;
  int time = 0;
  iomanager->save_snapshot(Uout, iter++, time++);

  
  real_t rcore = 2*Hernquist::r0;
  {
    DataArrayBlock::HostMirror U_host = Kokkos::create_mirror_view(Uout.U);
    Kokkos::deep_copy(U_host, Uout.U);
    real_t Phi_ana_mean = 0;
    real_t Phi_num_mean = 0;
    real_t Vcore = 0;
    for( uint32_t iOct=0; iOct<nbOcts; iOct++ )
    {
      auto oct_pos = amr_mesh->getCoordinates(iOct);
      real_t oct_size = amr_mesh->getSize(iOct)[0];
      real_t Vcell = (oct_size/bx)*(oct_size/by)*(oct_size/bz);
      
      for( uint32_t c=0; c<nbCellsPerOct; c++ )
      {
        uint32_t cz = c/(bx*by);
        uint32_t cy = (c - cz*bx*by)/bx;
        uint32_t cx = c - cz*bx*by - cy*bx;

        real_t x = oct_pos[IX] + (cx+0.5)*oct_size/bx - 0.5;
        real_t y = oct_pos[IY] + (cy+0.5)*oct_size/by - 0.5;
        real_t z = oct_pos[IZ] + (cz+0.5)*oct_size/bz - 0.5;

        real_t r2 = x*x + y*y + z*z;
        if( r2 < rcore*rcore )
        {
          Phi_ana_mean += Hernquist::phi_periodic(x,y,z)*Vcell;
          Phi_num_mean += U_host(c, fm[IGPHI], iOct)*Vcell;
          Vcore += Vcell;
        }        
      }
    }

    Phi_ana_mean /= Vcore;
    Phi_num_mean /= Vcore;

    for( uint32_t iOct=0; iOct<nbOcts; iOct++ )
    {
      auto oct_pos = amr_mesh->getCoordinates(iOct);
      real_t oct_size = amr_mesh->getSize(iOct)[0];
      
      for( uint32_t c=0; c<nbCellsPerOct; c++ )
      {
        uint32_t cz = c/(bx*by);
        uint32_t cy = (c - cz*bx*by)/bx;
        uint32_t cx = c - cz*bx*by - cy*bx;

        real_t x = oct_pos[IX] + (cx+0.5)*oct_size/bx - 0.5;
        real_t y = oct_pos[IY] + (cy+0.5)*oct_size/by - 0.5;
        real_t z = oct_pos[IZ] + (cz+0.5)*oct_size/bz - 0.5;

        real_t r2 = x*x + y*y + z*z;

        real_t Phi_ana = Hernquist::phi_periodic(x,y,z);
        real_t Phi_num = U_host(c, fm[IGPHI], iOct)+(Phi_ana_mean-Phi_num_mean);
        real_t Phi_err = std::abs(Phi_ana-Phi_num)/std::abs(Phi_ana);

        if( r2 < rcore*rcore )
          EXPECT_NEAR( Phi_ana, Phi_num, 1E-2 );
        U_host(c, fm[IGPHI], iOct) = U_host(c, fm[IGPHI], iOct);
        U_host(c, fm[IGX], iOct) = Phi_ana;
        U_host(c, fm[IGY], iOct) = Phi_num;
        U_host(c, fm[IGZ], iOct) = Phi_err;


      }
    }
    Kokkos::deep_copy( Uout.U, U_host );
  }

  iomanager->save_snapshot(Uout, iter++, time++);
  
}

} // namespace dyablo

TEST(Test_GravitySolver_cg, mesh_amrgrid_semiperiodic_sphere)
{
  dyablo::test_GravitySolver(dyablo::mesh_amrgrid_semiperiodic_sphere());
}


