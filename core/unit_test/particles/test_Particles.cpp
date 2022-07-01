#include "gtest/gtest.h"

#include "amr/AMRmesh.h"
#include "amr/LightOctree.h"
#include "io/IOManager.h"

#include "particles/ForeachParticle.h"

namespace dyablo
{

void run_test(int ndim)
{
  std::cout << "// =========================================\n";
  std::cout << "// Testing Paricles ..\n";
  std::cout << "// =========================================\n";

  std::cout << "Create mesh..." << std::endl;

  int level_min = 1;
  int level_max = 8;
  std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
  {
    amr_mesh = std::make_shared<AMRmesh>(ndim, ndim, std::array<bool,3>{false,false,false}, level_min, level_max );
    // amr_mesh->setBalanceCodimension(ndim);
    // uint32_t idx = 0;
    // amr_mesh->setBalance(idx,true);
    // amr_mesh->setPeriodic(0);
    // amr_mesh->setPeriodic(1);
    // amr_mesh->setPeriodic(2);
    // amr_mesh->setPeriodic(3);
    // amr_mesh->setPeriodic(4);
    // amr_mesh->setPeriodic(5);

    amr_mesh->adaptGlobalRefine();
    amr_mesh->adaptGlobalRefine();
    amr_mesh->adaptGlobalRefine();

    amr_mesh->adapt();
    amr_mesh->updateConnectivity();
    amr_mesh->loadBalance();
    amr_mesh->updateConnectivity();
  }

  //Timers timers;

  std::string config_str = 
    "[output]\n"
    "hdf5_enabled=true\n"
    "write_mesh_info=true\n"
    "write_variables=rho_vx,rho_vy,rho_vz\n"
    "write_particle_variables=rho_vx,rho_vy,rho_vz\n"
    "write_iOct=false\n"
    "outputPrefix=output\n"
    "outputDir=./\n"
    "[amr]\n"
    "use_block_data=true\n"
    "bx=8\n"
    "by=8\n";
  ConfigMap configMap(config_str);

  configMap.getValue<int>("mesh", "ndim", ndim);
  configMap.getValue<int>("amr", "bz", ndim==2?1:8);
  
  ForeachCell foreach_cell( *amr_mesh, configMap );
  
  enum VarIndex_test{S, CX, CY, CZ};

  FieldManager field_manager({S,CX,CY,CZ});

  ForeachCell::CellArray_global_ghosted U = foreach_cell.allocate_ghosted_array("U", field_manager);
  ForeachCell::CellMetaData cmd = foreach_cell.getCellMetaData();

  // Initialize U
  foreach_cell.foreach_cell( "Init_U", U,
    CELL_LAMBDA( const ForeachCell::CellIndex& iCell )
  {
    U.at(iCell, S) = cmd.getCellSize(iCell)[IX];
    auto pos = cmd.getCellCenter(iCell);
    U.at(iCell, CX) = pos[IX];
    U.at(iCell, CY) = pos[IY];
    U.at(iCell, CZ) = pos[IZ];
  });

  std::string iomanager_id = "IOManager_hdf5";
  Timers timers;
  std::unique_ptr<IOManager> io_manager = IOManagerFactory::make_instance( iomanager_id,
    configMap,
    foreach_cell,
    timers
  );

  uint32_t px=10, py=10, pz=10;
  uint32_t nParticles_tot = px*py*pz;
  ForeachParticle foreach_particle( *amr_mesh, configMap);

  int rank = GlobalMpiSession::get_comm_world().MPI_Comm_rank();
  uint32_t nParticles = (rank == 0) ? nParticles_tot : 0;

  ParticleData particles( "Particles", nParticles, field_manager );

  // Set particle positions to form a grid inside the domain
  foreach_particle.foreach_particle( "set_particle_pos", particles,
    PARTICLE_LAMBDA( ParticleData::ParticleIndex iPart )
  {
      uint32_t ix =  iPart%px;
      uint32_t iy = (iPart/px)%py;
      uint32_t iz =  iPart/(px*py);

      particles.pos( iPart, IX ) = (ix+0.5)/px;
      particles.pos( iPart, IY ) = (iy+0.5)/py;
      particles.pos( iPart, IZ ) = (ndim-2)*((iz+0.5)/pz);
  });

  //io_manager->save_snapshot(U, 0, 1);

  // Exchange particles between MPI domains to match local AMR mesh
  foreach_particle.distribute( particles );

  //io_manager->save_snapshot(U, 1, 2);
  
  { // Check particle count
    int nParticles_tot_old = nParticles_tot;
    int nParticles_tot_new = 0;
    int nParticles_new = particles.getNumParticles();
    GlobalMpiSession::get_comm_world().MPI_Allreduce(&nParticles_new, &nParticles_tot_new, 1, MpiComm::MPI_Op_t::SUM);
    EXPECT_EQ(nParticles_tot_new, nParticles_tot_old);
  }

  // Interact with AMR grid :
  // Copy CX, CY, CZ, S from grid to particle (filled with cell center and size)
  foreach_particle.foreach_particle( "copy_cell_center", particles,
    PARTICLE_LAMBDA( ParticleData::ParticleIndex iPart )
  {
    // TODO abstract interface to convert particle index directly to cell index (without needing position)
    //ForeachCell::CellIndex iCell = pci.getCell( particles, iPart );
    ForeachCell::CellMetaData::pos_t particle_pos = {
      particles.pos( iPart, IX ),
      particles.pos( iPart, IY ),
      particles.pos( iPart, IZ )
    };
    ForeachCell::CellIndex iCell = cmd.getCellFromPos( particle_pos );

    particles.at( iPart, S ) = U.at( iCell, S );
    particles.at( iPart, CX ) = U.at( iCell, CX );
    particles.at( iPart, CY ) = U.at( iCell, CY );
    particles.at( iPart, CZ ) = U.at( iCell, CZ );
  });

  //io_manager->save_snapshot(U, 2, 3);

  int nerrors = 0;
  foreach_particle.reduce_particle( "check_particle_cell", particles,
    PARTICLE_LAMBDA( ParticleData::ParticleIndex iPart, int& nerrors )
  {
    real_t px = particles.pos( iPart, IX );
    real_t py = particles.pos( iPart, IY );
    real_t pz = particles.pos( iPart, IZ );
    real_t s = particles.at( iPart, S );
    real_t cx = particles.at( iPart, CX );
    real_t cy = particles.at( iPart, CY );
    real_t cz = particles.at( iPart, CZ );

    if( abs(px - cx) > s/2 )
      nerrors++;
    if( abs(py - cy) > s/2 )
      nerrors++;
    if( abs(pz - cz) > s/2 )
      nerrors++;

    #ifndef __CUDA_ARCH__
    EXPECT_NEAR( px, cx, s/2 );
    EXPECT_NEAR( py, cy, s/2 );
    EXPECT_NEAR( pz, cz, s/2 );
    #endif
  }, nerrors);
  EXPECT_EQ( 0, nerrors );
} // run_test

} // namespace dyablo

class Test_Particles
  : public testing::TestWithParam<std::tuple<int>> 
{};

TEST_P(Test_Particles, getCell_works)
{
  int ndim = std::get<0>(GetParam());
  dyablo::run_test(ndim);
}

INSTANTIATE_TEST_SUITE_P(
    Test_Particles, Test_Particles,
    testing::Combine(
        testing::Values(2,3)
    ),
    [](const testing::TestParamInfo<Test_Particles::ParamType>& info) {
      std::string name = 
          (std::get<0>(info.param) == 2 ? std::string("2D") : std::string("3D"));
      return name;
    }
);