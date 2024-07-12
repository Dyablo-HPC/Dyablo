/**
 * \file test_CommGhosts.cpp
 * \author A. Durocher
 * Tests ghost octants MPI communication
 */
#include "gtest/gtest.h"

#include "mpi/ViewCommunicator.h"

#include "legacy/utils_block.h"
#include "amr/AMRmesh.h"
#include "utils/io/AMRMesh_output_vtk.h"

#include "foreach_cell/ForeachCell.h"
#include "foreach_cell/ForeachCell_utils.h"

namespace dyablo
{

// =======================================================================
// =======================================================================
template<typename Array_t>
Kokkos::LayoutLeft layout(int bx, int by, int bz, int nbfields, int nbOcts );

real_t& at(const DataArrayBlock::HostMirror& U, uint32_t c, uint32_t f, uint32_t iOct )
{
  return U(c,f,iOct);
}
int extent(const DataArrayBlock& U, int i)
{
  return U.extent(i);
}
template<>
Kokkos::LayoutLeft layout<DataArrayBlock>(int bx, int by, int bz, int nbfields, int nbOcts )
{
  return Kokkos::LayoutLeft(bx*by*bz,nbfields,nbOcts);
}

real_t& at(const DataArray::HostMirror& U, uint32_t c, uint32_t f, uint32_t iOct )
{
  return U(iOct,f);
}
int extent(const DataArray& U, int i)
{
  switch(i){
    case 1:
      return U.extent(1);
    case 2: 
      return U.extent(0);
    default:
      return 1;    
  }
}
template<>
Kokkos::LayoutLeft layout<DataArray>(int bx, int by, int bz, int nbfields, int nbOcts )
{
  return Kokkos::LayoutLeft(nbOcts,nbfields);
}

template< typename Array_t >
void run_test()
{
  std::cout << "// =========================================\n";
  std::cout << "// Testing ViewCommunicator ...\n";
  std::cout << "// =========================================\n";

  std::cout << "Create mesh..." << std::endl;
  std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
  {
    int ndim = 3;
    amr_mesh = std::make_shared<AMRmesh>(ndim, ndim, std::array<bool,3>{false,false,false}, 3, 7);
    //amr_mesh->setBalanceCodimension(ndim);
    //uint32_t idx = 0;
    //amr_mesh->setBalance(idx,true);
    // mr_mesh->setPeriodic(0);
    // amr_mesh->setPeriodic(1);
    // amr_mesh->setPeriodic(2);
    // amr_mesh->setPeriodic(3);
    //amr_mesh->setPeriodic(4);
    //amr_mesh->setPeriodic(5);

    debug::output_vtk("before_initial", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt1", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt2", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt3", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt4", *amr_mesh);

    //amr_mesh->loadBalance();
    amr_mesh->updateConnectivity();
  }

  uint32_t bx = 8;
  uint32_t by = 8;
  uint32_t bz = 8;
  uint32_t nbfields = 3;
  uint32_t nbOcts = amr_mesh->getNumOctants();
  auto Ulayout = layout<Array_t>(bx,by,bz,nbfields,nbOcts);

  Array_t U("U", Ulayout );
  uint32_t nbCellsPerOct = extent(U, 0);  
  { // Initialize U
    typename Array_t::HostMirror U_host = Kokkos::create_mirror_view(U);
    for( uint32_t iOct=0; iOct<nbOcts; iOct++ )
    {
      auto oct_pos = amr_mesh->getCoordinates(iOct);
      real_t oct_size = amr_mesh->getSize(iOct)[0];
      
      for( uint32_t c=0; c<nbCellsPerOct; c++ )
      {
        uint32_t cz = c/(bx*by);
        uint32_t cy = (c - cz*bx*by)/bx;
        uint32_t cx = c - cz*bx*by - cy*bx;

        at(U_host, c, IX, iOct) = oct_pos[IX] + cx*oct_size/bx;
        at(U_host, c, IY, iOct) = oct_pos[IY] + cy*oct_size/by;
        at(U_host, c, IZ, iOct) = oct_pos[IZ] + cz*oct_size/bz;
      }
    }
    Kokkos::deep_copy( U, U_host );
  }

  dyablo::ViewCommunicator ghost_communicator = ViewCommunicator::from_mesh( amr_mesh->getMesh() );

  auto Ughostlayout = layout<Array_t>(bx,by,bz,nbfields,ghost_communicator.getNumGhosts());
  Array_t Ughost("Ughost", Ughostlayout);
  constexpr int iOct = std::is_same_v<Array_t, DataArrayBlock> ? 2 : 0;
  ghost_communicator.exchange_ghosts<iOct>(U, Ughost);

  // Test Ughost
  {
    uint32_t nGhosts = amr_mesh->getNumGhosts();

    std::cout << "Check Ughost ( nGhosts=" << nGhosts << ")" << std::endl;

    EXPECT_EQ(extent( Ughost, 2), nGhosts);

    //if(nGhosts!=0)
    {
      EXPECT_EQ(extent( Ughost, 0), nbCellsPerOct);
      EXPECT_EQ(extent( Ughost, 1), nbfields);
    }
    

    auto Ughost_host = Kokkos::create_mirror_view(Ughost);
    Kokkos::deep_copy(Ughost_host, Ughost);

    for( uint32_t iGhost=0; iGhost<nGhosts; iGhost++ )
    {
      auto oct_pos = amr_mesh->getCoordinatesGhost(iGhost);
      real_t oct_size = amr_mesh->getSizeGhost(iGhost)[0];
      
      for( uint32_t c=0; c<nbCellsPerOct; c++ )
      {
        uint32_t cz = c/(bx*by);
        uint32_t cy = (c - cz*bx*by)/bx;
        uint32_t cx = c - cz*bx*by - cy*bx;

        real_t expected_x = oct_pos[IX] + cx*oct_size/bx;
        real_t expected_y = oct_pos[IY] + cy*oct_size/by;
        real_t expected_z = oct_pos[IZ] + cz*oct_size/bz;

        EXPECT_NEAR( at(Ughost_host, c, IX, iGhost), expected_x , 0.01);
        EXPECT_NEAR( at(Ughost_host, c, IY, iGhost), expected_y , 0.01);
        EXPECT_NEAR( at(Ughost_host, c, IZ, iGhost), expected_z , 0.01);
      }
    }
  }

} // run_test



} // namespace dyablo

TEST(dyablo, test_ViewCommunicator_block)
{

  dyablo::run_test<dyablo::DataArrayBlock>();

}

TEST(dyablo, test_ViewCommunicator_cell)
{

  dyablo::run_test<dyablo::DataArray>();

}

void test_ViewCommunicator_domains()
{
  auto comm_world = dyablo::GlobalMpiSession::get_comm_world();
  uint32_t mpi_size = comm_world.MPI_Comm_size();
  uint32_t mpi_rank = comm_world.MPI_Comm_rank();

  int N_old = 1000;
  Kokkos::View< double**, Kokkos::LayoutLeft > particles("particles", N_old, 2);
  Kokkos::View< int* > domains("domains", N_old);

  Kokkos::parallel_for( "fill_domain", N_old,
    KOKKOS_LAMBDA(int i)
  {
    int domain = (i/3)%mpi_size;
    domains(i) = domain;
    particles(i,0) = mpi_rank;
    particles(i,1) = domain;
  });

  dyablo::ViewCommunicator ghost_comm(domains, comm_world);
  int N_new = ghost_comm.getNumGhosts();

  int N_new_tot;
  comm_world.MPI_Allreduce( &N_new, &N_new_tot, 1, dyablo::MpiComm::MPI_Op_t::SUM );
  ASSERT_EQ( N_old*mpi_size, N_new_tot );

  Kokkos::View< double**, Kokkos::LayoutLeft > particles_new("particles_new", N_new, 2);
  ghost_comm.exchange_ghosts<0>(particles, particles_new);

  int errors = 0;
  Kokkos::parallel_reduce( "check_domain", N_new,
    KOKKOS_LAMBDA(int i, int& errors)
  {
    if( particles_new(i,1) != mpi_rank )
    {
      //printf( "Error : Rank %d, Particle %d, Domain %d\n", mpi_rank, i, (int)particles_new(i,1) );
      errors++;
    }
  }, errors);

  EXPECT_EQ(0, errors);
}

TEST(dyablo, test_ViewCommunicator_domains)
{
  test_ViewCommunicator_domains();
}

void run_test_reduce()
{
  using namespace dyablo;

  std::cout << "// =========================================\n";
  std::cout << "// Testing ViewCommunicator ...\n";
  std::cout << "// =========================================\n";

  std::cout << "Create mesh..." << std::endl;
  std::shared_ptr<AMRmesh> amr_mesh; //solver->amr_mesh 
  {
    int ndim = 3;
    amr_mesh = std::make_shared<AMRmesh>(ndim, ndim, std::array<bool,3>{true,true,true}, 3, 7);
    //amr_mesh->setBalanceCodimension(ndim);
    //uint32_t idx = 0;
    //amr_mesh->setBalance(idx,true);
    // mr_mesh->setPeriodic(0);
    // amr_mesh->setPeriodic(1);
    // amr_mesh->setPeriodic(2);
    // amr_mesh->setPeriodic(3);
    //amr_mesh->setPeriodic(4);
    //amr_mesh->setPeriodic(5);

    debug::output_vtk("before_initial", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt1", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt2", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt3", *amr_mesh);
    if( amr_mesh->getRank() == 0 )
      amr_mesh->setMarker(amr_mesh->getNumOctants()-1 ,1);      
    amr_mesh->adapt();
    debug::output_vtk("after_adapt4", *amr_mesh);

    amr_mesh->loadBalance();
    amr_mesh->updateConnectivity();
  }

  // Content of .ini file used ton configure configmap and HydroParams
  ConfigMap configMap("");

  // Fill default values
  constexpr int ndim = 3;
  configMap.getValue<int>("run", "ndim", ndim);
  /*uint32_t bx = */configMap.getValue<uint32_t>("amr", "bx", 8);
  /*uint32_t by = */configMap.getValue<uint32_t>("amr", "by", 8);
  /*uint32_t bz = */configMap.getValue<uint32_t>("amr", "bz", (ndim==2)?1:8);

  ForeachCell foreach_cell( *amr_mesh, configMap );
  enum VarIndex_test{ IU,IV,IW };
  FieldManager fieldManager({IU,IV,IW});
  typename ForeachCell::CellArray_global_ghosted U = foreach_cell.allocate_ghosted_array("U", fieldManager);

  using CellIndex = typename ForeachCell::CellIndex;

  foreach_cell.foreach_cell( "Fill_neighbors", U,
    CELL_LAMBDA( const CellIndex& iCell )
  {
    auto fill_neighbor = [&](CellIndex::offset_t offset, VarIndex iVar)
    {
      CellIndex iCell_n = iCell.getNeighbor_ghost( offset, U );    
      assert( iCell_n.is_valid() ); // This test uses periodic
      assert( iCell_n.level_diff() >= -1 &&  iCell_n.level_diff() <= 1 );
      if( iCell_n.level_diff() == 0 ) // Same size
      {
        Kokkos::atomic_add(&U.at( iCell_n, iVar ), 1); 
      }
      else if( iCell_n.level_diff() == 1 ) // Neighbor is bigger
      {
        Kokkos::atomic_add(&U.at( iCell_n, iVar ), 0.25); 
      }
      else if( iCell_n.level_diff() == -1 ) // Neighbors are smaller
      {
        foreach_smaller_neighbor<ndim>( iCell_n, offset, U,
          [&]( const CellIndex& iCell_ns )
        {
          Kokkos::atomic_add(&U.at( iCell_ns, iVar ), 1);
        });       
      }
      else
      {
        assert(false);
      }
    };
    fill_neighbor( { 1, 0, 0}, IU );
    fill_neighbor( {-1, 0, 0}, IU );
    fill_neighbor( { 0, 1, 0}, IV );
    fill_neighbor( { 0,-1, 0}, IV );
    fill_neighbor( { 0, 0, 1}, IW );
    fill_neighbor( { 0, 0,-1}, IW );
  });

  dyablo::ViewCommunicator ghost_communicator = ViewCommunicator::from_mesh( amr_mesh->getMesh() );
  ghost_communicator.reduce_ghosts<2>( U.U, U.Ughost );

  int nerrors = 0;
  foreach_cell.reduce_cell( "check_values", U,
    CELL_LAMBDA( const CellIndex& iCell, int& nerrors )
  {
    constexpr real_t eps = 1e-10;
    real_t U_IU = U.at( iCell, IU );
    real_t U_IV = U.at( iCell, IV );
    real_t U_IW = U.at( iCell, IW );

    #ifndef __CUDA_ARCH__
    EXPECT_NEAR( 2, U_IU, eps);
    EXPECT_NEAR( 2, U_IV, eps);
    EXPECT_NEAR( 2, U_IW, eps);
    #endif

    if( abs( U_IU - 2) > eps )
      nerrors++;
    if( abs( U_IV - 2) > eps )
      nerrors++;
    if( abs( U_IW - 2) > eps )
      nerrors++;
    
  }, nerrors);

  EXPECT_EQ(nerrors, 0);

} // run_test_reduce

TEST(dyablo, test_ViewCommunicator_reduce)
{
  run_test_reduce();
}
