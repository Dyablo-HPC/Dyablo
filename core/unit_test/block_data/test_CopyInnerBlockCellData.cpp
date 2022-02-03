/**
 * \file test_CopyInnerBlockCellData.cpp
 * \author Pierre Kestener
 */

#include <iomanip>

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#include <Kokkos_Macros.hpp> // for KOKKOS_ENABLE_XXX

#include <impl/Kokkos_Error.hpp>

#include "real_type.h"    // choose between single and double precision
#include "legacy/HydroParams.h"  // read parameter file
#include "FieldManager.h"


#include "legacy/CopyInnerBlockCellData.h"

using Device = Kokkos::DefaultExecutionSpace;

#include <boost/test/unit_test.hpp>
using namespace boost::unit_test;


namespace dyablo
{

// =======================================================================
// =======================================================================
void init_U(DataArrayBlock U,
            uint32_t nbOcts,
            int ndim, 
            blockSize_t blockSizes,
            uint32_t ghostWidth,
            id2index_t fm)
{

  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  uint32_t nbTeams_ = 4;
  
  team_policy_t policy (nbTeams_,
                        Kokkos::AUTO() /* team size chosen by kokkos */);


  uint32_t bx = blockSizes[IX];
  uint32_t by = blockSizes[IY];
  uint32_t bz = blockSizes[IZ];

  Kokkos::parallel_for("init_U", policy,
                       KOKKOS_LAMBDA(const thread_t& member)
                       {
                         uint32_t nbCells = ndim==2 ? 
                           bx*by : bx*by*bz;
                         uint32_t iOct = member.league_rank();

                         while (iOct <  nbOcts) {

                           Kokkos::parallel_for(
                             Kokkos::TeamVectorRange(member, nbCells),
                             KOKKOS_LAMBDA(const int32_t index) {
                               U(index, fm[ID], iOct) =
                                 index + iOct * nbCells;
                             }); // end TeamVectorRange
                           
                           iOct += nbTeams_;

                         }

                       });


} // init_U

// =======================================================================
// =======================================================================
void run_test(int argc, char *argv[], uint32_t bSize, uint32_t nbBlocks)
{

  /*
   * testing CopyInnerBlockCellDataFunctor
   */
  std::cout << "// =========================================\n";
  std::cout << "// Testing CopyInnerBlockCellDataFunctor ...\n";
  std::cout << "// =========================================\n";

  /*
   * read parameter file and initialize a ConfigMap object
   */
  // only MPI rank 0 actually reads input file
  std::string input_file = "./block_data/test_blast_2D_block.ini";
  ConfigMap configMap = ConfigMap::broadcast_parameters(input_file);

  int ndim = 2;
  GravityType gravity_type = GRAVITY_NONE;

  FieldManager fieldMgr = FieldManager::setup(ndim, gravity_type);

  auto fm = fieldMgr.get_id2index();

  // "geometry" setup
  uint32_t ghostWidth = configMap.getValue<uint32_t>("amr", "ghostwidth", 2);

  uint32_t bx = configMap.getValue<uint32_t>("amr", "bx", 0);
  uint32_t by = configMap.getValue<uint32_t>("amr", "by", 0);
  uint32_t bz = configMap.getValue<uint32_t>("amr", "bz", 1);

  uint32_t bx_g = bx + 2 * ghostWidth;
  uint32_t by_g = by + 2 * ghostWidth;
  uint32_t bz_g = bz + 2 * ghostWidth;

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

  uint32_t nbCellsPerOct = ndim==2 ? bx * by : bx * by * bz;
  uint32_t nbCellsPerOct_g =
      ndim==2 ? bx_g * by_g : bx_g * by_g * bz_g;

  uint32_t nbOctsPerGroup = configMap.getValue<uint32_t>("amr", "nbOctsPerGroup", 32);

  /*
   * allocate/initialize U / Ugroup
   */

  uint32_t nbOcts = 128;

  DataArrayBlock U = DataArrayBlock("U", nbCellsPerOct, fieldMgr.nbfields(), nbOcts);

  DataArrayBlock Ugroup = DataArrayBlock("Ugroup", nbCellsPerOct_g, fieldMgr.nbfields(), nbOctsPerGroup);

  uint32_t iGroup = 1;
  uint32_t iOctOffset = 1;

  uint32_t iOct = iOctOffset + iGroup * nbOctsPerGroup;

  // TODO initialize U, reset Ugroup - write a functor for that
  std::cout << "Initializing data...\n";
  init_U(U, nbOcts, ndim, blockSizes, ghostWidth, fm);

  std::cout << "Printing U data from iOct = " << iOct << "\n";
  for (uint32_t iz=0; iz<bz; ++iz)
  {
    for (uint32_t iy=0; iy<by; ++iy)
     {
      for (uint32_t ix=0; ix<bx; ++ix)
       {
         uint32_t index = ix + bx*(iy+by*iz);
         std::cout << std::right << std::setw(5) << U(index,fm[ID],iOct) << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }

  CopyInnerBlockCellDataFunctor::apply({ndim, gravity_type}, fm, 
                                       blockSizes,
                                       ghostWidth, 
                                       nbOcts,
                                       nbOctsPerGroup,
                                       U, Ugroup, iGroup);

  // print data from from the chosen iGroup 
  std::cout << "Printing Ugroup data from iOct = " << iOct << " | iOctLocal =" << iOctOffset << " and iGroup=" << iGroup << "\n";
  if (bz>1)
  {
    
    for (uint32_t iz=0; iz<bz_g; ++iz)
    {
      for (uint32_t iy=0; iy<by_g; ++iy)
      {
        for (uint32_t ix = 0; ix < bx_g; ++ix)
        {
          uint32_t index = ix + bx_g * (iy + by_g * iz);
          std::cout << std::right << std::setw(6) << Ugroup(index, fm[ID], iOctOffset) << " ";
        }
        std::cout << "\n";
      }
      std::cout << "\n";
    }
    
  } 
  else
  {
    
    for (uint32_t iy=0; iy<by_g; ++iy)
    {
      for (uint32_t ix = 0; ix < bx_g; ++ix)
      {
        uint32_t index = ix + bx_g * iy;
        std::cout << std::right << std::setw(6) << Ugroup(index, fm[ID], iOctOffset) << " ";

        if (ix>= ghostWidth and 
            iy>= ghostWidth and
            ix < ghostWidth + bx and
            iy < ghostWidth + by )
        {
          uint32_t uindex = (ix-ghostWidth) + bx * (iy-ghostWidth);
          BOOST_CHECK_CLOSE(Ugroup(index, fm[ID], iOctOffset),
                            U(uindex, fm[ID], iOct),
                            0.1);
        }


      }
      std::cout << "\n";
    }

  }

} // run_test



} // namespace dyablo

BOOST_AUTO_TEST_SUITE(dyablo)

BOOST_AUTO_TEST_SUITE(muscl_block)

BOOST_AUTO_TEST_CASE(test_CopyGhostBlockCellData)
{

  uint32_t bSize = 4;
  uint32_t nbBlocks = 32;

  run_test(framework::master_test_suite().argc,
           framework::master_test_suite().argv,
           bSize, nbBlocks);

}

BOOST_AUTO_TEST_SUITE_END() /* muscl_block */

BOOST_AUTO_TEST_SUITE_END() /* dyablo */