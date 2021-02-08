/**
 * \file InitRayleighTaylor.h
 * \author Maxime Delorme
 **/

/**
 * This is a template for initialization of a problem in dyablo
 *
 * The initial conditions of a problem require two functors:
 * - The first is the Data functor (InitXXXXXDataFunctor) which takes as input
 *   an already refined mesh and fills the Data array accordingly. This is done
 *   on host which means the copy to the device should be done outside of the 
 *   functor.
 * - The second functor is the Refine functor (InitXXXXXRefineFunctor) which
 *   takes an unrefined mesh and performs initial mesh refinement.
 *
 * To make your own problem init, here's a checklist of things to do: 
 *
 * (0-) Make a copy of this file with a clear name of your test (eg InitBlast.h)
 * (1-) Make a problem parameter structure if your problem requires initialization parameters
 *      this structure should go in src/shared/problems/XXXXXXParams.h. 
 *      If you don't need parameters for your ICs, look for xxParams and remove all occurences
 *  2-  Replace all the XXXXX by the name of your problem
 *  3-  Fill in the operator() functions of both InitXXXXXDataFunctor and InitXXXXXRefineFunctor
 * (4-) If not already done, copy and adapt the InitTemplate.cpp file as well !  
 *  5-  Update the HydroInitFunctors.h to add the function defined in InitTemplate.cpp
 *  6-  Update the init method in SolverHydroMusclBlock.cpp to include your init function
 *  7-  Update the CMakeLists.txt in src/muscl_block to account for the new .cpp file
 **/

#ifndef MUSCL_BLOCK_RAYLEIGHTAYLOR_H_
#define MUSCL_BLOCK_RAYLEIGHTAYLOR_H_

#include <limits> // for std::numeric_limits

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"

// Specific param structure of the problem
#include "shared/problems/RTParams.h"

#include "muscl_block/utils_block.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

// Additional includes here

namespace dyablo
{
namespace muscl_block
{

/**
 * Functor to initialize the Rayleigh-Taylor instability problem
 *
 * This functor takes as input an already refined mesh and initializes the user 
 * data on host. Copying the data from host to device should be done outside
 *
 * The parameter structures RtParams (if necessary) should be located in 
 * src/shared/problems
 **/
class InitRayleighTaylorDataFunctor
{
private:
  uint32_t nbTeams; //!< Number of thread teams;

public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::OpenMP, Kokkos::IndexType<uint32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) { nbTeams = nbTeams_; };

  // Constructor of the functor
  InitRayleighTaylorDataFunctor(std::shared_ptr<AMRmesh> pmesh,
                                HydroParams params,
                                RTParams rtParams,
                                id2index_t fm,
                                blockSize_t blockSizes,
                                DataArrayBlockHost Udata_h)
      : pmesh(pmesh), params(params), rtParams(rtParams),
        fm(fm), blockSizes(blockSizes), Udata_h(Udata_h) {};

  // Static function that initializes and applies the functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
                    HydroParams params,
                    ConfigMap configMap,
                    id2index_t fm,
                    blockSize_t blockSizes,
                    DataArrayBlockHost Udata_h)
  {
    // Loading up specific parameters if necessary
    RTParams rtParams(configMap);

    // Instantiation of the functor
    InitRayleighTaylorDataFunctor functor(pmesh, params, rtParams, fm, blockSizes, Udata_h);

    // And applying it to the mesh
    uint32_t nbTeams = configMap.getInteger("init", "nbTeams", 16);
    functor.setNbTeams(nbTeams);

    team_policy_t policy(Kokkos::OpenMP(),
                         nbTeams,
                         Kokkos::AUTO());

    Kokkos::parallel_for("dyablo::muscl_block::InitRayleighTaylorDataFunctor",
                         policy, functor);
  }

  // Actual initialization operator
  void operator()(thread_t member) const
  {
    uint32_t iOct = member.league_rank();

    const int &bx = blockSizes[IX];
    const int &by = blockSizes[IY];
    const int &bz = blockSizes[IZ];

    const bool twoD = (params.dimType == TWO_D);
    const bool threeD = !twoD;

    uint32_t nbCells = twoD ? bx * by : bx * by * bz;

    const real_t g = params.gy;

    // Looping over all octants
    while (iOct < pmesh->getNumOctants())
    {

      // get octant size
      const real_t octSize = pmesh->getSize(iOct);

      const real_t dx = octSize / bx;
      const real_t dy = octSize / by;

      // coordinates of the lower left corner
      const real_t x0 = pmesh->getCoordinates(iOct)[IX];
      const real_t y0 = pmesh->getCoordinates(iOct)[IY];

      // Iterating on all cells
      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, nbCells),
          KOKKOS_LAMBDA(const int32_t index) {
            coord_t iCoord;
            uint32_t &ix = iCoord[IX];
            uint32_t &iy = iCoord[IY];

            if (twoD)
            {
              iCoord = index_to_coord<2>(index, blockSizes);
            }
            else
            {
              iCoord = index_to_coord<3>(index, blockSizes);
            }

            real_t x = x0 + ix * dx + dx / 2;
            real_t y = y0 + iy * dy + dy / 2;

            real_t rho = (y > rtParams.rt_y ? rtParams.rho_up : rtParams.rho_down);
            real_t vy = rtParams.amplitude * (1.0 + cos(2.0 * M_PI * (x - 0.5))) * (1.0 + cos(2.0 * M_PI * (y - 0.5))) * 0.25;
            real_t P = (rtParams.P0 + rho * g * (y - 0.5)) / (params.settings.gamma0 - 1.0);

            Udata_h(index, fm[ID], iOct) = rho;
            Udata_h(index, fm[IU], iOct) = 0.0;
            Udata_h(index, fm[IV], iOct) = vy;
            Udata_h(index, fm[IP], iOct) = P;
            if (threeD)
              Udata_h(index, fm[IW], iOct) = 0.0;


            //std::cerr << "Gtype = " << params.gravity_type << " " << params.gx << " " << params.gy << " " << std::endl;

            if (params.gravity_type & GRAVITY_FIELD) {
              Udata_h(index, fm[IGX], iOct) = params.gx;
              Udata_h(index, fm[IGY], iOct) = params.gy;
              if (params.dimType == THREE_D)
                Udata_h(index, fm[IGZ], iOct) = params.gz;
            }
          });

      iOct += nbTeams;
    } // end while iOct
  }   // end operator()

  //! Mesh of the simulation
  std::shared_ptr<AMRmesh> pmesh;

  //! Global parameters of the simulation
  HydroParams params;

  //! Specific parameters
  RTParams rtParams;

  //! Field manager
  id2index_t fm;

  //! Block sizes array
  blockSize_t blockSizes;

  //! Data array on host
  DataArrayBlockHost Udata_h;

  //! Type of gravity selected
  uint8_t gravity_type;
};

/**
 * Functor to initialize the RayleighTaylor problem
 *
 * This functor takes as input an unrefined mesh and performs initial refinement
 **/
class InitRayleighTaylorRefineFunctor
{
public:
  using range_policy_t = Kokkos::RangePolicy<Kokkos::OpenMP>;
  InitRayleighTaylorRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
                                  HydroParams params,
                                  RTParams rtParams,
                                  uint8_t level_refine)
      : pmesh(pmesh), params(params), rtParams(rtParams), level_refine(level_refine){};

  // Applying the functor to the mesh
  static void apply(std::shared_ptr<AMRmesh> pmesh,
                    ConfigMap configMap,
                    HydroParams params,
                    uint8_t level_refine)
  {
    RTParams rtParams(configMap);

    // iterate functor for refinement
    InitRayleighTaylorRefineFunctor functor(pmesh, params, rtParams, level_refine);

    range_policy_t policy(Kokkos::OpenMP(), 0, pmesh->getNumOctants());
    Kokkos::parallel_for("dyablo::muscl_block::InitRayleighTaylorRefineFunctor", policy, functor);
  }

  void operator()(const uint32_t &iOct) const
  {
    uint8_t level = pmesh->getLevel(iOct);

    /**
     * The functor is called once per level to refine (from level_min to level_max-1)
     * The decision if the cell should be refined should be only made in this test
     * which checks that the current cell is effectively on the current level of 
     * consideration
     **/
    if (level == level_refine)
    {
      std::array<double, 3> center = pmesh->getCenter(iOct);
      const real_t y = center[1];
      const real_t rt_y = rtParams.rt_y;

      double cellSize = pmesh->getSize(iOct);

      bool should_refine = fabs(y - rt_y) < cellSize;
      if (should_refine)
        pmesh->setMarker(iOct, 1);
    } // end if level == level_refine
  }   // end operator()

  //! The mesh to refine
  std::shared_ptr<AMRmesh> pmesh;

  //! Global simulation parameters
  HydroParams params;

  //! Specific parameters
  RTParams rtParams;

  //! Level to refine
  uint8_t level_refine;
};

} // end namespace muscl_block
} // end namespace dyablo

#endif
