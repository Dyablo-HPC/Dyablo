/**
 * \file InitSod.h
 * \author Maxime Delorme
 **/
#ifndef MUSCL_BLOCK_HYDRO_INIT_SOD_
#define MUSCL_BLOCK_HYDRO_INIT_SOD_

#include <limits> // for std::numeric_limits

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"

#include "muscl_block/utils_block.h"

#include "bitpit_PABLO.hpp"
#include "shared/amr/AMRmesh.h"

namespace dyablo { namespace muscl_block {

/**
 * This functor initialises a Sod shock tube in 2D
 **/
class InitSodDataFunctor {
 private:
  uint32_t nbTeams;

 public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::OpenMP, Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) {nbTeams = nbTeams_;};

  InitSodDataFunctor(std::shared_ptr<AMRmesh> pmesh,
		     HydroParams              params,
		     ConfigMap                configMap,
		     id2index_t               fm,
		     blockSize_t              blockSizes,
		     DataArrayBlockHost       Udata_h)
    : pmesh(pmesh), params(params), configMap(configMap), fm(fm),
      blockSizes(blockSizes), Udata_h(Udata_h) {
    
    const real_t gamma0 = params.settings.gamma0;
    
    // Left state
    uL[ID] = configMap.getFloat("sod", "rho_L", 1.0);
    uL[IU] = 0.0;
    uL[IV] = 0.0;
    uL[IP] = configMap.getFloat("sod", "P_L", 1.0/(gamma0-1.0));

    // Right state
    uR[ID] = configMap.getFloat("sod", "rho_R", 0.125);
    uR[IU] = 0.0;
    uR[IV] = 0.0;
    uR[IP] = configMap.getFloat("sod", "P_R", 0.1/(gamma0-1.0));

    if (params.dimType == THREE_D) {
      uL[IW] = 0.0;
      uR[IW] = 0.0;
    }

    // Location of the discontinuity
    discont_x = configMap.getFloat("sod", "discont_x", 0.5);
  }; // constructor

  // static method which creates and applies the functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams              params,
		    ConfigMap                configMap,
		    id2index_t               fm,
		    blockSize_t              blockSizes,
		    DataArrayBlockHost       Udata_h) {
    // Functor init
    InitSodDataFunctor functor(pmesh, params, configMap, fm, blockSizes, Udata_h);

    // Kokkos execution policy
    uint32_t nbTeams = configMap.getInteger("init", "nbTeams", 16);
    functor.setNbTeams(nbTeams);
    
    // Perform initialization on host
    team_policy_t policy (Kokkos::OpenMP(),
                          nbTeams,
                          Kokkos::AUTO() /* team size chosen by kokkos */);
    
    // Init is performed on host
    Kokkos::parallel_for("dyablo::muscl_block::InitSodDataFunctor", policy, functor);
  }

  void operator()(thread_t member) const {
    uint32_t iOct = member.league_rank();
    
    const int &bx = blockSizes[IX];
    const int &by = blockSizes[IY];
    const int &bz = blockSizes[IZ];

    uint32_t nbCells = params.dimType == TWO_D ? bx*by : bx*by*bz;

    while (iOct < pmesh->getNumOctants()) {
      // Extracting position of the current cell
      const real_t octSize  = pmesh->getSize(iOct);
      const real_t dx       = octSize / bx;
      const real_t x0 = pmesh->getCoordinates(iOct)[IX];

      Kokkos::parallel_for(
	Kokkos::TeamVectorRange(member, nbCells),
	[&](const int32_t index) {
	  coord_t iCoord = (params.dimType == TWO_D
			    ? index_to_coord<2>(index, blockSizes)
			    : index_to_coord<3>(index, blockSizes));
	  const real_t x = x0 + iCoord[IX] * dx + dx * 0.5;

	  const HydroState2d &st = (x < discont_x ? uL : uR);
	  
	  Udata_h(index, fm[ID], iOct) = st[ID];
	  Udata_h(index, fm[IP], iOct) = st[IP];
	  Udata_h(index, fm[IU], iOct) = st[IU];
	  Udata_h(index, fm[IV], iOct) = st[IV];

	  if (params.dimType == THREE_D)
	    Udata_h(index, fm[IW], iOct) = st[IW];
	}); // end parallel_for

      iOct += nbTeams;
    } // end while
    
  } // end operator()

  //! AMR mesh
  std::shared_ptr<AMRmesh> pmesh;

  //! General parameters
  HydroParams              params;

  //! Configuration map
  ConfigMap                configMap;

  //! Field manager
  id2index_t               fm;

  //! Block sizes
  blockSize_t              blockSizes;

  //! Heavy data on host
  DataArrayBlockHost       Udata_h;

  //! Discontinuity location
  real_t discont_x;

  //! Left state
  HydroState2d uL;

  //! Right state
  HydroState2d uR; 
}; // InitSodDataFunctor

/**
 * Refinement functor for Sod shock test
 **/
class InitSodRefineFunctor {
public:
  using range_policy_t = Kokkos::RangePolicy<Kokkos::OpenMP>;

  InitSodRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
		       HydroParams              params,
		       ConfigMap                configMap,
		       int                      level_refine)
    : pmesh(pmesh), params(params), configMap(configMap), level_refine(level_refine)
  {
    discont_x = configMap.getFloat("sod", "discont_x", 0.5);
  };

  // Static method to create and apply the functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams              params,
		    ConfigMap                configMap,
		    int                      level_refine)
  {
    InitSodRefineFunctor functor(pmesh, params, configMap, level_refine);
    range_policy_t policy(Kokkos::OpenMP(), 0, pmesh->getNumOctants());
    Kokkos::parallel_for("dyablo::muscl_block::InitSodRefineFunctor", policy, functor);
  }; // end apply

  void operator() (const size_t &iOct) const {

    // Getting current level
    uint8_t level = pmesh->getLevel(iOct);

    if (level == level_refine) {
      std::array<double, 3> center = pmesh->getCenter(iOct);
      const real_t x = center[0];

      double cellSize2 = pmesh->getSize(iOct)*0.75;

      if (fabs(x-discont_x) < cellSize2)
	pmesh->setMarker(iOct, 1);
    } // end if level == level_refine
  } // end operator()

  //! AMR mesh
  std::shared_ptr<AMRmesh> pmesh;

  //! General parameters
  HydroParams params;

  //! Configuration map
  ConfigMap configMap;

  //! Level of refinement
  int level_refine;

  //! Location of the discontinuity
  real_t discont_x;
}; // InitSodRefineFunctor

    
} // namespace dyablo
} // namespace muscl_block

#endif
