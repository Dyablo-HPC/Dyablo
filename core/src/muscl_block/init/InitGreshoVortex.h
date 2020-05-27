/**
 * \file InitGreshoVortex.h
 * \author Pierre Kestener
 */
#ifndef MUSCL_BLOCK_HYDRO_INIT_GRESHO_VORTEX_H_
#define MUSCL_BLOCK_HYDRO_INIT_GRESHO_VORTEX_H_

#include <limits> // for std::numeric_limits

#include "shared/FieldManager.h"
#include "shared/HydroState.h"
#include "shared/kokkos_shared.h"
#include "shared/problems/GreshoVortexParams.h"

#include "muscl_block/utils_block.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

namespace dyablo {
namespace muscl_block {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve the gresho vortex problem.
 *
 * This functor takes as input a mesh, already refined, and initializes
 * user data.
 *
 * Initial conditions is refined near strong density gradients.
 *
 */

class InitGreshoVortexDataFunctor {

private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::OpenMP,Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) {nbTeams = nbTeams_;}; 

  InitGreshoVortexDataFunctor(std::shared_ptr<AMRmesh> pmesh,
                              HydroParams        params, 
                              GreshoVortexParams gvParams,
                              id2index_t         fm,
                              blockSize_t        blockSizes,
                              DataArrayBlockHost Udata_h) :
    pmesh(pmesh), 
    params(params),
    gvParams(gvParams),
    fm(fm),
    blockSizes(blockSizes),
    Udata_h(Udata_h) {};

  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
                    HydroParams params,
                    ConfigMap configMap,
                    id2index_t fm,
                    blockSize_t    blockSizes,
                    DataArrayBlockHost Udata_h) 
  {
    // gresho vortex specific parameters
    GreshoVortexParams gvParams = GreshoVortexParams(configMap);

    // data init functor
    InitGreshoVortexDataFunctor functor(pmesh, params, gvParams, fm, blockSizes, Udata_h);

    // kokkos execution policy
    uint32_t nbTeams_ = configMap.getInteger("init","nbTeams",16);
    functor.setNbTeams ( nbTeams_  );
    
    team_policy_t policy (nbTeams_,
                          Kokkos::AUTO() /* team size chosen by kokkos */);
    
    Kokkos::parallel_for("dyablo::muscl_block::InitGreshoVortexDataFunctor",
                         policy, functor);
  } // apply

  // ================================================
  // ================================================
  //KOKKOS_INLINE_FUNCTION
  inline
  void operator()(thread_t member) const
  {

    uint32_t iOct = member.league_rank();

    const int& bx = blockSizes[IX];
    const int& by = blockSizes[IY];
    const int& bz = blockSizes[IZ];

    uint32_t nbCells = params.dimType == TWO_D ? bx*by : bx*by*bz;

    while (iOct <  pmesh->getNumOctants() )
    {

      // get octant size
      const real_t octSize = pmesh->getSize(iOct);

      // the following assumes bx=by=bz
      // TODO : allow non-cubic blocks
      const real_t cellSize = octSize/bx;

      // coordinates of the lower left corner
      const real_t x0 = pmesh->getNode(iOct, 0)[IX];
      const real_t y0 = pmesh->getNode(iOct, 0)[IY];
      //const real_t z0 = pmesh->getNode(iOct, 0)[IZ];

      // gresho vortex parameters
      const real_t gamma0  = params.settings.gamma0;
      const real_t rho0  = gvParams.rho0;
      const real_t Ma    = gvParams.Ma;
      const real_t p0    = rho0 / (gamma0 * Ma * Ma);

      Kokkos::parallel_for(
          Kokkos::TeamVectorRange(member, nbCells),
          KOKKOS_LAMBDA(const int32_t index) {
            // convert index into ix,iy,iz of local cell inside
            // block
            coord_t iCoord;
            uint32_t &ix = iCoord[IX];
            uint32_t &iy = iCoord[IY];
            //uint32_t &iz = iCoord[IZ];

            if (params.dimType == TWO_D) {
              iCoord = index_to_coord<2>(index, blockSizes);
            } else {
              iCoord = index_to_coord<3>(index, blockSizes);
            }

            // compute x,y,z for that cell (cell center)
            real_t x = x0 + ix * cellSize + cellSize / 2;
            real_t y = y0 + iy * cellSize + cellSize / 2;
            // real_t z = z0 + iz*cellSize + cellSize/2;

            // fluid specific heat ratio
            const real_t gamma0 = params.settings.gamma0;

            x -= 0.5;
            y -= 0.5;

            real_t r = sqrt(x * x + y * y);
            real_t theta = atan2(y, x);

            // polar coordinate
            real_t cosT = cos(theta);
            real_t sinT = sin(theta);

            real_t uphi, p;

            if (r < 0.2) {

              uphi = 5 * r;
              p = p0 + 25 / 2.0 * r * r;

            } else if (r < 0.4) {

              uphi = 2 - 5 * r;
              p = p0 + 25 / 2.0 * r * r + 4 * (1 - 5 * r - log(0.2) + log(r));

            } else {

              uphi = 0;
              p = p0 - 2 + 4 * log(2.0);
            }

            Udata_h(index, fm[ID], iOct) = rho0;
            Udata_h(index, fm[IU], iOct) = rho0 * (-sinT * uphi);
            Udata_h(index, fm[IV], iOct) = rho0 * (cosT * uphi);
            Udata_h(index, fm[IP], iOct) =
                p / (gamma0 - 1.0) + 0.5 *
              (Udata_h(index, fm[IU], iOct) * Udata_h(index, fm[IU], iOct) +
               Udata_h(index, fm[IV], iOct) * Udata_h(index, fm[IV], iOct)) /
              Udata_h(index, fm[ID], iOct);
            ;

            if (params.dimType == THREE_D)
              Udata_h(index, fm[IW], iOct) = 0.0;
          }); // end teamVectorRange

      iOct += nbTeams;
    
    } // end while

  } // end operator ()

  //! AMR mesh
  std::shared_ptr<AMRmesh> pmesh;

  //! general parameters
  HydroParams params;

  //! Gresho vortex problem specific parameters
  GreshoVortexParams gvParams;
  
  //! field manager
  id2index_t fm;

  //! block sizes
  blockSize_t    blockSizes;
  
  //! heavy data
  DataArrayBlockHost Udata_h;

}; // InitGreshoVortexDataFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve gresho vortex problem.
 *
 * This functor only performs mesh refinement, no user data init.
 *
 * Initial conditions is refined near initial density gradients.
 *
 * \sa InitGreshoVortexDataFunctor
 *
 */
class InitGreshoVortexRefineFunctor {

public:
  using range_policy_t = Kokkos::RangePolicy<Kokkos::OpenMP>;

  InitGreshoVortexRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
                                HydroParams params,
                                GreshoVortexParams gvParams,
                                int level_refine) :
    pmesh(pmesh), 
    params(params), 
    gvParams(gvParams),
    level_refine(level_refine)
  {};

  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
                    ConfigMap   configMap,
                    HydroParams params,
                    int         level_refine)
  {

    GreshoVortexParams gvParams = GreshoVortexParams(configMap);

    // iterate functor for refinement
    InitGreshoVortexRefineFunctor functor(pmesh,
                                          params,
                                          gvParams,
                                          level_refine);

    range_policy_t policy(Kokkos::OpenMP(), 0, pmesh->getNumOctants());

    Kokkos::parallel_for("dyablo::muscl::InitGreshoVortexRefineFunctor",
                         policy, functor);
  }

  //KOKKOS_INLINE_FUNCTION
  inline
  void operator()(const size_t &iOct) const
  {

    // get cell level
    uint8_t level = pmesh->getLevel(iOct);

    // only look at level - 1
    if (level == level_refine) {

      // get cell center coordinate in the unit domain
      // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
      std::array<double, 3> center = pmesh->getCenter(iOct);

      real_t x = center[0];
      real_t y = center[1];
      real_t z = center[2];

      // retrieve coordinates to domain center
      x -= 0.5;
      y -= 0.5;
      z -= 0.5;

      double cellSize2 = pmesh->getSize(iOct) * 0.75;

      bool should_refine = false;

      // refine near r=0.2
      real_t radius = 0.2;

      real_t d2 = x*x + y*y;

      if (params.dimType == THREE_D)
        d2 += z*z;

      real_t d = sqrt(d2);

      if ( fabs(d - radius) < cellSize2 )
        should_refine = true;
      
      if ( d > 0.15 and d < 0.25 )
        should_refine = true;

      if (should_refine)
        pmesh->setMarker(iOct, 1);

    } // end if level == level_refine

  } // end operator ()

  //! AMR mesh
  std::shared_ptr<AMRmesh> pmesh;
  
  //! general parameters
  HydroParams params;
  
  //! Gresho vortex problem specific parameters
  GreshoVortexParams gvParams;
  
  //! which level should we look at
  int level_refine;

}; // InitGreshoVortexRefineFunctor

} // namespace muscl_block

} // namespace dyablo

#endif // MUSCL_BLOCK_HYDRO_INIT_GRESHO_VORTEX_H_
