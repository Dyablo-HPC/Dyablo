/**
 * \file InitImplode.h
 * \author Pierre Kestener
 */
#ifndef MUSCL_BLOCK_HYDRO_INIT_IMPLODE_H_
#define MUSCL_BLOCK_HYDRO_INIT_IMPLODE_H_

#include <limits> // for std::numeric_limits

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"
#include "shared/problems/ImplodeParams.h"

#include "muscl_block/utils_block.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

namespace dyablo { namespace muscl_block {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve implode problem.
 *
 * This functor takes as input a mesh, already refined, and initializes
 * user data on host. Copying data from host to device, should be 
 * done outside.
 * 
 *
 * Initial conditions is refined near strong density gradients.
 *
 * Using Kokkos Team policy, one thread team per block (one block per octree leaf).
 *
 */
class InitImplodeDataFunctor {

private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::OpenMP, Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) {nbTeams = nbTeams_;}; 

  InitImplodeDataFunctor(std::shared_ptr<AMRmesh> pmesh,
                         HydroParams    params,
                         ImplodeParams  iParams,
                         id2index_t     fm,
                         blockSize_t    blockSizes,
                         DataArrayBlockHost Udata_h) :
    pmesh(pmesh), params(params), iParams(iParams),
    fm(fm), blockSizes(blockSizes), Udata_h(Udata_h)
  {
  };
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams    params,
                    ConfigMap      configMap,
		    id2index_t     fm,
                    blockSize_t    blockSizes,
                    DataArrayBlockHost Udata_h)
  {
    ImplodeParams implodeParams = ImplodeParams(configMap);
    
    // data init functor
    InitImplodeDataFunctor functor(pmesh, params, implodeParams, fm, blockSizes, Udata_h);

    // kokkos execution policy
    uint32_t nbTeams_ = configMap.getInteger("init","nbTeams",16);
    functor.setNbTeams ( nbTeams_  );

    // perform initialization on host
    team_policy_t policy (Kokkos::OpenMP(),
                          nbTeams_,
                          Kokkos::AUTO() /* team size chosen by kokkos */);

    Kokkos::parallel_for("dyablo::muscl_block::InitImplodeDataFunctor",
                         policy, functor);
  }
  
  inline 
  static HydroState3d value( const HydroParams& params, const ImplodeParams& iParams, real_t x, real_t y, real_t z)
  {
    const real_t xmin = params.xmin;
    const real_t xmax = params.xmax;
    const real_t ymin = params.ymin;
    const real_t zmin = params.zmin;

    // implode problem parameters
    // outer parameters
    const real_t rho_out = iParams.rho_out;
    const real_t p_out = iParams.p_out;
    const real_t u_out = iParams.u_out;
    const real_t v_out = iParams.v_out;
    const real_t w_out = iParams.w_out;

    // inner parameters
    const real_t rho_in = iParams.rho_in;
    const real_t p_in = iParams.p_in;
    const real_t u_in = iParams.u_in;
    const real_t v_in = iParams.v_in;
    const real_t w_in = iParams.w_in;

    const int shape = iParams.shape;

    const bool debug = iParams.debug;

    const real_t gamma0 = params.settings.gamma0;

    HydroState3d res;

    // initialize
    bool tmp;
    if (shape == 1) {
      if (params.dimType == TWO_D)
        tmp = x+y*y > 0.5 and x+y*y < 1.5;
      else
        tmp = x+y+z > 0.5 and x+y+z < 2.5;
    } else {
      if (params.dimType == TWO_D)
        tmp = x+y > (xmin+xmax)/2. + ymin;
      else
        tmp = x+y+z > (xmin+xmax)/2. + ymin + zmin;
    }

    if (tmp) {
      res[ID] = rho_out;
      res[IP] = p_out/(gamma0-1.0) +
        0.5 * rho_out * (u_out*u_out + v_out*v_out);
      res[IU] = u_out;
      res[IV] = v_out;
    } else {
      res[ID] = rho_in;
      res[IP] = p_in/(gamma0-1.0) +
        0.5 * rho_in * (u_in*u_in + v_in*v_in);
      res[IU] = u_in;
      res[IV] = v_in;
    }

    if (params.dimType == THREE_D) {
      if (tmp) {
        res[IW] = w_out;
        res[IP] = p_out/(gamma0-1.0) +
          0.5 * rho_out * (u_out*u_out + v_out*v_out + w_out*w_out);
      } else {
        res[IW] = w_in;
        res[IP] = p_in/(gamma0-1.0) +
          0.5 * rho_in * (u_in*u_in + v_in*v_in + w_in*w_in);
      }
    }

    if (debug) {
      // just add a gradient to density
      real_t delta = params.dimType == TWO_D ?
        (x+y)*0.05 :
        (x+y+z)*0.05;
      res[ID] += delta;
      res[IP] += 2*delta;    
    }

    return res;
  }

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
      const real_t x0 = pmesh->getCoordinates(iOct)[IX];
      const real_t y0 = pmesh->getCoordinates(iOct)[IY];
      const real_t z0 = pmesh->getCoordinates(iOct)[IZ];

      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCells),
        [=](const int32_t index) {

          // convert index into ix,iy,iz of local cell inside
          // block
          coord_t iCoord;
          uint32_t& ix = iCoord[IX];
          uint32_t& iy = iCoord[IY];
          uint32_t& iz = iCoord[IZ];                    

          if (params.dimType == TWO_D) {
            iCoord = index_to_coord<2>(index,blockSizes);
          } else {
            iCoord = index_to_coord<3>(index,blockSizes);
          }

          // compute x,y,z for that cell (cell center)
          real_t x = x0 + ix*cellSize + cellSize/2;
          real_t y = y0 + iy*cellSize + cellSize/2;
          real_t z = 0;          
          if( params.dimType == THREE_D) 
            z = z0 + iz*cellSize + cellSize/2;

          HydroState3d val = value(params, iParams, x,y,z);
          
          Udata_h(index, fm[ID], iOct) = val[ID];
          Udata_h(index, fm[IP], iOct) = val[IP];
          Udata_h(index, fm[IU], iOct) = val[IU];
          Udata_h(index, fm[IV], iOct) = val[IV];
          if (params.dimType == THREE_D) {
              Udata_h(index, fm[IW], iOct) = val[IW];
          }
        }); // end TeamVectorRange

      iOct += nbTeams;

    } // end while

  } // end operator ()

  //! AMR mesh
  std::shared_ptr<AMRmesh> pmesh;

  //! general parameters
  HydroParams    params;

  //! Implode problem specific parameters
  ImplodeParams    iParams;

  //! field manager
  id2index_t     fm;

  //! block sizes
  blockSize_t    blockSizes;

  //! heavy data on host
  DataArrayBlockHost Udata_h;

}; // InitImplodeDataFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve implode problem.
 *
 * This functor only performs mesh refinement based only on octant
 * location, no user data init.
 * In an octant is near the discontinuity, we apply refine operator.
 *
 * Warning: it is only usefull for testing/debug; in a real
 * run, initial refinement is done through a genuine refinement
 * criterium (e.g. gradient based).
 * 
 * Initial conditions is refined near initial density gradients.
 * This functor is only valid when shape is 0. 
 *
 * \sa InitImplodeDataFunctor
 *
 */
class InitImplodeRefineFunctor {
  
public:
  using range_policy_t = Kokkos::RangePolicy<Kokkos::OpenMP>;

  InitImplodeRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
                           HydroParams   params,
                           ImplodeParams iParams,
                           int           level_refine) :
    pmesh(pmesh), 
    params(params),
    iParams(iParams),
    level_refine(level_refine)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
                    ConfigMap     configMap,
		    HydroParams   params,
		    int           level_refine)
  {
    ImplodeParams implodeParams = ImplodeParams(configMap);

    // iterate functor for refinement
    InitImplodeRefineFunctor functor(pmesh,
                                     params,
                                     implodeParams,
                                     level_refine);

    range_policy_t policy(Kokkos::OpenMP(), 0, pmesh->getNumOctants());

    Kokkos::parallel_for("dyablo::muscl_block::InitImplodeRefineFunctor",
                         policy, functor);
    
  } // apply
  
  //KOKKOS_INLINE_FUNCTION
  inline
  void operator()(const size_t& iOct) const
  {

    // implode problem parameters
    const real_t xmin = params.xmin;
    const real_t xmax = params.xmax;
    const real_t ymin = params.ymin;
    const real_t zmin = params.zmin;

    //constexpr double eps = 0.005;
    
    // get cell level
    uint8_t level = pmesh->getLevel(iOct);
    
    // only look at level - 1
    if (level == level_refine) {

      // get cell center coordinate in the unit domain
      // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
      std::array<double,3> center = pmesh->getCenter(iOct);
      
      const real_t x = center[0];
      const real_t y = center[1];
      const real_t z = center[2];

      const double cellSize  = pmesh->getSize(iOct);
      //const double cellSize2 = pmesh->getSize(iOct)*0.75;
      
      bool should_refine = false;

      const int shape = iParams.shape;

      if (shape == 0) {

        // compute distance to discontinuity
        real_t d = params.dimType == TWO_D ?
          fabs(x + y - ( 0.5*(xmin+xmax) + ymin ) ) :
          fabs(x + y + z - ( 0.5*(xmin+xmax) + ymin + zmin ) ) ;

        if ( d < 1.5*cellSize )
          should_refine = true;

        if (should_refine)
          pmesh->setMarker(iOct, 1);

      }

    } // end if level == level_refine
    
  } // end operator ()

  //! AMR mesh
  std::shared_ptr<AMRmesh> pmesh;

  //! general parameters
  HydroParams    params;

  //! Implode problem specific parameters
  ImplodeParams    iParams;

  //! block sizes
  //blockSize_t  blockSizes;

  //! which level should we look at
  int            level_refine;
  
}; // InitImplodeRefineFunctor

} // namespace muscl_block

} // namespace dyablo

#endif // MUSCL_BLOCK_HYDRO_INIT_IMPLODE_H_
