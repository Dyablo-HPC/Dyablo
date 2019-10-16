/**
 * \file ComputeDtHydroFunctor.h
 * \author Pierre Kestener
 */
#ifndef COMPUTE_DT_HYDRO_MUSCL_BLOCK_FUNCTOR_H_
#define COMPUTE_DT_HYDRO_MUSCL_BLOCK_FUNCTOR_H_

#include <limits> // for std::numeric_limits

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"
#include "shared/problems/initRiemannConfig2d.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

// hydro utils
#include "shared/utils_hydro.h"

#include "utils_block.h"

namespace dyablo { namespace muscl_block {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Simplest CFL computational functor.
 * All cell, whatever level, contribute equally to the CFL condition.
 *
 */
class ComputeDtHydroFunctor {

private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) {nbTeams = nbTeams_;}; 

  ComputeDtHydroFunctor(std::shared_ptr<AMRmesh> pmesh,
			HydroParams    params,
			id2index_t     fm,
                        blockSize_t    blockSizes,
			DataArrayBlock Udata) :
    pmesh(pmesh), params(params),
    fm(fm), blockSizes(blockSizes),
    Udata(Udata)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    ConfigMap      configMap,
                    HydroParams    params,
		    id2index_t     fm,
                    blockSize_t    blockSizes,
                    DataArrayBlock Udata,
		    double        &invDt)
  {
    
    ComputeDtHydroFunctor functor(pmesh, params, fm, blockSizes, Udata);

    // kokkos execution policy
    uint32_t nbTeams_ = configMap.getInteger("amr","nbTeams",16);
    functor.setNbTeams ( nbTeams_ );

    team_policy_t policy (nbTeams_,
                          Kokkos::AUTO() /* team size chosen by kokkos */);

    Kokkos::parallel_reduce("dyablo::muscl_block::ComputeDtHydroFunctor",
                            policy, functor, invDt);
  } // apply

  // ====================================================================
  // ====================================================================
  // Tell each thread how to initialize its reduction result.
  KOKKOS_INLINE_FUNCTION
  void init (real_t& dst) const
  {
    // The identity under max is -Inf.
    // Kokkos does not come with a portable way to access
    // floating-point Inf and NaN. 
#ifdef __CUDA_ARCH__
    dst = -CUDART_INF;
#else
    dst = std::numeric_limits<real_t>::min();
#endif // __CUDA_ARCH__
  } // init

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void operator_2d(const thread_t& member, real_t &invDt) const
  {
    uint32_t iOct = member.league_rank();
    //uint32_t iCell = member.team_rank();

    // total number of octants (of current MPI processor)
    uint32_t nbOct = pmesh->getNumOctants();

    const int& bx = blockSizes[IX];
    const int& by = blockSizes[IY];
    const int& bz = blockSizes[IZ];

    // number of cells per octant
    uint32_t nbCells = params.dimType == TWO_D ? bx*by : bx*by*bz;

    // initialize reduction variable
    real_t invDt_local = invDt;

    // 2D version
    HydroState2d uLoc; // conservative variables in current cell
    HydroState2d qLoc; // primitive    variables in current cell
    real_t c = 0.0;
    real_t vx, vy;

    while (iOct < nbOct) {

      // get cell level
      uint8_t level = pmesh->getLevel(iOct);
      
      // retrieve cell size from mesh
      real_t dx = pmesh->levelToSize(level) / blockSizes[IX];
      real_t dy = pmesh->levelToSize(level) / blockSizes[IY];

      // initialialize cell id
      uint32_t iCell = member.team_rank();
      while (iCell < nbCells) {
        
        // get local conservative variable
        uLoc[ID] = Udata(iCell,fm[ID],iOct);
        uLoc[IP] = Udata(iCell,fm[IP],iOct);
        uLoc[IU] = Udata(iCell,fm[IU],iOct);
        uLoc[IV] = Udata(iCell,fm[IV],iOct);
        
        // get primitive variables in current cell
        computePrimitives(uLoc, &c, qLoc, params);

        if (params.rsst_enabled and params.rsst_cfl_enabled) {
          vx = c/params.rsst_ksi + FABS(qLoc[IU]);
          vy = c/params.rsst_ksi + FABS(qLoc[IV]);
        } else {
          vx = c + FABS(qLoc[IU]);
          vy = c + FABS(qLoc[IV]);
        }

        invDt_local = FMAX(invDt_local, vx / dx + vy / dy);

        iCell += member.team_size();
      
      } // end while iCell

      iOct += nbTeams;

    } // end while iOct

    // update global reduced value
    if (invDt < invDt_local)
      invDt = invDt_local;

  } // operator_2d

  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void operator_3d(const thread_t& member, real_t &invDt) const
  {
    uint32_t iOct = member.league_rank();
    uint32_t iCell = member.team_rank();

    uint32_t nbOct = pmesh->getNumOctants();

    const int& bx = blockSizes[IX];
    const int& by = blockSizes[IY];
    const int& bz = blockSizes[IZ];

    uint32_t nbCells = params.dimType == TWO_D ? bx*by : bx*by*bz;

    real_t invDt_local = invDt;
    
    // 3D version
    HydroState3d uLoc; // conservative variables in current cell
    HydroState3d qLoc; // primitive    variables in current cell
    real_t c = 0.0;
    real_t vx, vy, vz;

    while (iOct < nbOct) {

      // get cell level
      uint8_t level = pmesh->getLevel(iOct);
      
      // retrieve cell size from mesh
      real_t dx = pmesh->levelToSize(level) / blockSizes[IX];
      real_t dy = pmesh->levelToSize(level) / blockSizes[IY];
      real_t dz = pmesh->levelToSize(level) / blockSizes[IZ];

      while (iCell < nbCells) {
    
        // get local conservative variable
        uLoc[ID] = Udata(iCell,fm[ID],iOct);
        uLoc[IP] = Udata(iCell,fm[IP],iOct);
        uLoc[IU] = Udata(iCell,fm[IU],iOct);
        uLoc[IV] = Udata(iCell,fm[IV],iOct);
        uLoc[IW] = Udata(iCell,fm[IW],iOct);
        
        // get primitive variables in current cell
        computePrimitives(uLoc, &c, qLoc, params);

        if (params.rsst_enabled and params.rsst_cfl_enabled) {
          vx = c/params.rsst_ksi + FABS(qLoc[IU]);
          vy = c/params.rsst_ksi + FABS(qLoc[IV]);
          vz = c/params.rsst_ksi + FABS(qLoc[IW]);
        } else {
          vx = c + FABS(qLoc[IU]);
          vy = c + FABS(qLoc[IV]);
          vz = c + FABS(qLoc[IW]);
        }

        invDt_local = FMAX(invDt_local, vx / dx + vy / dy + vz / dz);

        iCell += member.team_size();
      
      } // end while iCell

      iOct += nbTeams;

    } // end while iOct

    // update global reduced value
    if (invDt < invDt_local)
      invDt = invDt_local;

  } // operator_3d


  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void operator()(const thread_t& member, real_t &invDt) const
  {

    if (params.dimType == TWO_D)
      operator_2d(member,invDt);

    if (params.dimType == THREE_D)
      operator_3d(member,invDt);
    
  }
  
  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. Note that both input
  // arguments MUST be declared volatile.
  KOKKOS_INLINE_FUNCTION
  void join (volatile real_t& dst,
	     const volatile real_t& src) const
  {
    // max reduce
    if (dst < src) {
      dst = src;
    }
  } // join

  //! AMR mesh
  std::shared_ptr<AMRmesh> pmesh;
  
  //! general parameters
  HydroParams  params;

  //! field manager
  id2index_t   fm;

  //! block sizes
  blockSize_t blockSizes;

  //! heavy data - conservative variables
  DataArrayBlock Udata;
  
}; // ComputeDtHydroFunctor

} // namespace muscl_block

} // namespace dyablo

#endif // COMPUTE_DT_HYDRO_MUSCL_BLOCK_FUNCTOR_H_
