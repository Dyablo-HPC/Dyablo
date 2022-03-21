/**
 * \file ComputeDtHydroFunctor.h
 * \author Pierre Kestener
 */
#ifndef COMPUTE_DT_HYDRO_MUSCL_BLOCK_FUNCTOR_H_
#define COMPUTE_DT_HYDRO_MUSCL_BLOCK_FUNCTOR_H_

#include <limits> // for std::numeric_limits

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "HydroState.h"

// AMR utils
#include "amr/LightOctree.h"

// hydro utils
#include "utils_hydro.h"

#include "utils_block.h"

#ifdef __CUDA_ARCH__
#include "math_constants.h"
#endif

namespace dyablo { 


/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Simplest CFL computational functor.
 * All cell, whatever level, contribute equally to the CFL condition.
 *
 */
class ComputeDtHydroFunctor {

public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  struct Params{
    Params( ConfigMap& configMap )
    :
      xmin( configMap.getValue<real_t>("mesh", "xmin", 0.0) ),
      ymin( configMap.getValue<real_t>("mesh", "ymin", 0.0) ),
      zmin( configMap.getValue<real_t>("mesh", "zmin", 0.0) ),
      xmax( configMap.getValue<real_t>("mesh", "xmax", 1.0) ),
      ymax( configMap.getValue<real_t>("mesh", "ymax", 1.0) ),
      zmax( configMap.getValue<real_t>("mesh", "zmax", 1.0) ),
      gamma0( configMap.getValue<real_t>("hydro","gamma0", 1.4) ),
      smallr( configMap.getValue<real_t>("hydro","smallr", 1e-10) ),
      smallc( configMap.getValue<real_t>("hydro","smallc", 1e-10) ),
      smallp( smallc*smallc/gamma0 ),
      rsst_enabled( configMap.getValue<bool>("low_mach", "rsst_enabled", false) ),
      rsst_cfl_enabled( configMap.getValue<bool>("low_mach", "rsst_cfl_enabled", false) ),
      rsst_ksi( configMap.getValue<real_t>("low_mach", "rsst_ksi", 10.0) )
    {}

    real_t xmin, ymin, zmin;
    real_t xmax, ymax, zmax;
    real_t gamma0, smallr, smallc, smallp;
    bool rsst_enabled, rsst_cfl_enabled;
    real_t rsst_ksi;
  };

  ComputeDtHydroFunctor(LightOctree lmesh,
			Params    params,
			id2index_t     fm,
                        blockSize_t    blockSizes,
			DataArrayBlock Udata) :
    lmesh(lmesh), 
    params(params),
    fm(fm), blockSizes(blockSizes),
    Udata(Udata)
  {}
  
  // static method which does it all: create and execute functor
  static void apply(LightOctree lmesh,
                    Params    params,
		    id2index_t     fm,
                    blockSize_t    blockSizes,
                    DataArrayBlock Udata,
		    double        &invDt)
  {
    
    ComputeDtHydroFunctor functor(lmesh, params, fm, blockSizes, Udata);

    team_policy_t policy ( lmesh.getNumOctants(),
                          Kokkos::AUTO() /* team size chosen by kokkos */);

    Kokkos::parallel_reduce("dyablo::ComputeDtHydroFunctor",
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

    const int& bx = blockSizes[IX];
    const int& by = blockSizes[IY];

    const real_t Lx = params.xmax - params.xmin;
    const real_t Ly = params.ymax - params.ymin;

    // number of cells per octant
    uint32_t nbCells = bx*by;

    // initialize reduction variable
    real_t invDt_local = invDt;

    // 2D version
    HydroState2d uLoc; // conservative variables in current cell
    HydroState2d qLoc; // primitive    variables in current cell
    real_t c = 0.0;
    real_t vx, vy;
      
    // retrieve cell size from mesh
    real_t dx = lmesh.getSize({iOct,false}) * Lx / bx;
    real_t dy = lmesh.getSize({iOct,false}) * Ly / by;

    // initialialize cell id
    uint32_t iCell = member.team_rank();
    while (iCell < nbCells) {
      
      // get local conservative variable
      uLoc[ID] = Udata(iCell,fm[ID],iOct);
      uLoc[IP] = Udata(iCell,fm[IP],iOct);
      uLoc[IU] = Udata(iCell,fm[IU],iOct);
      uLoc[IV] = Udata(iCell,fm[IV],iOct);
      
      // get primitive variables in current cell
      computePrimitives(uLoc, &c, qLoc, params.gamma0, params.smallr, params.smallp);

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

    const int& bx = blockSizes[IX];
    const int& by = blockSizes[IY];
    const int& bz = blockSizes[IZ];

    const real_t Lx = params.xmax - params.xmin;
    const real_t Ly = params.ymax - params.ymin;
    const real_t Lz = params.zmax - params.zmin;

    uint32_t nbCells = bx*by*bz;

    real_t invDt_local = invDt;
    
    // 3D version
    HydroState3d uLoc; // conservative variables in current cell
    HydroState3d qLoc; // primitive    variables in current cell
    real_t c = 0.0;
    real_t vx, vy, vz;

    // retrieve cell size from mesh
    real_t dx = lmesh.getSize({iOct,false}) * Lx / blockSizes[IX];
    real_t dy = lmesh.getSize({iOct,false}) * Ly / blockSizes[IY];
    real_t dz = lmesh.getSize({iOct,false}) * Lz / blockSizes[IZ];

    uint32_t iCell = member.team_rank();
    while (iCell < nbCells) {
  
      // get local conservative variable
      uLoc[ID] = Udata(iCell,fm[ID],iOct);
      uLoc[IP] = Udata(iCell,fm[IP],iOct);
      uLoc[IU] = Udata(iCell,fm[IU],iOct);
      uLoc[IV] = Udata(iCell,fm[IV],iOct);
      uLoc[IW] = Udata(iCell,fm[IW],iOct);
      
      // get primitive variables in current cell
      computePrimitives(uLoc, &c, qLoc, params.gamma0, params.smallr, params.smallp);

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


    // update global reduced value
    if (invDt < invDt_local)
      invDt = invDt_local;

  } // operator_3d


  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void operator()(const thread_t& member, real_t &invDt) const
  {

    int ndims = lmesh.getNdim();
    if (ndims == 2)
      operator_2d(member,invDt);

    if (ndims == 3)
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
  LightOctree lmesh;
  
  //! general parameters
  Params  params;

  //! field manager
  id2index_t   fm;

  //! block sizes
  blockSize_t blockSizes;

  //! heavy data - conservative variables
  DataArrayBlock Udata;
  
}; // ComputeDtHydroFunctor



} // namespace dyablo

#endif // COMPUTE_DT_HYDRO_MUSCL_BLOCK_FUNCTOR_H_