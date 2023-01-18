/**
 * \file ComputeDtHydroFunctor.h
 * \author Pierre Kestener
 */
#ifndef COMPUTE_DT_HYDRO_MUSCL_BLOCK_FUNCTOR_H_
#define COMPUTE_DT_HYDRO_MUSCL_BLOCK_FUNCTOR_H_

#include <limits> // for std::numeric_limits

#include "kokkos_shared.h"
#include "FieldManager.h"
#include "states/State_hydro.h"

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
  template <
    int ndim>
  KOKKOS_INLINE_FUNCTION
  void apply(const thread_t& member, real_t &invDt) const
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
    ConsHydroState uLoc; // conservative variables in current cell
    PrimHydroState qLoc; // primitive    variables in current cell
    real_t c = 0.0;
    real_t vx, vy, vz;

    // retrieve cell size from mesh
    auto oct_size = lmesh.getSize({iOct,false});
    real_t dx = oct_size[IX] * Lx / blockSizes[IX];
    real_t dy = oct_size[IY] * Ly / blockSizes[IY];
    real_t dz = oct_size[IZ] * Lz / blockSizes[IZ];

    uint32_t iCell = member.team_rank();
    while (iCell < nbCells) {
  
      // get local conservative variable
      uLoc.rho   = Udata(iCell, fm[ID], iOct);
      uLoc.e_tot = Udata(iCell, fm[IE], iOct);
      uLoc.rho_u = Udata(iCell, fm[IU], iOct);
      uLoc.rho_v = Udata(iCell, fm[IV], iOct);
      if (ndim == 3)
        uLoc.rho_w = Udata(iCell, fm[IW], iOct);
      
      // get primitive variables in current cell
      computePrimitives(uLoc, &c, qLoc, params.gamma0, params.smallr, params.smallp);

      vx = c + FABS(qLoc.u);
      vy = c + FABS(qLoc.v);
      vz = (ndim == 3 ? c + FABS(qLoc.w) : 0.0);

      invDt_local = FMAX(invDt_local, vx / dx + vy / dy + vz / dz);

      iCell += member.team_size();
    
    } // end while iCell


    // update global reduced value
    if (invDt < invDt_local)
      invDt = invDt_local;

  }


  // ====================================================================
  // ====================================================================
  KOKKOS_INLINE_FUNCTION
  void operator()(const thread_t& member, real_t &invDt) const
  {
    int ndims = lmesh.getNdim();
    if (ndims == 2)
      apply<2>(member, invDt);
    else
      apply<3>(member, invDt);
  }
  
  // "Join" intermediate results from different threads.
  // This should normally implement the same reduction
  // operation as operator() above. 
  KOKKOS_INLINE_FUNCTION
  void join (real_t& dst,
	     const real_t& src) const
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
