/**
 * \file InitBlast.h
 * \author Pierre Kestener
 */
#ifndef HYDRO_INIT_BLAST_H_
#define HYDRO_INIT_BLAST_H_

#include <limits> // for std::numeric_limits

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"
#include "shared/problems/BlastParams.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

namespace dyablo { namespace muscl {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve blast problem.
 *
 * This functor takes as input a mesh, already refined, and initializes
 * user data.
 * 
 *
 * Initial conditions is refined near strong density gradients.
 *
 */
class InitBlastDataFunctor {

public:
  InitBlastDataFunctor(std::shared_ptr<AMRmesh> pmesh,
                       HydroParams   params,
                       BlastParams   bParams,
                       id2index_t    fm,
                       DataArrayHost Udata) :
    pmesh(pmesh), params(params), bParams(bParams),
    fm(fm), Udata(Udata)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    HydroParams   params,
                    ConfigMap     configMap,
		    id2index_t    fm,
                    DataArrayHost     Udata)
  {
    BlastParams blastParams = BlastParams(configMap);
    
    // data init functor
    InitBlastDataFunctor functor(pmesh, params, blastParams, fm, Udata);

    Kokkos::parallel_for("dyablo::muscl::InitBlastDataFunctor", Kokkos::RangePolicy<Kokkos::OpenMP>(0, pmesh->getNumOctants()), functor);
  }
  
  void operator()(const size_t& i) const
  {
    
    // blast problem parameters
    const real_t blast_radius      = bParams.blast_radius;
    const real_t radius2           = blast_radius*blast_radius;
    const real_t blast_center_x    = bParams.blast_center_x;
    const real_t blast_center_y    = bParams.blast_center_y;
    const real_t blast_center_z    = bParams.blast_center_z;
    const real_t blast_density_in  = bParams.blast_density_in;
    const real_t blast_density_out = bParams.blast_density_out;
    const real_t blast_pressure_in = bParams.blast_pressure_in;
    const real_t blast_pressure_out= bParams.blast_pressure_out;
    
    const real_t gamma0            = params.settings.gamma0;

    // get cell center coordinate in the unit domain
    // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
    std::array<double,3> center = pmesh->getCenter(i);

    real_t x = center[0];
    real_t y = center[1];
    real_t z = center[2];

    // Quadrant size
    const real_t qx = 1.0 / bParams.blast_nx;
    const real_t qy = 1.0 / bParams.blast_ny;
    const real_t qz = 1.0 / bParams.blast_nz;
    
    const int qix = (int)(x / qx);
    const int qiy = (int)(y / qy);
    const int qiz = (int)(z / qz);

    // Rescaling position wrt the current blast quadrant
    x = (x - qix * qx) / qx;
    y = (y - qiy * qy) / qy;
    z = (z - qiz * qz) / qz;
    
    real_t d2 = 
      (x-blast_center_x)*(x-blast_center_x)+
      (y-blast_center_y)*(y-blast_center_y);  

    if (params.dimType == THREE_D)
      d2 += (z-blast_center_z)*(z-blast_center_z);

    if (d2 < radius2) {
      Udata(i    , fm[ID]) = blast_density_in; 
      Udata(i    , fm[IP]) = blast_pressure_in/(gamma0-1.0); 
      Udata(i    , fm[IU]) = 0.0;
      Udata(i    , fm[IV]) = 0.0;
    } else {
      Udata(i    , fm[ID]) = blast_density_out; 
      Udata(i    , fm[IP]) = blast_pressure_out/(gamma0-1.0); 
      Udata(i    , fm[IU]) = 0.0;
      Udata(i    , fm[IV]) = 0.0;
    }

    if (params.dimType == THREE_D)
      Udata(i, fm[IW]) = 0.0;
    
  } // end operator ()

  std::shared_ptr<AMRmesh> pmesh;
  HydroParams  params;
  BlastParams  bParams;
  id2index_t   fm;
  DataArrayHost    Udata;
  
}; // InitBlastDataFunctor

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Implement initialization functor to solve blast problem.
 *
 * This functor only performs mesh refinement, no user data init.
 * 
 * Initial conditions is refined near initial density gradients.
 *
 * \sa InitBlastDataFunctor
 *
 */
class InitBlastRefineFunctor {
  
public:
  InitBlastRefineFunctor(std::shared_ptr<AMRmesh> pmesh,
                         HydroParams  params,
                         BlastParams bParams,
                         int         level_refine) :
    pmesh(pmesh), params(params), bParams(bParams),
    level_refine(level_refine)
  {};
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
                    ConfigMap     configMap,
		    HydroParams   params,
		    int           level_refine)
  {
    BlastParams blastParams = BlastParams(configMap);

    // iterate functor for refinement
    InitBlastRefineFunctor functor(pmesh, params, blastParams, 
                                   level_refine);
    Kokkos::parallel_for("dyablo::muscl::InitBlastRefineFunctor", Kokkos::RangePolicy<Kokkos::OpenMP>(0, pmesh->getNumOctants()), functor);
    
  }
  
  void operator()(const size_t& i) const
  {

    // blast problem parameters
    const real_t radius            = bParams.blast_radius;
    //const real_t radius2           = radius*radius;
    const real_t blast_center_x    = bParams.blast_center_x;
    const real_t blast_center_y    = bParams.blast_center_y;
    const real_t blast_center_z    = bParams.blast_center_z;

    //constexpr double eps = 0.005;
    
    // get cell level
    uint8_t level = pmesh->getLevel(i);
    
    // only look at level - 1
    if (level == level_refine) {

      // get cell center coordinate in the unit domain
      // FIXME : need to refactor AMRmesh interface to use Kokkos::Array
      std::array<double,3> center = pmesh->getCenter(i);
      
      real_t x = center[0];
      real_t y = center[1];
      real_t z = center[2];

      // Quadrant size
      const real_t qx = 1.0 / bParams.blast_nx;
      const real_t qy = 1.0 / bParams.blast_ny;
      const real_t qz = 1.0 / bParams.blast_nz;
      
      const int qix = (int)(x / qx);
      const int qiy = (int)(y / qy);
      const int qiz = (int)(z / qz);

      // Rescaling position wrt the current blast quadrant
      x = (x - qix * qx) / qx;
      y = (y - qiy * qy) / qy;
      z = (z - qiz * qz) / qz;

      // 0.87 > sqrt(3)/2 : distance between center to edge
      double cellSize = pmesh->getSize(i)*0.87;
      
      // We refine if the current size is bigger than a cell
      bool should_refine = (cellSize > std::min({qx, qy, qz}));

      real_t d2 = 
        (x-blast_center_x)*(x-blast_center_x)+
        (y-blast_center_y)*(y-blast_center_y);  

      if (params.dimType == THREE_D)
        d2 += (z-blast_center_z)*(z-blast_center_z);

      if ( fabs(sqrt(d2) - radius) < cellSize )
	      should_refine = true;
      
      if (should_refine)
	      pmesh->setMarker(i, 1);

    } // end if level == level_refine
    
  } // end operator ()

  std::shared_ptr<AMRmesh> pmesh;
  HydroParams    params;
  BlastParams    bParams;
  int            level_refine;
  
}; // InitBlastRefineFunctor

} // namespace muscl

} // namespace dyablo

#endif // HYDRO_INIT_BLAST_H_
