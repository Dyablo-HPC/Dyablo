// !!!!!!!!!!!!!!!!!!!!!!!!!!
// UNFINISHED
// !!!!!!!!!!!!!!!!!!!!!!!!!!
/**
 * \file MarkOctantsHydroFunctor.h
 * \author Pierre Kestener
 */
#ifndef DYABLO_MUSCL_BLOCK_MARK_OCTANTS_HYDRO_FUNCTOR_H_
#define DYABLO_MUSCL_BLOCK_MARK_OCTANTS_HYDRO_FUNCTOR_H_

#include "shared/kokkos_shared.h"
#include "shared/FieldManager.h"
#include "shared/HydroState.h"

#include "bitpit_PABLO.hpp"
#include "shared/bitpit_common.h"

// utils hydro
#include "shared/utils_hydro.h"

namespace dyablo { namespace muscl_block {

/*************************************************/
/*************************************************/
/*************************************************/
/**
 * Mark octants for refine or coarsen according to a
 * gradient-like conditions.
 *
 * We adopt here a different strategy from muscl, here
 * in muscl_block we have one block of cells per leaf (i.e. octant).
 * 
 * To mark an octant for refinement, we have adapted ideas 
 * from code AMUN, i.e. for each inner cells, we apply the
 * refinement criterium, then reduce the results to have a
 * single indicator to decide whether or not the octant is
 * flag for refinement.
 *
 * We do not use directly the global array U (of conservative variables
 * in block, without ghost cells) but Qgroup containing a smaller 
 * number of octants, but with ghost cells. This means the computation
 * is organized in a piecewise fashion, i.e. this functor is aimed
 * to be called inside a loop over all sub-groups of octants.
 *
 */
class MarkOctantsHydroFunctor {

private:
  uint32_t nbTeams; //!< number of thread teams

public:
  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  void setNbTeams(uint32_t nbTeams_) {nbTeams = nbTeams_;};

public:
  /**
   * Mark cells for refine/coarsen functor.
   *
   * \param[in] pmesh AMR mesh Pablo data structure
   * \param[in] params
   * \param[in] fm field map to access user data
   * \param[in] Qgroup primitive variables (block data with ghost cells)
   * \param[in] iGroup identifies a group of octants among all subgroup
   * \param[in] epsilon_refine threshold value
   * \param[in] epsilon_coarsen threshold value
   *
   *
   * \todo refactor interface : there is no need to have a PabloUniform (pmesh)
   * object here; it is only used to call setMarker. All we need is to make
   * Pablo exposes the array of refinement flags (as Kokkos::View).
   *
   * \todo the total number of octants (for current MPI process) is retrieve from
   * pmesh object; should be passed as an argument.
   *
   */
  MarkOctantsHydroFunctor(std::shared_ptr<AMRmesh> pmesh,
                          HydroParams    params,
                          id2index_t     fm,
                          blockSize_t    blockSizes,
                          uint32_t       ghostWidth,
                          uint32_t       nbOcts,
                          uint32_t       nbOctsPerGroup,
                          DataArrayBlock Qgroup,
                          uint32_t       iGroup,
                          real_t         error_min,
                          real_t         error_max) :
    pmesh(pmesh),
    params(params),
    fm(fm),
    blockSizes(blockSizes),
    ghostWidth(ghostWidth),
    nbOcts(nbOcts),
    nbOctsPerGroup(nbOctsPerGroup),
    Qgroup(Qgroup), 
    iGroup(iGroup), 
    error_min(error_min),
    error_max(error_max),
    eps(std::numeric_limits<real_t>::epsilon()),
    epsref(0.01)
  {
  
    bx_g = blockSizes[IX] + 2 * (ghostWidth);
    by_g = blockSizes[IY] + 2 * (ghostWidth);
    bz_g = blockSizes[IZ] + 2 * (ghostWidth);

    bx = blockSizes[IX];
    by = blockSizes[IY];
    bz = blockSizes[IZ];

    nbCellsPerBlock = params.dimType == TWO_D ?
      bx * by :
      bx * by * bz;

  };
  
  // static method which does it all: create and execute functor
  static void apply(std::shared_ptr<AMRmesh> pmesh,
		    ConfigMap      configMap,
                    HydroParams    params,
		    id2index_t     fm,
                    blockSize_t    blockSizes,
                    uint32_t       ghostWidth,
                    uint32_t       nbOcts,
                    uint32_t       nbOctsPerGroup,
                    DataArrayBlock Qgroup,
                    uint32_t       iGroup,
                    real_t         error_min,
                    real_t         error_max)
  {
    MarkOctantsHydroFunctor functor(pmesh, params, fm,
                                    blockSizes,
                                    ghostWidth,
                                    nbOcts,
                                    nbOctsPerGroup,
                                    Qgroup, 
                                    iGroup,
                                    error_min, 
                                    error_max);

    // kokkos execution policy
    uint32_t nbTeams_ = configMap.getInteger("amr","nbTeams",16);
    functor.setNbTeams ( nbTeams_ );
    
    team_policy_t policy (nbTeams_,
                          Kokkos::AUTO() /* team size chosen by kokkos */);

    // this is a parallel for loop
    Kokkos::parallel_for("dyablo::muscl_block::MarkOctantsHydroFunctor",
                         policy, 
                         functor);
  } // apply

  // ======================================================
  // ======================================================
  /**
   * compute second derivative of a given primitive variable at a given
   * cell from a block.
   *
   * this is the 2d version.
   *
   * \param[in] ivar integer id (which primitive variable ?)
   * \param[in] i integer x-coordinate in non-ghosted block
   * \param[in] j integer y-coordinate in non-ghosted block
   * \param[in] dir is direction used to compute second derivative (IX,IY,IZ)
   * \param[in] iOct_local octant local id (local to current group)
   *
   */
  KOKKOS_INLINE_FUNCTION
  real_t compute_second_derivative_error(uint8_t  ivar, 
					 uint32_t i,
					 uint32_t j,
					 uint8_t  dir,
					 uint32_t iOct_local) const
  {

    real_t res = 0;

    const uint32_t iCell = i+ghostWidth + bx_g * (j+ghostWidth);

    uint32_t iCellm1, iCellp1;

    if (dir == IX) {
      
      iCellm1 = i-1+ghostWidth + bx_g * (j+ghostWidth);
      iCellp1 = i+1+ghostWidth + bx_g * (j+ghostWidth);

    } else if (dir == IY) {
    
      iCellm1 = i+ghostWidth + bx_g * (j-1+ghostWidth);
      iCellp1 = i+ghostWidth + bx_g * (j+1+ghostWidth);

    }

    const real_t q   = Qgroup(iCell  ,fm[ivar],iOct_local);
    const real_t qm1 = Qgroup(iCellm1,fm[ivar],iOct_local);
    const real_t qp1 = Qgroup(iCellp1,fm[ivar],iOct_local);

    const real_t fr = qp1 - q;    
    const real_t fl = qm1 - q;
    
    const real_t fc = FABS(qp1) + FABS(qm1) + 2 * FABS(q);
    res = FABS(fr + fl) / (FABS(fr) + FABS(fl) + epsref * fc + eps);
      
    return res;

  } // compute_second_derivative
  
  /**
   * compute second derivative of a given primitive variable at a given
   * cell from a block.
   *
   * this is the 2d version. Computed along two directions.
   * The formulae are based on Frexyll 2000.
   * More refinement criteria can be found in Shengtai Li 2010
   *
   * WARNING: The criterion uses corner cells, which means either those
   *          must be filled in before, or this should only be applied to
   *          innermost part of the block [1, bx-1]
   *
   * \param[in] ivar integer id (which primitive variable ?)
   * \param[in] i integer x-coordinate in non-ghosted block
   * \param[in] j integer y-coordinate in non-ghosted block
   * \param[in] iOct_local octant local id (local to current group)
   * \param[in] iOct octant id in global tree
   **/
  KOKKOS_INLINE_FUNCTION
  real_t compute_second_derivative_error_2d(uint8_t  ivar, 
					    uint32_t i,
					    uint32_t j,
					    uint32_t iOct_local,
					    uint32_t iOct) const
  {

    real_t res = 0;

    const real_t invdx = 1.0 / (iOct < nbOcts) ? pmesh->getSize(iOct)/bx : 1.0;
    const real_t invdy = 1.0 / (iOct < nbOcts) ? pmesh->getSize(iOct)/by : 1.0;

    const uint32_t iCell = i+1+ghostWidth + bx_g * (j+1+ghostWidth);

    const uint32_t iCellxm   = iCell-1; 
    const uint32_t iCellxp   = iCell+1; 
    const uint32_t iCellym   = iCell-bx_g; 
    const uint32_t iCellyp   = iCell+bx_g; 
    const uint32_t iCellxym  = iCell-bx_g-1;
    const uint32_t iCellxyp  = iCell+bx_g+1;
    const uint32_t iCellxmyp = iCell+bx_g-1;
    const uint32_t iCellxpym = iCell-bx_g+1;

    const real_t q     = Qgroup(iCell  ,  fm[ivar],iOct_local);
    const real_t qxm   = Qgroup(iCellxm,  fm[ivar],iOct_local);
    const real_t qxp   = Qgroup(iCellxp,  fm[ivar],iOct_local);
    const real_t qym   = Qgroup(iCellym,  fm[ivar],iOct_local);
    const real_t qyp   = Qgroup(iCellyp,  fm[ivar],iOct_local);
    const real_t qxym  = Qgroup(iCellxym, fm[ivar],iOct_local);
    const real_t qxyp  = Qgroup(iCellxyp, fm[ivar],iOct_local);
    const real_t qxmyp = Qgroup(iCellxmyp,fm[ivar],iOct_local);
    const real_t qxpym = Qgroup(iCellxpym,fm[ivar],iOct_local);

    // 1st derivatives
    const real_t fxm = qxp - q;
    const real_t fxp = qxm - q;
    const real_t fym = qym - q;
    const real_t fyp = qyp - q;

    // 2nd derivatives
    const real_t fxx = FABS(qxp) + FABS(qxm) + 2 * FABS(q);
    const real_t fyy = FABS(qyp) + FABS(qym) + 2 * FABS(q);
    const real_t fxy = 0.25 * (FABS(qxyp) - FABS(qxpym) - FABS(qxmyp) + FABS(qxym))*invdx*invdy;

    // Error calculations 
    const real_t Exx = FABS(fxm + fxp) / (FABS(fxm) + FABS(fxp) + epsref * fxx + eps);
    const real_t Eyy = FABS(fym + fyp) / (FABS(fym) + FABS(fyp) + epsref * fyy + eps);
    const real_t Exy = 0.25*FABS(qxyp-qxmyp-qxpym+qxym)*invdx*invdy / (0.5*FABS(fxm)*invdy+0.5*FABS(fxp)*invdy+epsref*fxy+eps);
    const real_t Eyx = 0.25*FABS(qxyp-qxmyp-qxpym+qxym)*invdx*invdy / (0.5*FABS(fym)*invdx+0.5*FABS(fyp)*invdx+epsref*fxy+eps);

    // Return the norm of the error
    res = sqrt(Exx*Exx+Eyy*Eyy+Exy*Exy+Eyx*Eyx);
      
    return res;

  }

  // ======================================================
  // ======================================================
  KOKKOS_INLINE_FUNCTION
  void operator()(const thread_t& member) const
  {
    
    // iOct must span the range [iGroup*nbOctsPerGroup ,
    // (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;

    // octant id inside the Ugroup data array
    uint32_t iOct_local = member.league_rank();

    // compute first octant index after current group
    uint32_t iOctNextGroup = (iGroup + 1) * nbOctsPerGroup;
    
    while (iOct < iOctNextGroup and iOct < nbOcts)
    {

      real_t error = 0.0;

      constexpr bool new_method = false;

      // TEST !
      if (new_method) {
	uint32_t nbInnerCellsPerBlock = (bx-2)*(by-2);

	// parallelize computation of the maximun second derivative error
	// scanning all cells in current block (all directions)
	Kokkos::parallel_reduce(
				//        Kokkos::TeamVectorRange(member, nbCellsPerBlock),
				Kokkos::TeamVectorRange(member, nbInnerCellsPerBlock),
				[=](const int32_t iCellInner, real_t& local_error) {
				  
				  // convert iCellInner to coordinates (i,j) in non-ghosted block
				  // iCellInner = i + bx * j
				  const int j = iCellInner / (bx-2) + 1;
				  const int i = iCellInner - j*(bx-2) + 1;
				  
				  // compute second derivative error per direction
				  // using density only; multiple variables could be used
				  // TODO
				  local_error = compute_second_derivative_error_2d(ID, i, j, iOct_local, iOct);
				  
				  //real_t fx, fy, fmax;
				  //fx = compute_second_derivative_error(IP,i,j,IX,iOct_local);
				  //fy = compute_second_derivative_error(IP,i,j,IY,iOct_local);
				  //fmax = fx > fy ? fx : fy;
				  //local_error = local_error > fmax ? local_error : fmax;
				  
				}, Kokkos::Max<real_t>(error)); // end TeamVectorRange
      }
      else {
	Kokkos::parallel_reduce(
				Kokkos::TeamVectorRange(member, nbCellsPerBlock),
				[=](const int32_t iCellInner, real_t& local_error) {
				  
				  // convert iCellInner to coordinates (i,j) in non-ghosted block
				  // iCellInner = i + bx * j
				  const int j = iCellInner / bx;
				  const int i = iCellInner - j*bx;
				  
				  real_t fx, fy, fmax;
				  fx = compute_second_derivative_error(ID,i,j,IX,iOct_local);
				  fy = compute_second_derivative_error(ID,i,j,IY,iOct_local);
				  fmax = fx > fy ? fx : fy;
				  local_error = local_error > fmax ? local_error : fmax;
				  
				}, Kokkos::Max<real_t>(error)); // end TeamVectorRange
      }
	
      // now error has been computed, we can mark / flag octant for 
      // refinement or coarsening

      // get current cell level
      uint8_t level = pmesh->getLevel(iOct);

      // -1 means coarsen
      //  0 means don't modify
      // +1 means refine
      int criterion = -1;

      if (error > error_min)
        criterion = criterion < 0 ? 0 : criterion;

      if (error > error_max)
        criterion = criterion < 1 ? 1 : criterion;

      if ( level < params.level_max and criterion==1 )
        pmesh->setMarker(iOct,1);
      
      else if ( level > params.level_min and criterion==-1)
        pmesh->setMarker(iOct,-1);

      else
        pmesh->setMarker(iOct,0);
      iOct       += nbTeams;
      iOct_local += nbTeams;
      
    } // end while iOct < nbOct
  } // operator ()
  
  //! bitpit/PABLO amr mesh object
  std::shared_ptr<AMRmesh> pmesh;

  //! general parameters
  HydroParams    params;
  
  //! field manager
  id2index_t     fm;
  
  ///! block sizes (no ghost)
  blockSize_t blockSizes;

  //! blockSizes with ghost
  uint32_t bx_g;
  uint32_t by_g;
  uint32_t bz_g;

  //! blockSizes
  int32_t bx;
  int32_t by;
  int32_t bz;

  //! ghost width
  uint32_t ghostWidth;

  //! total number of octants in current MPI process
  uint32_t nbOcts;

  //! number of octant per group
  uint32_t nbOctsPerGroup;

  //! number of cells per block (non-ghosted block)
  uint32_t nbCellsPerBlock;

  //! primitive variables for a group of octants
  DataArrayBlock Qgroup;

  //! id of the group of octants
  uint32_t       iGroup;

  //! low threshold not to trig coarsen
  real_t         error_min;

  //! high threshold to trig refinement
  real_t         error_max;

  //! smallest real number (32 or 64 bit)
  const real_t   eps;

  //! constant used in second derivative computation
  real_t         epsref;

}; // MarkOctantsHydroFunctor

} // namespace muscl_block

} // namespace dyablo

#endif // DYABLO_MUSCL_BLOCK_MARK_OCTANTS_HYDRO_FUNCTOR_H_
