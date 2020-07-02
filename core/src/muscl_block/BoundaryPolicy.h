#ifndef _BOUNDARY_POLICY_H_
#define _BOUNDARY_POLICY_H_

#include "shared/FieldManager.h"
#include "shared/kokkos_shared.h"

// utils hydro
#include "shared/utils_hydro.h"

// utils block
#include "muscl_block/utils_block.h"

namespace dyablo {  
  namespace muscl_block {

    class BoundaryPolicy {
      public:
      BoundaryPolicy() {};
      BoundaryPolicy(HydroParams    params,
                     id2index_t     fm,
                     blockSize_t    blockSizes,
                     uint32_t       ghostWidth) :
        params(params), fm(fm), ghostWidth(ghostWidth)
      {
        bx   = blockSizes[IX];
        by   = blockSizes[IY];
        bz   = blockSizes[IZ];
        bx_g = blockSizes[IX] + 2 * ghostWidth;
        by_g = blockSizes[IY] + 2 * ghostWidth;
        bz_g = blockSizes[IZ] + 2 * ghostWidth;
      }

      KOKKOS_INLINE_FUNCTION
      HydroState2d fill_reflecting_2d(DataArrayBlock Ugroup, uint32_t iOct_local, uint32_t index, FACE_ID face, coord_t coords, real_t* xi) {
        coord_t copy_coords = coords;

        // Getting the correct source coordinates
        if (face == FACE_LEFT)
          copy_coords[IX] = 2 * ghostWidth - coords[IX] - 1;
        else if (face == FACE_BOTTOM)
          copy_coords[IY] = 2 * ghostWidth - coords[IY] - 1;
        else if (face == FACE_RIGHT)
          copy_coords[IX] = 2*(bx+ghostWidth) - 1 - coords[IX];
        else if (face == FACE_TOP)
          copy_coords[IY] = 2*(by+ghostWidth) - 1 - coords[IY];

        uint32_t index_copy = copy_coords[IX] + bx_g * copy_coords[IY];

        // Copying
        HydroState2d Uout;
        Uout[ID] = Ugroup(index_copy, fm[ID], iOct_local);
        Uout[IU] = Ugroup(index_copy, fm[IU], iOct_local);
        Uout[IV] = Ugroup(index_copy, fm[IV], iOct_local);
        Uout[IE] = Ugroup(index_copy, fm[IE], iOct_local);

        // And inverting velocity sign
        if (face == FACE_LEFT or face == FACE_RIGHT)
          Uout[IU] *= -1.0;
        if (face == FACE_BOTTOM or face == FACE_TOP)
          Uout[IV] *= -1.0;

        return Uout;
      }

        KOKKOS_INLINE_FUNCTION
        HydroState2d fill_absorbing_2d(DataArrayBlock Ugroup, uint32_t iOct_local, uint32_t index, FACE_ID face, coord_t coords, real_t* xi) const {
          // Getting the coordinates of the source cell
          coord_t copy_coords = coords;
          if (face == FACE_LEFT)
            copy_coords[IX] = ghostWidth;
          else if (face == FACE_RIGHT)
            copy_coords[IX] = bx_g - ghostWidth - 1;
          else if (face == FACE_BOTTOM)
            copy_coords[IY] = ghostWidth;
          else if (face == FACE_TOP)
            copy_coords[IY] = by_g - ghostWidth - 1;

          uint32_t index_copy = copy_coords[IX] + bx_g * copy_coords[IY];

          // Copying
          HydroState2d Uout;
          Uout[ID] = Ugroup(index_copy, fm[ID], iOct_local);
          Uout[IU] = Ugroup(index_copy, fm[IU], iOct_local);
          Uout[IV] = Ugroup(index_copy, fm[IV], iOct_local);
          Uout[IE] = Ugroup(index_copy, fm[IE], iOct_local);

          return Uout;
        }

        KOKKOS_INLINE_FUNCTION
        HydroState2d fill_bc_2d(DataArrayBlock Ugroup, uint32_t iOct_local, uint32_t index, FACE_ID face, coord_t coords, real_t* xi) const {
          // Should be overwritten by user in UserPolicies
          return HydroState2d{0.0, 0.0, 0.0, 0.0};
        }

        KOKKOS_INLINE_FUNCTION
        HydroState3d fill_reflecting_3d(DataArrayBlock Ugroup, uint32_t iOct_local, uint32_t index, FACE_ID face, coord_t coords, real_t* xi) const {
          // TODO
          return HydroState3d{0.0};
        }
        KOKKOS_INLINE_FUNCTION
        HydroState3d fill_absorbing_3d(DataArrayBlock Ugroup, uint32_t iOct_local, uint32_t index, FACE_ID face, coord_t coords, real_t* xi) const {
          // TODO
          return HydroState3d{0.0};
        }
        
        KOKKOS_INLINE_FUNCTION
        HydroState3d fill_bc_3d(DataArrayBlock Ugroup, uint32_t iOct_local, uint32_t index, FACE_ID face, coord_t coords, real_t* xi) const {
          // Should be overwritten by user in UserPolicies
          return HydroState3d{0.0};
        }

      //! parameters of the simulation
      HydroParams params;

      //! index converter for data
      id2index_t fm;

      //! size of the ghost layer
      uint32_t ghostWidth;

      //! size of the blocks without ghosts
      uint32_t bx, by, bz;

      //! size of the blocks with ghosts
      uint32_t bx_g, by_g, bz_g;
    }; // end BoundaryPolicy
  } // end namespace muscl_block
} // end namespace dyablo

#endif