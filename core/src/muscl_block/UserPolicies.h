#ifndef _USER_POLICIES_H_
#define _USER_POLICIES_H_

#include "BoundaryPolicy.h"

namespace dyablo {
  namespace muscl_block {

    class UserPolicies : public BoundaryPolicy {
      public:
      UserPolicies() {};

      UserPolicies(HydroParams    params,
                   ConfigMap      configMap,
                   id2index_t     fm,
                   blockSize_t    blockSizes,
                   uint32_t       ghostWidth) :
        BoundaryPolicy(params, fm, blockSizes, ghostWidth)
      {
        // HERE : User defined constants/variables, etc.
      }

      KOKKOS_INLINE_FUNCTION
      HydroState2d fill_bc_2d(DataArrayBlock Ugroup, uint32_t iOct_local, uint32_t index, FACE_ID face, coord_t coords, real_t* xi) {
        // HERE : User defined boundary conditions
        HydroState2d Uout{0.0};

        return Uout;
      }
    }; // end class USerPolicies
  } // end namespace muscl_block
} // end namespace dyablo

#endif
