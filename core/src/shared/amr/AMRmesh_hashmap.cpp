#include "AMRmesh_hashmap.h"

namespace dyablo{

void AMRmesh_hashmap::adaptGlobalRefine()
{
    uint32_t nb_subocts = std::pow(2, dim);
    uint32_t new_octs_count = local_octs_count*nb_subocts;
    oct_view_t new_octs("local_octs_coord", new_octs_count);
    for(uint32_t iOct_old=0; iOct_old<local_octs_count; iOct_old++)
    {
        uint16_t x = local_octs_coord(iOct_old, IX);
        uint16_t y = local_octs_coord(iOct_old, IY);
        uint16_t z = local_octs_coord(iOct_old, IZ);
        uint16_t level = local_octs_coord(iOct_old, LEVEL);

        for(uint8_t iz=0; iz<(dim-1); iz++)
            for(uint16_t iy=0; iy<2; iy++)
                for(uint16_t ix=0; ix<2; ix++)
                {
                    uint32_t iOct_new = iOct_old*nb_subocts + ix + 2*iy + 4*iz;
                    new_octs(iOct_new, IX) = 2*x + ix;
                    new_octs(iOct_new, IY) = 2*y + iy;
                    new_octs(iOct_new, IZ) = 2*z + iz;
                    new_octs(iOct_new, LEVEL) = level + 1;
                }
    }

    this->local_octs_count = new_octs_count;
    this->local_octs_coord = new_octs;
}

}