#include "AMRmesh_hashmap.h"

#include "shared/LightOctree.h"

namespace dyablo{

AMRmesh_hashmap::AMRmesh_hashmap( int dim, int balance_codim, const std::array<bool,3>& periodic, uint8_t level_min, uint8_t level_max )
        : dim(dim), periodic(periodic), markers(1), level_min(level_min), level_max(level_max)
{
    assert(dim == 2 || dim == 3);
    assert(balance_codim <= dim);

    this->local_octs_count = 1;
    this->local_octs_coord = oct_view_t("local_octs_coord", 1);
}

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

    this->markers = markers_t(local_octs_count);
}

void AMRmesh_hashmap::setMarkersCapacity(uint32_t capa)
{
}

void AMRmesh_hashmap::setMarker(uint32_t iOct, int marker)
{
    auto inserted = this->markers.insert( iOct, marker );

    assert(inserted.success());
}

namespace {

using oct_view_t = AMRmesh_hashmap::oct_view_t;
using octs_coord_id = AMRmesh_hashmap::octs_coord_id;
using markers_t = AMRmesh_hashmap::markers_t;
constexpr octs_coord_id IX = octs_coord_id::IX;
constexpr octs_coord_id IY = octs_coord_id::IY;
constexpr octs_coord_id IZ = octs_coord_id::IZ;
constexpr octs_coord_id LEVEL = octs_coord_id::LEVEL;

/// Check il all siblings are marked for coarsening
bool is_full_coarsening(const LightOctree_hashmap& lmesh, 
                        const oct_view_t& local_octs_coord, 
                        const markers_t& markers,
                        uint32_t iOct)
{
    //Position in parent octant
    uint16_t ix = local_octs_coord(iOct, IX);
    uint16_t iy = local_octs_coord(iOct, IY);
    uint16_t iz = local_octs_coord(iOct, IZ);
    uint16_t level = local_octs_coord(iOct, LEVEL);
    uint16_t target_level = level-1;

    int sx = (ix%2==0)?1:-1;
    int sy = (iy%2==0)?1:-1;
    int sz = (iz%2==0)?1:-1;

    for(int8_t z=0; z<2; z++)
        for(int8_t y=0; y<2; y++)
            for(int8_t x=0; x<2; x++)
            {
                if( x!=0 || y!=0 || z!=0 )
                {
                    uint32_t iOct_other = lmesh.findNeighbors({iOct,false},{sx*x,sy*y,sz*z})[0].iOct;
                    
                    uint16_t target_level_n = local_octs_coord(iOct_other, LEVEL);
                    uint32_t im = markers.find(iOct_other);
                    if( markers.valid_at(im) )
                        target_level_n += markers.value_at(im);
                    
                    if( target_level_n != target_level ) return false;
                }
            }
    return true;
}

/// Clean markers, remove octants that are only partially coarsened from markers list
void adapt_clean(   const LightOctree_hashmap& lmesh, 
                    const oct_view_t& local_octs_coord, 
                    markers_t& markers )
{
    markers_t markers_out(lmesh.getNumOctants());
    
    Kokkos::parallel_for( "adapt::clean", markers.capacity(),
                            KOKKOS_LAMBDA(uint32_t idx)
    {
        if(!markers.valid_at(idx)) return;

        uint32_t iOct = markers.key_at(idx);
        int marker = markers.value_at(idx);
        assert( marker==-1 || marker==1 );
        // Remove partial octants coarsening        
        if(marker==1 ||  ( marker==-1 && is_full_coarsening(lmesh, local_octs_coord, markers, iOct)))
        { 
            auto res = markers_out.insert(iOct, marker);
            assert(res.success());
        }
    });

    markers = markers_out;
}

/// Returns true if cell needs to be refined to verify 2:1 criterion
bool need_refine_to_balance(const LightOctree_hashmap& lmesh, 
                        const oct_view_t& local_octs_coord, 
                        const markers_t& markers,
                        uint32_t iOct)
{
    uint16_t new_level = local_octs_coord(iOct, LEVEL);
    uint32_t idm = markers.find(iOct);
    if( markers.valid_at(idm) )
        new_level += markers.value_at(idm);

    for( int8_t nz=-1; nz<=1; nz++ )
        for( int8_t ny=-1; ny<=1; ny++ )
            for( int8_t nx=-1; nx<=1; nx++ )
            {
                if( nx!=0 || ny!=0 || nz!=0 )
                {
                    LightOctree_base::NeighborList neighbors = 
                        lmesh.findNeighbors({iOct,false}, {nx,ny,nz});
                    if(neighbors.size() != 0) 
                    {
                        uint32_t iOct_neighbor = neighbors[0].iOct;
                        uint16_t new_level_neighbor = local_octs_coord(iOct_neighbor, LEVEL);
                        uint32_t idmn = markers.find(iOct_neighbor);
                        if( markers.valid_at(idmn) )
                            new_level_neighbor += markers.value_at(idmn);

                        if( new_level_neighbor - new_level > 1 )
                        {
                            return true;
                        }
                    }
                }
            }
    return false;
}

/// Iterate over all octants and modify markers when not verifying 2:1
bool adapt_21balance_step(  const LightOctree_hashmap& lmesh, 
                            const oct_view_t& local_octs_coord, 
                            markers_t& markers)
{
    uint32_t nbOcts = lmesh.getNumOctants();
    markers_t markers_out(nbOcts);

    int modified = 0;
    Kokkos::parallel_reduce( "adapt::2:1step", nbOcts,
                          KOKKOS_LAMBDA(uint32_t iOct, int& modified_local)
    {
        //Get current marker
        int marker;
        {
            uint32_t idm = markers.find(iOct);        
            if( markers.valid_at(idm) )
                marker = markers.value_at(idm);
            else
                marker = 0;
        }
        
        if( need_refine_to_balance(lmesh, local_octs_coord, markers, iOct) )
        {
            modified_local++;
            marker++;
        }

        if(marker != 0)
            markers_out.insert(iOct, marker);
        
    }, Kokkos::Sum<int>(modified));

    markers = markers_out;

    return modified;
}

oct_view_t adapt_apply( const oct_view_t& local_octs_coord, 
                        const markers_t& markers )
{
    int n_refine = 8;

    uint32_t count_refine=0, count_coarsen=0;
    Kokkos::parallel_reduce( "adapt::count_markers+", markers.capacity(),
                             KOKKOS_LAMBDA(uint32_t im, uint32_t& count_local)
    {
        if( markers.valid_at(im) && markers.value_at(im)==1 )
            count_local++;
    }, count_refine);
    Kokkos::parallel_reduce( "adapt::count_markers+", markers.capacity(),
                             KOKKOS_LAMBDA(uint32_t im, uint32_t& count_local)
    {
        if( markers.valid_at(im) && markers.value_at(im)==-1 )
            count_local++;
    }, count_coarsen);
    uint32_t old_size = local_octs_coord.extent(0);

    uint32_t new_size = (old_size - count_refine - count_coarsen) + n_refine * count_refine + count_coarsen/n_refine;

    oct_view_t local_octs_coord_out("local_octs_coord", new_size);

    Kokkos::parallel_scan( "adapt::apply", old_size,
                            KOKKOS_LAMBDA(uint32_t iOct_old, uint32_t& iOct_new, const bool final)
    {
        //Get marker
        int marker;
        {
            uint32_t idm = markers.find(iOct_old);  
            if( markers.valid_at(idm) )
                marker = markers.value_at(idm);
            else
                marker = 0;
        }
        //Compute number of new octants for current old octant
        uint32_t n_new_octants;
        {
            if( marker == 1 )
                n_new_octants = n_refine;
            else if( marker == 0 )
                n_new_octants = 1;
            else if( marker == -1 )
            {
                // Add coarsend octant only for first one in morton order
                uint16_t ix = local_octs_coord(iOct_old, IX);
                uint16_t iy = local_octs_coord(iOct_old, IY);
                uint16_t iz = local_octs_coord(iOct_old, IZ);
                if( ix%2==0 && iy%2==0 && iz%2==0 )
                    n_new_octants = 1;
                else 
                    n_new_octants = 0;
            }
            else
            {
                n_new_octants = 0;
                assert(false); // Only +1 -1 allowed in markers
            }
        }
        
        if(final && n_new_octants>0)
        {
            uint16_t ix = local_octs_coord(iOct_old, IX);
            uint16_t iy = local_octs_coord(iOct_old, IY);
            uint16_t iz = local_octs_coord(iOct_old, IZ);
            uint16_t level = local_octs_coord(iOct_old, LEVEL);

            if(marker==1)
            {
                ix *= 2;
                iy *= 2;
                iz *= 2;
                for( uint32_t j=0; j<n_new_octants; j++ )
                {
                    uint32_t dz = j/(2*2);
                    uint32_t dy = (j-dz*2*2)/2;
                    uint32_t dx = j-dz*2*2-dy*2;

                    local_octs_coord_out(iOct_new + j, IX) = ix + dx;
                    local_octs_coord_out(iOct_new + j, IY) = iy + dy;
                    local_octs_coord_out(iOct_new + j, IZ) = iz + dz;
                    local_octs_coord_out(iOct_new + j, LEVEL) = level+1;
                }
            }
            else 
            {
                if(marker==-1)
                {
                    ix /= 2;
                    iy /= 2;
                    iz /= 2;
                    level -= 1;
                }
                local_octs_coord_out(iOct_new, IX) = ix;
                local_octs_coord_out(iOct_new, IY) = iy;
                local_octs_coord_out(iOct_new, IZ) = iz;
                local_octs_coord_out(iOct_new, LEVEL) = level;
            }            
        }

        iOct_new += n_new_octants;
    });

    return local_octs_coord_out;
}

} // namespace

void AMRmesh_hashmap::adapt(bool dummy)
{
    std::shared_ptr<AMRmesh_hashmap> this_sptr(this, [](AMRmesh_hashmap*){});
    LightOctree_hashmap lmesh(this_sptr, level_min, level_max);

    std::cout << "---- Adapt ----" << std::endl;
    std::cout << "Octs before adapt " << local_octs_count << std::endl;

    bool modified = true;
    while(modified)
    {
        adapt_clean(lmesh, local_octs_coord, markers);
        modified = adapt_21balance_step(lmesh, local_octs_coord, markers);
    }
    adapt_clean(lmesh, local_octs_coord, markers);

    oct_view_t new_octs = adapt_apply( local_octs_coord, markers );

    local_octs_coord = new_octs;
    local_octs_count = new_octs.extent(0);

    std::cout << "Octs after adapt " << local_octs_count << std::endl;
    std::cout << "---- End Adapt ----" << std::endl;

    this->markers = markers_t(local_octs_count);
}

}