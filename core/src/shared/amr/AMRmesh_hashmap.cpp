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

    if(inserted.existing())
        markers.value_at(inserted.index()) = marker;
    
    assert(!inserted.failed());
}

namespace {

using oct_view_t = AMRmesh_hashmap::oct_view_t;
using oct_view_device_t = AMRmesh_hashmap::oct_view_device_t;
using octs_coord_id = AMRmesh_hashmap::octs_coord_id;
using markers_t = AMRmesh_hashmap::markers_t;
using markers_device_t = AMRmesh_hashmap::markers_device_t;

/// Check il all siblings are marked for coarsening
KOKKOS_INLINE_FUNCTION
bool is_full_coarsening(const LightOctree_hashmap& lmesh, 
                        const oct_view_device_t& local_octs_coord, 
                        const markers_device_t& markers,
                        uint32_t iOct)
{
    //Position in parent octant
    uint16_t ix = local_octs_coord(iOct, octs_coord_id::IX);
    uint16_t iy = local_octs_coord(iOct, octs_coord_id::IY);
    uint16_t iz = local_octs_coord(iOct, octs_coord_id::IZ);
    uint16_t level = local_octs_coord(iOct, octs_coord_id::LEVEL);
    uint16_t target_level = level-1;

    int sx = (ix%2==0)?1:-1;
    int sy = (iy%2==0)?1:-1;
    int sz = (iz%2==0)?1:-1;

    int8_t ndim = lmesh.getNdim();

    for(int8_t z=0; z<(ndim-1); z++)
        for(int8_t y=0; y<2; y++)
            for(int8_t x=0; x<2; x++)
            {
                if( x!=0 || y!=0 || z!=0 )
                {
                    auto ns = lmesh.findNeighbors({iOct,false},{(int8_t)(sx*x),(int8_t)(sy*y),(int8_t)(sz*z)});
                    assert(ns.size()>0);
                    uint32_t iOct_other = ns[0].iOct;
                    
                    uint16_t target_level_n = local_octs_coord(iOct_other, octs_coord_id::LEVEL);
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
                    const oct_view_device_t& local_octs_coord, 
                    markers_device_t& markers )
{
    markers_device_t markers_out(lmesh.getNumOctants());
    
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
KOKKOS_INLINE_FUNCTION
bool need_refine_to_balance(const LightOctree_hashmap& lmesh, 
                        const oct_view_device_t& local_octs_coord, 
                        const markers_device_t& markers,
                        uint32_t iOct)
{
    uint16_t new_level = local_octs_coord(iOct, octs_coord_id::LEVEL);
    uint32_t idm = markers.find(iOct);
    if( markers.valid_at(idm) )
        new_level += markers.value_at(idm);

    int8_t nz_max = lmesh.getNdim() == 2? 0:1;
    for( int8_t nz=-nz_max; nz<=nz_max; nz++ )
        for( int8_t ny=-1; ny<=1; ny++ )
            for( int8_t nx=-1; nx<=1; nx++ )
            {
                if( nx!=0 || ny!=0 || nz!=0 )
                {
                    LightOctree_base::NeighborList neighbors = 
                        lmesh.findNeighbors({iOct,false}, {nx,ny,nz});
                    for(uint32_t n=0; n<neighbors.size(); n++)
                    {
                        uint32_t iOct_neighbor = neighbors[n].iOct;
                        uint16_t new_level_neighbor = local_octs_coord(iOct_neighbor, octs_coord_id::LEVEL);
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
                            const oct_view_device_t& local_octs_coord, 
                            markers_device_t& markers)
{
    uint32_t nbOcts = lmesh.getNumOctants();
    markers_device_t markers_out(nbOcts);

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

            assert(marker<=1);
        }

        if(marker != 0)
            markers_out.insert(iOct, marker);
        
    }, Kokkos::Sum<int>(modified));

    markers = markers_out;

    return modified!=0;
}

oct_view_t adapt_apply( const oct_view_t& local_octs_coord, 
                        const markers_t& markers, uint8_t ndim )
{
    int n_refine = (ndim==2) ? 4 : 8;

    uint32_t count_refine=0, count_coarsen=0;
    Kokkos::parallel_reduce( "adapt::count_markers+", 
                             Kokkos::RangePolicy<Kokkos::OpenMP>(0, markers.capacity()),
                             KOKKOS_LAMBDA(uint32_t im, uint32_t& count_local)
    {
        if( markers.valid_at(im) && markers.value_at(im)==1 )
            count_local++;
    }, count_refine);
    Kokkos::parallel_reduce( "adapt::count_markers-",  
                             Kokkos::RangePolicy<Kokkos::OpenMP>(0, markers.capacity()),
                             KOKKOS_LAMBDA(uint32_t im, uint32_t& count_local)
    {
        if( markers.valid_at(im) && markers.value_at(im)==-1 )
            count_local++;
    }, count_coarsen);
    uint32_t old_size = local_octs_coord.extent(0);

    uint32_t new_size = (old_size - count_refine - count_coarsen) + n_refine * count_refine + count_coarsen/n_refine;

    oct_view_t local_octs_coord_out("local_octs_coord", new_size);

    Kokkos::parallel_scan( "adapt::apply",  
                            Kokkos::RangePolicy<Kokkos::OpenMP>(0, old_size),
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
                uint16_t ix = local_octs_coord(iOct_old, octs_coord_id::IX);
                uint16_t iy = local_octs_coord(iOct_old, octs_coord_id::IY);
                uint16_t iz = local_octs_coord(iOct_old, octs_coord_id::IZ);
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
            uint16_t ix = local_octs_coord(iOct_old, octs_coord_id::IX);
            uint16_t iy = local_octs_coord(iOct_old, octs_coord_id::IY);
            uint16_t iz = local_octs_coord(iOct_old, octs_coord_id::IZ);
            uint16_t level = local_octs_coord(iOct_old, octs_coord_id::LEVEL);

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

                    local_octs_coord_out(iOct_new + j, octs_coord_id::IX) = ix + dx;
                    local_octs_coord_out(iOct_new + j, octs_coord_id::IY) = iy + dy;
                    local_octs_coord_out(iOct_new + j, octs_coord_id::IZ) = iz + dz;
                    local_octs_coord_out(iOct_new + j, octs_coord_id::LEVEL) = level+1;
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
                local_octs_coord_out(iOct_new, octs_coord_id::IX) = ix;
                local_octs_coord_out(iOct_new, octs_coord_id::IY) = iy;
                local_octs_coord_out(iOct_new, octs_coord_id::IZ) = iz;
                local_octs_coord_out(iOct_new, octs_coord_id::LEVEL) = level;
            }            
        }

        iOct_new += n_new_octants;
    });

    return local_octs_coord_out;
}

} // namespace

void AMRmesh_hashmap::adapt(bool dummy)
{
    std::cout << "---- Adapt ----" << std::endl;
    std::cout << "Octs before adapt " << local_octs_count << std::endl;

    std::shared_ptr<AMRmesh_hashmap> this_sptr(this, [](AMRmesh_hashmap*){});
    LightOctree_hashmap lmesh(this_sptr, level_min, level_max);

    {
        oct_view_device_t local_octs_coord_device("local_octs_coord_device", local_octs_count);
        markers_device_t markers_device(markers.capacity());
        Kokkos::deep_copy(local_octs_coord_device, local_octs_coord);
        Kokkos::deep_copy(markers_device, markers);

        bool modified = true;
        while(modified)
        {
                adapt_clean(lmesh, local_octs_coord_device, markers_device);
                modified = adapt_21balance_step(lmesh, local_octs_coord_device, markers_device);
        }
        adapt_clean(lmesh, local_octs_coord_device, markers_device);

        markers.clear();
        Kokkos::deep_copy(markers, markers_device);
    }

    oct_view_t new_octs = adapt_apply( local_octs_coord, markers, dim );

    local_octs_count = new_octs.extent(0);
    local_octs_coord = new_octs;    

    std::cout << "Octs after adapt " << local_octs_count << std::endl;
    std::cout << "---- End Adapt ----" << std::endl;

    this->markers = markers_t(local_octs_count);
}

}