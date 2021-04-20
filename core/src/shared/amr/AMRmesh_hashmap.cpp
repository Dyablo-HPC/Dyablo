#include "AMRmesh_hashmap.h"

#include "shared/amr/LightOctree.h"
#include "shared/morton_utils.h"
//#include "utils/io/AMRMesh_output_vtk.h"

#include "shared/mpi/GhostCommunicator.h"

// TODO cleaner conditional compilation
#ifdef DYABLO_USE_GPU_MESH

namespace dyablo{

AMRmesh_hashmap::AMRmesh_hashmap( int dim, int balance_codim, const std::array<bool,3>& periodic, uint8_t level_min, uint8_t level_max )
        : dim(dim), periodic(periodic), markers(1), level_min(level_min), level_max(level_max)
{
    assert(dim == 2 || dim == 3);
    assert(balance_codim <= dim);

    this->total_octs_count = 1;
    this->global_id_begin = 0;

    int nb_octs = ( this->getRank() == 0 ) ? 1 : 0;
    this->local_octs_coord = oct_view_t("local_octs_coord", NUM_OCTS_COORDS, nb_octs);
    this->ghost_octs_coord = oct_view_t("ghost_octs_coord", NUM_OCTS_COORDS, 0);
}

void AMRmesh_hashmap::adaptGlobalRefine()
{
    assert( this->sequential_mesh ); //No global refine after scattering

    uint32_t nb_subocts = std::pow(2, dim);
    uint32_t new_octs_count = this->getNumOctants()*nb_subocts;
    oct_view_t new_octs("local_octs_coord",  NUM_OCTS_COORDS, new_octs_count);
    for(uint32_t iOct_old=0; iOct_old<this->getNumOctants(); iOct_old++)
    {
        uint16_t x = local_octs_coord(IX, iOct_old);
        uint16_t y = local_octs_coord(IY, iOct_old);
        uint16_t z = local_octs_coord(IZ, iOct_old);
        uint16_t level = local_octs_coord(LEVEL, iOct_old);

        for(uint8_t iz=0; iz<(dim-1); iz++)
            for(uint16_t iy=0; iy<2; iy++)
                for(uint16_t ix=0; ix<2; ix++)
                {
                    uint32_t iOct_new = iOct_old*nb_subocts + ix + 2*iy + 4*iz;
                    new_octs(IX, iOct_new) = 2*x + ix;
                    new_octs(IY, iOct_new) = 2*y + iy;
                    new_octs(IZ, iOct_new) = 2*z + iz;
                    new_octs(LEVEL, iOct_new) = level + 1;
                }
    }

    this->local_octs_coord = new_octs;
    this->markers = markers_t(this->getNumOctants());

    uint32_t nb_octs = this->getNumOctants();
    MPI_Allreduce( &nb_octs, &total_octs_count, 1, MPI_UINT32_T, MPI::SUM, MPI_COMM_WORLD );
}

void AMRmesh_hashmap::setMarkersCapacity(uint32_t capa)
{
}

void AMRmesh_hashmap::setMarker(uint32_t iOct, int marker)
{
    if(marker==0) return;
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
using OctantIndex = LightOctree_base::OctantIndex;

KOKKOS_INLINE_FUNCTION
Kokkos::Array<uint16_t, 3> oct_getpos(  const oct_view_device_t& local_octs_coord, 
                                        const oct_view_device_t& ghost_octs_coord,
                                        const OctantIndex& iOct)
{
    Kokkos::Array<uint16_t, 3> pos;
    if(iOct.isGhost)
    {
        pos[IX] = ghost_octs_coord(octs_coord_id::IX, iOct.iOct);
        pos[IY] = ghost_octs_coord(octs_coord_id::IY, iOct.iOct);
        pos[IZ] = ghost_octs_coord(octs_coord_id::IZ, iOct.iOct);
    }
    else
    {
        pos[IX] = local_octs_coord(octs_coord_id::IX, iOct.iOct);
        pos[IY] = local_octs_coord(octs_coord_id::IY, iOct.iOct);
        pos[IZ] = local_octs_coord(octs_coord_id::IZ, iOct.iOct);
    }
    return pos;
}

KOKKOS_INLINE_FUNCTION
uint16_t oct_getlevel(  const oct_view_device_t& local_octs_coord, 
                        const oct_view_device_t& ghost_octs_coord,
                        const OctantIndex& iOct)
{
    if(iOct.isGhost)
    {
        return ghost_octs_coord(octs_coord_id::LEVEL, iOct.iOct);
    }
    else
    {
        return local_octs_coord(octs_coord_id::LEVEL, iOct.iOct);
    }
}

/// Check if all siblings are marked for coarsening
/// iOct Cannot be a ghost : we may not have all the siblings
KOKKOS_INLINE_FUNCTION
bool is_full_coarsening(const LightOctree_hashmap& lmesh, 
                        const oct_view_device_t& local_octs_coord, 
                        const oct_view_device_t& ghost_octs_coord, 
                        const markers_device_t& markers,
                        const OctantIndex& iOct)
{
    assert(!iOct.isGhost); //Cannot be a ghost : we may not have all the siblings

    uint32_t nbOcts = lmesh.getNumOctants();
    //Position in parent octant
    auto pos = oct_getpos(local_octs_coord, ghost_octs_coord, iOct);
    uint16_t level = oct_getlevel(local_octs_coord, ghost_octs_coord, iOct);
    
    uint16_t target_level = level-1;

    int sx = (pos[IX]%2==0)?1:-1;
    int sy = (pos[IY]%2==0)?1:-1;
    int sz = (pos[IZ]%2==0)?1:-1;

    int8_t ndim = lmesh.getNdim();

    for(int8_t z=0; z<(ndim-1); z++)
        for(int8_t y=0; y<2; y++)
            for(int8_t x=0; x<2; x++)
            {
                if( x!=0 || y!=0 || z!=0 )
                {
                    auto ns = lmesh.findNeighbors(iOct,{(int8_t)(sx*x),(int8_t)(sy*y),(int8_t)(sz*z)});
                    assert(ns.size()>0);
                    const OctantIndex& iOct_other = ns[0];
                    uint16_t target_level_n = oct_getlevel(local_octs_coord, ghost_octs_coord, iOct_other);;
                    uint32_t im = markers.find(OctantIndex::OctantIndex_to_iOctLocal(iOct_other, nbOcts));
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
                    const oct_view_device_t& ghost_octs_coord, 
                    markers_device_t& markers )
{
    uint32_t numOcts = local_octs_coord.extent(1);
    uint32_t numGhosts = ghost_octs_coord.extent(1);
    markers_device_t markers_out(numOcts+numGhosts);
    
    Kokkos::parallel_for( "adapt::clean", markers.capacity(),
                            KOKKOS_LAMBDA(uint32_t idx)
    {
        if(!markers.valid_at(idx)) return;

        uint32_t iOct_local = markers.key_at(idx);
        OctantIndex iOct = OctantIndex::iOctLocal_to_OctantIndex( iOct_local, numOcts)  ;

        if(iOct.isGhost) return; // Don't modify ghosts

        int marker = markers.value_at(idx);
        assert( marker==-1 || marker==1 );
        // Remove partial octants coarsening        
        if(marker==1 ||  ( marker==-1 && is_full_coarsening(lmesh, local_octs_coord, ghost_octs_coord, markers, iOct)))
        { 
            auto res = markers_out.insert(iOct_local, marker);
            assert(res.success());
        }
    });

    markers = markers_out;
}

/// Returns true if cell needs to be refined to verify 2:1 criterion
KOKKOS_INLINE_FUNCTION
bool need_refine_to_balance(const LightOctree_hashmap& lmesh, 
                        const oct_view_device_t& local_octs_coord, 
                        const oct_view_device_t& ghost_octs_coord, 
                        const markers_device_t& markers,
                        const OctantIndex& iOct)
{
    assert(!iOct.isGhost);

    uint32_t nbOcts = lmesh.getNumOctants();

    uint16_t new_level = oct_getlevel(local_octs_coord, ghost_octs_coord, iOct);
    uint32_t idm = markers.find(OctantIndex::OctantIndex_to_iOctLocal(iOct, nbOcts));
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
                        lmesh.findNeighbors(iOct, {nx,ny,nz});
                    for(uint32_t n=0; n<neighbors.size(); n++)
                    {
                        const OctantIndex& iOct_neighbor = neighbors[n];
                        uint16_t new_level_neighbor = oct_getlevel(local_octs_coord, ghost_octs_coord, iOct_neighbor);
                        uint32_t idmn = markers.find(OctantIndex::OctantIndex_to_iOctLocal(iOct_neighbor, nbOcts));
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
                            const oct_view_device_t& ghost_octs_coord, 
                            markers_device_t& markers)
{
    uint32_t nbOcts = lmesh.getNumOctants();
    markers_device_t markers_out(nbOcts);

    int modified = 0;
    Kokkos::parallel_reduce( "adapt::2:1step", nbOcts,
                          KOKKOS_LAMBDA(uint32_t iOct_local, int& modified_local)
    {
        //Get current marker
        int marker;
        {
            uint32_t idm = markers.find(iOct_local);        
            if( markers.valid_at(idm) )
                marker = markers.value_at(idm);
            else
                marker = 0;
        }
        
        OctantIndex iOct = OctantIndex::iOctLocal_to_OctantIndex(iOct_local, nbOcts);
        if( need_refine_to_balance(lmesh, local_octs_coord, ghost_octs_coord, markers, iOct) )
        {
            modified_local++;
            marker++;

            assert(marker<=1);
        }

        if(marker != 0)
            markers_out.insert(iOct_local, marker);
        
    }, Kokkos::Sum<int>(modified));

    markers = markers_out;

    return modified!=0;
}

oct_view_t adapt_apply( const oct_view_t& local_octs_coord, 
                        const markers_t& markers, uint8_t ndim )
{
    int n_refine = (ndim==2) ? 4 : 8;
    uint32_t old_size = local_octs_coord.extent(1);
    uint32_t count_refine=0, count_coarsen=0;
    Kokkos::parallel_reduce( "adapt::count_markers+", 
                             Kokkos::RangePolicy<Kokkos::OpenMP>(0, markers.capacity()),
                             KOKKOS_LAMBDA(uint32_t im, uint32_t& count_local)
    {
        if( markers.valid_at(im) && markers.value_at(im)==1 && markers.key_at(im)<old_size )
            count_local++;
    }, count_refine);
    Kokkos::parallel_reduce( "adapt::count_markers-",  
                             Kokkos::RangePolicy<Kokkos::OpenMP>(0, markers.capacity()),
                             KOKKOS_LAMBDA(uint32_t im, uint32_t& count_local)
    {
        if( markers.valid_at(im) && markers.value_at(im)==-1 && markers.key_at(im)<old_size )
            count_local++;
    }, count_coarsen);
    

    uint32_t new_size = (old_size - count_refine - count_coarsen) + n_refine * count_refine + count_coarsen/n_refine;

    oct_view_t local_octs_coord_out("local_octs_coord", octs_coord_id::NUM_OCTS_COORDS, new_size);

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
                uint16_t ix = local_octs_coord(octs_coord_id::IX, iOct_old);
                uint16_t iy = local_octs_coord(octs_coord_id::IY, iOct_old);
                uint16_t iz = local_octs_coord(octs_coord_id::IZ, iOct_old);
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
            uint16_t ix = local_octs_coord(octs_coord_id::IX, iOct_old);
            uint16_t iy = local_octs_coord(octs_coord_id::IY, iOct_old);
            uint16_t iz = local_octs_coord(octs_coord_id::IZ, iOct_old);
            uint16_t level = local_octs_coord(octs_coord_id::LEVEL, iOct_old);

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

                    local_octs_coord_out(octs_coord_id::IX, iOct_new + j) = ix + dx;
                    local_octs_coord_out(octs_coord_id::IY, iOct_new + j) = iy + dy;
                    local_octs_coord_out(octs_coord_id::IZ, iOct_new + j) = iz + dz;
                    local_octs_coord_out(octs_coord_id::LEVEL, iOct_new + j) = level+1;
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
                local_octs_coord_out(octs_coord_id::IX, iOct_new) = ix;
                local_octs_coord_out(octs_coord_id::IY, iOct_new) = iy;
                local_octs_coord_out(octs_coord_id::IZ, iOct_new) = iz;
                local_octs_coord_out(octs_coord_id::LEVEL, iOct_new) = level;
            }            
        }

        iOct_new += n_new_octants;
        assert(iOct_new <= new_size);
    });

    return local_octs_coord_out;
}

using morton_t = uint64_t;

morton_t shift_level( morton_t m, uint16_t from_level, uint16_t to_level  )
{
    int16_t ldiff = to_level-from_level;
    if(ldiff > 0)
        return m << 3*ldiff;
    else 
        return m >> -3*ldiff;
}

// Compute morton at level_max;
morton_t get_morton_smaller(uint16_t ix, uint16_t iy, uint16_t iz, uint16_t level, uint16_t level_max)
{
    morton_t morton = compute_morton_key( ix,iy,iz );
    return shift_level(morton, level, level_max);
};

morton_t get_morton_smaller(const oct_view_t& local_octs_coord, uint32_t idx, uint16_t level_max)
{
    uint16_t ix = local_octs_coord(octs_coord_id::IX, idx);
    uint16_t iy = local_octs_coord(octs_coord_id::IY, idx);
    uint16_t iz = local_octs_coord(octs_coord_id::IZ, idx);
    uint16_t level = local_octs_coord(octs_coord_id::LEVEL, idx);
    return get_morton_smaller(ix,iy,iz,level,level_max);
}

std::set< std::pair<int, uint32_t> > discover_ghosts(
        const std::vector<morton_t>& morton_intervals,
        const AMRmesh_hashmap::oct_view_t& local_octs_coord,
        uint8_t level_max, uint8_t ndim, const std::array<bool,3>& periodic)
{
    uint32_t mpi_rank = hydroSimu::GlobalMpiSession::getRank();
    uint32_t mpi_size = hydroSimu::GlobalMpiSession::getNProc();

    auto find_rank = [mpi_size, &morton_intervals](morton_t morton)
    {
        // upper_bound : first verifying value > morton
        auto it = std::upper_bound( morton_intervals.begin(), morton_intervals.end(), morton );
        // neighbor_rank: last interval value <= morton
        uint32_t neighbor_rank = it - morton_intervals.begin() - 1;
        assert( neighbor_rank<mpi_size );
        assert( morton_intervals[neighbor_rank] <= morton);
        assert( morton_intervals[neighbor_rank+1] > morton);
        return neighbor_rank;
    };

    // TODO : Use GPU
    // Compute new neighborhood : determine which octants to send each MPI rank
    // For each local octant, compute MPI rank for each same-size virtual neighbor 
    // using morton_intervals. If same-size virtual neighbor is split between multiple MPI
    // ranks, current octant must be sent to each of these MPI ranks
    std::set< std::pair<int, uint32_t> > local_octants_to_send_set;
    uint32_t nbOct_local = local_octs_coord.extent(1);
    for( uint32_t iOct=0; iOct < nbOct_local; iOct++ )
    {
        uint16_t ix = local_octs_coord(octs_coord_id::IX, iOct);
        uint16_t iy = local_octs_coord(octs_coord_id::IY, iOct);
        uint16_t iz = local_octs_coord(octs_coord_id::IZ, iOct);
        uint16_t level = local_octs_coord(octs_coord_id::LEVEL, iOct);
        uint16_t max_i = std::pow(2, level);

        assert(find_rank(get_morton_smaller(ix,iy,iz,level,level_max)) == mpi_rank);

        int8_t dz_max = (ndim == 2)? 0:1;
        for( int16_t dz=-dz_max; dz<=dz_max; dz++ )
        for( int16_t dy=-1; dy<=1; dy++ )
        for( int16_t dx=-1; dx<=1; dx++ )
        if(   (dx!=0 || dy!=0 || dz!=0)
           && (periodic[IX] || ( 0<=ix+dx && ix+dx<max_i ))
           && (periodic[IY] || ( 0<=iy+dy && iy+dy<max_i ))
           && (periodic[IZ] || ( 0<=iz+dz && iz+dz<max_i )) )
        {
            morton_t m_neighbor = get_morton_smaller((ix+dx+max_i)%max_i,(iy+dy+max_i)%max_i,(iz+dz+max_i)%max_i,level,level_max);
            // Find the first rank where morton_intervals[rank] >= m_neighbor
            uint32_t neighbor_rank = find_rank(m_neighbor);

            // Verify that the whole same-size virtual neighbor is owned by neighbor_rank
            // i.e : last suboctant of same-size neighbor is owned by the same MPI
            morton_t m_next = shift_level(m_neighbor, level_max, level) + 1;
                     m_next = shift_level(m_next, level, level_max);
            if( level<level_max && m_next > morton_intervals[neighbor_rank+1] )
            {// Neighbors are owned by only one MPI
                if(neighbor_rank != mpi_rank)
                    local_octants_to_send_set.insert({neighbor_rank, iOct});
            }
            else
            {// Neighbors are scattered between multiple MPIs
                //Iterate over neighbor's subcells
                for( int16_t sz=0; sz<=dz_max; sz++ )
                for( int16_t sy=0; sy<2; sy++ )
                for( int16_t sx=0; sx<2; sx++ )
                {
                    // Add smaller neighbor only if near original octant
                    // direction is unsconstrained OR offset left + suboctant right OR offset right + suboctant left
                    //         (dx == 0)           OR        (dx==-1 && sx==1)      OR     (dx==1 && sx==0)
                    if( ( (dx == 0) or (dx==-1 && sx==1) or (dx==1 && sx==0) )
                    and ( (dy == 0) or (dy==-1 && sy==1) or (dy==1 && sy==0) )
                    and ( (dz == 0) or (dz==-1 && sz==1) or (dz==1 && sz==0) ) )
                    {
                        morton_t m_suboctant = shift_level(m_neighbor, level_max, level+1); // Morton of first suboctant at level+1;
                        m_suboctant += (sz << IZ) + (sy << IY) + (sx << IX);                // Add current suboctant coordinates
                        m_suboctant = shift_level(m_suboctant, level+1, level_max);         // get morton of suboctant at level_max

                        uint32_t neighbor_rank = find_rank(m_suboctant);
                        if(neighbor_rank != mpi_rank)
                            local_octants_to_send_set.insert({neighbor_rank, iOct});
                    }
                }
            }
        }
    }
    return local_octants_to_send_set;
}

oct_view_t exchange_ghosts_octs(AMRmesh_hashmap& mesh, const oct_view_t& local_octs_coord)
{
    // TODO : avoid deep_copies
    std::shared_ptr<AMRmesh_hashmap> this_ptr(&mesh, [](AMRmesh_hashmap*){});
    muscl_block::GhostCommunicator_kokkos comm_ghosts(this_ptr);

    oct_view_device_t local_octs_coord_device("local_octs_coord_device", octs_coord_id::NUM_OCTS_COORDS, local_octs_coord.extent(1));
    Kokkos::deep_copy(local_octs_coord_device, local_octs_coord);
    oct_view_device_t ghost_octs_coord_device;

    comm_ghosts.exchange_ghosts(local_octs_coord_device, ghost_octs_coord_device);
    
    oct_view_t ghost_octs_coord("ghost_octs_coord", ghost_octs_coord_device.layout());
    Kokkos::deep_copy(ghost_octs_coord, ghost_octs_coord_device);

    return ghost_octs_coord;
}

/// Modifies update distant octants markers in `markers`
void exchange_markers(AMRmesh_hashmap& mesh, AMRmesh_hashmap::markers_device_t& markers)
{
    std::shared_ptr<AMRmesh_hashmap> this_ptr(&mesh, [](AMRmesh_hashmap*){});
    muscl_block::GhostCommunicator_kokkos comm_ghosts(this_ptr);

    uint32_t nbOcts = mesh.getNumOctants();
    uint32_t nbGhosts = mesh.getNumGhosts();

    Kokkos::View<int*, Kokkos::LayoutLeft> ghost_markers("ghost_markers", 0);
    {
        // TODO : either use local_markers to store markers all the time or avoid allocating a view of size mesh.getNumOctants() (avoid the copy)
        AMRmesh_hashmap::markers_device_t markers_new(nbOcts+nbGhosts);
        Kokkos::View<int*, Kokkos::LayoutLeft> local_markers("local_markers", nbOcts);
        // Copy local markers
        Kokkos::parallel_for( "copy markers", markers.capacity(),
                            KOKKOS_LAMBDA(uint32_t i)
        {
            if( markers.valid_at(i) && markers.key_at(i) < nbOcts )
            {
                local_markers[markers.key_at(i)] = markers.value_at(i);
                markers_new.insert( markers.key_at(i), markers.value_at(i) );
            }
        });
        comm_ghosts.exchange_ghosts(local_markers, ghost_markers);

        markers = markers_new;
    }   

    uint32_t ghosts_iOct_begin = mesh.getNumOctants();
    Kokkos::parallel_for( "copy back markers", ghost_markers.size(),
                          KOKKOS_LAMBDA(uint32_t i)
    {
        if( ghost_markers(i) !=0 )
        {
            uint32_t iOct = i+ghosts_iOct_begin;
            int marker = ghost_markers(i);

            uint32_t it = markers.find(iOct);
            if( markers.valid_at(it) )
                markers.value_at(it) = marker;
            else
                markers.insert(iOct, marker);
        }
    });
}

std::vector<morton_t> compute_current_morton_intervals(const AMRmesh_hashmap& mesh, const oct_view_t& local_octs_coord, uint8_t level_max)
{
    int mpi_rank = mesh.getRank();
    int mpi_size = mesh.getNproc();

    // Allgather here ensures that old_morton_interval_begin/end for each process forms is a partition of [0,+inf[
    // When there is no local octant, current interval must be determined by using values of neighbor processes, 
    // otherwise we could just recieve old_morton_interval_end from next rank
    std::vector<morton_t> old_morton_intervals(mpi_size+1);
    old_morton_intervals[mpi_rank] = mesh.getNumOctants() == 0 ? (morton_t)-1 : get_morton_smaller(local_octs_coord, 0, level_max);
    MPI_Allgather(  MPI_IN_PLACE, 0, 0,
                    old_morton_intervals.data(), 1, MPI_INT64_T,
                    MPI_COMM_WORLD);
    old_morton_intervals[mpi_size] = std::numeric_limits<morton_t>::max();
    for(int i=mpi_size-1; i>=0; i--)
    {
        if( old_morton_intervals[i] == (morton_t)-1 )
            old_morton_intervals[i] = old_morton_intervals[i+1];
    }
    return old_morton_intervals;
}

} // namespace

void AMRmesh_hashmap::adapt(bool dummy)
{
    std::cout << "---- Adapt ----" << std::endl;
    std::cout << "Rank " << this->getRank() << ": octs before adapt " << this->getNumOctants() << std::endl;

    uint32_t mpi_rank = hydroSimu::GlobalMpiSession::getRank();
    uint32_t mpi_size = hydroSimu::GlobalMpiSession::getNProc();

    std::shared_ptr<AMRmesh_hashmap> this_sptr(this, [](AMRmesh_hashmap*){});
    LightOctree_hashmap lmesh(this_sptr, level_min, level_max);
    
    {
        oct_view_device_t local_octs_coord_device("local_octs_coord_device", local_octs_coord.layout());
        oct_view_device_t ghost_octs_coord_device("ghost_octs_coord_device", ghost_octs_coord.layout());
        markers_device_t markers_device(markers.capacity()+this->getNumGhosts());
        Kokkos::deep_copy(local_octs_coord_device, local_octs_coord);
        Kokkos::deep_copy(ghost_octs_coord_device, ghost_octs_coord);
        Kokkos::deep_copy(markers_device, markers);

        // Iterate until all processes are 2:1 balanced with up to date ghost markers
        bool modified_global = true;
        while(modified_global)
        {
            // Clean partially coarsened octants and update ghost markers
            adapt_clean(lmesh, local_octs_coord_device, ghost_octs_coord_device, markers_device);
            exchange_markers(*this, markers_device);

            // Iterate until local mesh (+ 1 layer of ghosts) is 2:1 balanced
            bool modified_local = false; // if local process had to make a modification
            for(bool modified_iter=true; modified_iter;)
            {
                // increment markers in every cell (including ghosts) where 2:1 criterion is not respected
                modified_iter = adapt_21balance_step(lmesh, local_octs_coord_device, ghost_octs_coord_device, markers_device);
                if( modified_iter ) 
                {
                    adapt_clean(lmesh, local_octs_coord_device, ghost_octs_coord_device, markers_device);
                    modified_local = true;
                }
            }            

            // modified_global = true if at least one MPI had to make a modification since last exchange_markers()
            MPI_Allreduce(&modified_local, &modified_global, 1, MPI_CXX_BOOL, MPI_LOR, MPI_COMM_WORLD);
        }

        adapt_clean(lmesh, local_octs_coord_device, ghost_octs_coord_device, markers_device);

        markers.clear();
        Kokkos::deep_copy(markers, markers_device);
    }

    oct_view_t new_octs = adapt_apply( local_octs_coord, markers, dim );

    local_octs_coord = new_octs;    


    this->markers = markers_t(this->getNumOctants());

    uint32_t nb_octs = this->getNumOctants();
    MPI_Allreduce( &nb_octs, &total_octs_count, 1, MPI_UINT32_T, MPI::SUM, MPI_COMM_WORLD );

    // Update ghosts metadata
    // TODO : use markers to build new ghost list instead of complete reconstruction
    {
        // Compute morton intervals to discover ghosts
        // Each process writes morton of first octant then communicate
        std::vector<morton_t> morton_intervals = compute_current_morton_intervals(*this, local_octs_coord, level_max);
        
        // Discover ghosts
        std::set< std::pair<int, uint32_t> > local_octants_to_send_set = discover_ghosts(morton_intervals, local_octs_coord, level_max, dim, periodic);
        
        local_octants_to_send.clear();
        for( const auto& p : local_octants_to_send_set )
        {
            local_octants_to_send[p.first].push_back(p.second);
        }

        // Communicate ghosts
        this->ghost_octs_coord = exchange_ghosts_octs( *this, local_octs_coord );
    }

    // Update global index begin
    {
        std::vector<uint32_t> sizes(mpi_size);
        sizes[mpi_rank] = this->getNumOctants();
        MPI_Allgather(MPI_IN_PLACE, 1, MPI_UINT32_T, sizes.data(), 1, MPI_UINT32_T, MPI_COMM_WORLD); 
        this->global_id_begin = std::accumulate(&sizes[0], &sizes[mpi_rank], 0);
    }

    // {
    //     static int iter;        
    //     debug::output_vtk(std::string("after_adapt_iter")+std::to_string(iter), *this);
    //     iter++;
    // }

    std::cout << "Rank " << this->getRank() << ": octs after adapt " << this->getNumOctants() << std::endl;
    std::cout << "Rank " << this->getRank() << ": ghosts after adapt " << this->getNumGhosts() << std::endl;
    std::cout << "---- End Adapt ----" << std::endl;
}

std::map<int, std::vector<uint32_t>> AMRmesh_hashmap::loadBalance(uint8_t level)
{
    std::cout << "---- LoadBalance ----" << std::endl;
    std::cout << "Rank " << this->getRank() << ": octs before LoadBalance " << this->getNumOctants() << std::endl;
    std::cout << "Rank " << this->getRank() << ": ghosts before LoadBalance " << this->getNumGhosts() << std::endl;

    uint32_t mpi_rank = this->getRank();
    uint32_t mpi_size = this->getNproc();

    std::vector<uint32_t> new_intervals(mpi_size+1);
    std::vector<morton_t> morton_intervals(mpi_size+1);
    std::map<int, std::vector<uint32_t>> loadbalance_to_send;
    {
        // Get evenly distributed initial intervals
        int nb_mortons = 0;
        uint32_t iOct_begin = this->global_id_begin;
        uint32_t iOct_end = this->global_id_begin+this->getNumOctants();
        for(uint32_t i=0; i<=mpi_size; i++)
        {
            uint32_t idx = (this->total_octs_count*i)/mpi_size ;
            new_intervals[i] = idx;
            // For each ixd inside old domain, compute morton
            // and fill morton_intervals for this rank
            if( iOct_begin <= idx && idx < iOct_end )
            {
                morton_intervals[i] = get_morton_smaller(local_octs_coord, idx-iOct_begin, level_max); 
                nb_mortons++;
            }
        }

        // allgather morton_intervals
        {
            std::vector<int> morton_recv_count(mpi_size);
            morton_recv_count[mpi_rank] = nb_mortons;
            MPI_Allgather(  MPI_IN_PLACE,0,0,
                            morton_recv_count.data(), 1, MPI_INT, 
                            MPI_COMM_WORLD );
            std::vector<int> morton_recv_displ(mpi_size+1);
            std::partial_sum(morton_recv_count.begin(), morton_recv_count.end(), 
                            ++morton_recv_displ.begin()); 
            assert(morton_recv_displ[mpi_size] == mpi_size);       
            MPI_Allgatherv( MPI_IN_PLACE, 0, 0,
                            morton_intervals.data(), morton_recv_count.data(), morton_recv_displ.data(), MPI_UINT64_T, MPI_COMM_WORLD);
            morton_intervals[0] = 0;
            morton_intervals[mpi_size] = std::numeric_limits<morton_t>::max();
            for(uint32_t rank=0; rank<mpi_size; rank++)
            {
                // Truncate suboctants to keep `level` levels of suboctants compact
                morton_intervals[rank] = (morton_intervals[rank] >> (3*level)) << (3*level);
            }
            assert(morton_intervals[mpi_rank] <= morton_intervals[mpi_rank+1] );
        }

        std::cout << "Rank " << this->getRank() << ": new morton interval [" << morton_intervals[mpi_rank] << ", " << morton_intervals[mpi_rank+1] << "[" << std::endl;

        // Compute `new_intervals` corresponding to `morton_intervals`
        {
            uint64_t old_morton_interval_begin, old_morton_interval_end;
            {
                auto old_morton_intervals = compute_current_morton_intervals(*this, local_octs_coord, level_max);
                old_morton_interval_begin = old_morton_intervals[mpi_rank];
                old_morton_interval_end = old_morton_intervals[mpi_rank+1];
            }            

            std::cout << "Rank " << this->getRank() << ": old morton interval [" << old_morton_interval_begin << ", " << old_morton_interval_end << "[" << std::endl;

            int nb_local_pivots=0;
            for(uint32_t rank=0; rank<=mpi_size; rank++)
            {
                if( old_morton_interval_begin <= morton_intervals[rank] && morton_intervals[rank] < old_morton_interval_end ) // Determine if pivot is inside of local process
                {
                    // find first local octant with morton >= morton_interval[rank]
                    uint32_t pivot = this->getNumOctants();
                    // TODO use binary search to find pivot
                    for(uint32_t iOct=0; iOct<this->getNumOctants(); iOct++)
                    {
                        uint64_t morton = get_morton_smaller(local_octs_coord, iOct, level_max);
                        if( morton >= morton_intervals[rank]  )
                        {
                            pivot = iOct;
                            break;
                        }
                    }
                    nb_local_pivots++;
                    new_intervals[rank] = pivot+this->global_id_begin;
                }
            }

            // Compute and allgather pivots
            std::vector<int> pivots_recv_count(mpi_size);
            pivots_recv_count[mpi_rank] = nb_local_pivots;
            MPI_Allgather(  MPI_IN_PLACE,0,0,
                            pivots_recv_count.data(), 1, MPI_INT, 
                            MPI_COMM_WORLD );
            std::vector<int> pivots_recv_displ(mpi_size+1);
            std::partial_sum(pivots_recv_count.begin(), pivots_recv_count.end(), 
                            ++pivots_recv_displ.begin()); 
            assert(pivots_recv_displ[mpi_size] == mpi_size);       
            MPI_Allgatherv( MPI_IN_PLACE, 0, 0,
                            new_intervals.data(), pivots_recv_count.data(), pivots_recv_displ.data(), MPI_UINT32_T, MPI_COMM_WORLD);
            new_intervals[mpi_size] = this->total_octs_count;

        }
        std::cout << "Rank " << this->getRank() << ": iOct interval [" << new_intervals[mpi_rank] << ", " << new_intervals[mpi_rank+1] << "[" << std::endl;

        // List octants to exchange
        for( uint32_t rank=0; rank<mpi_size; rank++ )
        {
            int32_t to_send_begin = std::max((int32_t)new_intervals[rank]-(int32_t)this->global_id_begin, 0);
            int32_t to_send_end   = std::min((int32_t)new_intervals[rank+1]-(int32_t)this->global_id_begin, (int32_t)this->getNumOctants());
            
            if( to_send_begin < to_send_end )
            {
                std::vector<uint32_t> to_send_rank(to_send_end-to_send_begin);
                for(int32_t i=0; i<to_send_end-to_send_begin; i++)
                {
                    to_send_rank[i] = i+to_send_begin;
                }
                loadbalance_to_send[rank] = to_send_rank;
            }
        }

        // Exchange 
        {   //TODO : avoid CPU/GPU transfers
            muscl_block::GhostCommunicator_kokkos loadbalance_communicator( loadbalance_to_send );
            oct_view_device_t local_octs_coord_old( "local_octs_coord_old", local_octs_coord.layout() );
            oct_view_device_t local_octs_coord_new;
            Kokkos::deep_copy(local_octs_coord_old , this->local_octs_coord );
            loadbalance_communicator.exchange_ghosts(local_octs_coord_old, local_octs_coord_new);
            Kokkos::realloc(this->local_octs_coord, local_octs_coord_new.layout());
            Kokkos::deep_copy( this->local_octs_coord, local_octs_coord_new );
        }

        this->global_id_begin = new_intervals[mpi_rank];

        if( getNumOctants() != 0 )
        {
        std::cout << "Rank " << this->getRank() << ": actual morton interval [" << get_morton_smaller(local_octs_coord, 0, level_max) << ", " << get_morton_smaller(local_octs_coord, getNumOctants()-1, level_max) << "]" << std::endl;
        assert( get_morton_smaller(local_octs_coord, 0, level_max) >= morton_intervals[mpi_rank] );
        assert( get_morton_smaller(local_octs_coord, getNumOctants()-1, level_max) < morton_intervals[mpi_rank+1] );
        }
        else 
        {
            std::cout << "Rank " << this->getRank() << ": actual morton interval [EMPTY]" << std::endl;
            std::cout << "WARNING : Rank has 0 octant, this is probably not okay" << std::endl;
        }

        sequential_mesh = false;
    }

    std::set< std::pair<int, uint32_t> > local_octants_to_send_set = discover_ghosts(morton_intervals, local_octs_coord, level_max, dim, periodic);

    local_octants_to_send.clear();
    for( const auto& p : local_octants_to_send_set )
    {
        local_octants_to_send[p.first].push_back(p.second);
    }

    this->ghost_octs_coord = exchange_ghosts_octs( *this, local_octs_coord );
    markers = markers_t(this->getNumOctants());

    // {
    //     static int iter;        
    //     debug::output_vtk(std::string("after_loadbalance_iter")+std::to_string(iter), *this);
    //     iter++;
    // }

    std::cout << "Rank " << this->getRank() << ": octs after loadbalance " << this->getNumOctants() << std::endl;
    std::cout << "Rank " << this->getRank() << ": ghosts after loadbalance " << this->getNumGhosts() << std::endl;
    std::cout << "---- End LoadBalance----" << std::endl;

    assert(this->getNumOctants() > 0); // Process cannot be empty

    return loadbalance_to_send;
}

namespace{
template < typename View_t >
void AMRmesh_hashmap_loadBalance( AMRmesh_hashmap& mesh, int compact_levels, View_t& userData )
{
    auto octs_to_exchange = mesh.loadBalance(compact_levels);
    muscl_block::GhostCommunicator_kokkos lb_comm(octs_to_exchange);
    View_t userData_new;
    lb_comm.exchange_ghosts(userData, userData_new);
    userData = userData_new;
}
} // namespace

void AMRmesh_hashmap::loadBalance_userdata( int compact_levels, DataArrayBlock& userData )
{
    AMRmesh_hashmap_loadBalance(*this, compact_levels, userData);
}

void AMRmesh_hashmap::loadBalance_userdata( int compact_levels, DataArray& userData )
{
    AMRmesh_hashmap_loadBalance(*this, compact_levels, userData);
}

const std::map<int, std::vector<uint32_t>>& AMRmesh_hashmap::getBordersPerProc() const
{
    return local_octants_to_send;
}

}

#endif // DYABLO_USE_GPU_MESH