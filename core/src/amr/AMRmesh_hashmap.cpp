#include "AMRmesh_hashmap.h"

#include <numeric>
#include <set>

#include "mpi.h"

#include "amr/LightOctree.h"
#include "morton_utils.h"
//#include "utils/io/AMRMesh_output_vtk.h"

#include "mpi/ViewCommunicator.h"

#include "UserData.h"

namespace dyablo{

AMRmesh_hashmap::AMRmesh_hashmap( int dim, int balance_codim, const std::array<bool,3>& periodic, level_t level_min, level_t level_max, const MpiComm& mpi_comm )
        : dim(dim), periodic(periodic), markers(1), level_min(level_min), level_max(level_max), mpi_comm(mpi_comm)
{
    DYABLO_ASSERT_HOST_RELEASE(dim == 2 || dim == 3, "Invalid ndim : " << dim);
    DYABLO_ASSERT_HOST_RELEASE(balance_codim == dim, "Invalid balance_codim : " << balance_codim);
    this->total_octs_count = 1;
    this->global_id_begin = 0;

    int nb_octs = ( this->getRank() == 0 ) ? 1 : 0;
    this->local_octs_coord = oct_view_t("local_octs_coord", NUM_OCTS_COORDS, nb_octs);
    this->ghost_octs_coord = oct_view_t("ghost_octs_coord", NUM_OCTS_COORDS, 0);

    // Refine to level_min
    for (uint8_t level=0; level<level_min; ++level)
    {
        this->adaptGlobalRefine(); 
    } 
    this->loadBalance();
}

void AMRmesh_hashmap::adaptGlobalRefine()
{
    DYABLO_ASSERT_HOST_RELEASE( this->sequential_mesh, "No global refine after scattering" );

    uint32_t nb_subocts = std::pow(2, dim);
    uint32_t new_octs_count = this->getNumOctants()*nb_subocts;
    oct_view_t new_octs("local_octs_coord",  NUM_OCTS_COORDS, new_octs_count);
    for(uint32_t iOct_old=0; iOct_old<this->getNumOctants(); iOct_old++)
    {
        coord_t x = local_octs_coord(IX, iOct_old);
        coord_t y = local_octs_coord(IY, iOct_old);
        coord_t z = local_octs_coord(IZ, iOct_old);
        coord_t level = local_octs_coord(LEVEL, iOct_old);

        for(int iz=0; iz<(dim-1); iz++)
            for(int iy=0; iy<2; iy++)
                for(int ix=0; ix<2; ix++)
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

    uint64_t nb_octs = this->getNumOctants();
    mpi_comm.MPI_Allreduce( &nb_octs, &total_octs_count, 1, MpiComm::MPI_Op_t::SUM );
    pmesh_epoch++;
}

void AMRmesh_hashmap::setMarker(uint32_t iOct, int marker)
{
    if(marker==0) return;
    auto inserted = this->markers.insert( iOct, marker );

    if(inserted.existing())
        markers.value_at(inserted.index()) = marker;
    
    DYABLO_ASSERT_HOST_DEBUG(!inserted.failed(), "Insertion in Kokkos::UnorderedMap failed");
}

void AMRmesh_hashmap::setMarkers( const Kokkos::View<int*>& ioct_markers )
{
    uint32_t nbOcts = this->getNumOctants();
    DYABLO_ASSERT_HOST_RELEASE( nbOcts == ioct_markers.size(), "Markers count mismatch nbOcts=" << nbOcts << " != markers.size()=" << markers.size()  );

    markers_device_t markers_device( nbOcts );

    Kokkos::parallel_for( "AMRmesh_pablo::setMarkers", nbOcts,
                        KOKKOS_LAMBDA(uint32_t iOct)
    {
        int marker = ioct_markers(iOct);
        if( marker!=0 )
        {
            auto inserted = markers_device.insert( iOct, marker );
            if(inserted.existing())
                markers_device.value_at(inserted.index()) = marker;
            DYABLO_ASSERT_KOKKOS_DEBUG(!inserted.failed(), "Insertion in Kokkos::UnorderedMap failed");
        }
    });

    Kokkos::deep_copy(this->markers, markers_device);
}

namespace {

using morton_t = uint64_t;
using coord_t = AMRmesh_hashmap::coord_t;
using level_t = AMRmesh_hashmap::level_t;
using oct_view_t = AMRmesh_hashmap::oct_view_t;
using oct_view_device_t = AMRmesh_hashmap::oct_view_device_t;
using octs_coord_id = AMRmesh_hashmap::octs_coord_id;
using markers_t = AMRmesh_hashmap::markers_t;
using markers_device_t = AMRmesh_hashmap::markers_device_t;
using OctantIndex = LightOctree_base::OctantIndex;

KOKKOS_INLINE_FUNCTION
Kokkos::Array<coord_t, 3> oct_getpos(  const oct_view_device_t& local_octs_coord, 
                                        const oct_view_device_t& ghost_octs_coord,
                                        const OctantIndex& iOct)
{
    Kokkos::Array<coord_t, 3> pos;
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
level_t oct_getlevel(  const oct_view_device_t& local_octs_coord, 
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
    DYABLO_ASSERT_KOKKOS_DEBUG(!iOct.isGhost, "Cannot be a ghost : we may not have all the siblings");

    uint32_t nbOcts = lmesh.getNumOctants();
    //Position in parent octant
    auto pos = oct_getpos(local_octs_coord, ghost_octs_coord, iOct);
    level_t level = oct_getlevel(local_octs_coord, ghost_octs_coord, iOct);
    
    level_t target_level = level-1;

    int sx = (pos[IX]%2==0)?1:-1;
    int sy = (pos[IY]%2==0)?1:-1;
    int sz = (pos[IZ]%2==0)?1:-1;

    int ndim = lmesh.getNdim();

    for(int z=0; z<(ndim-1); z++)
        for(int y=0; y<2; y++)
            for(int x=0; x<2; x++)
            {
                if( x!=0 || y!=0 || z!=0 )
                {
                    auto ns = lmesh.findNeighbors(iOct,{(int8_t)(sx*x),(int8_t)(sy*y),(int8_t)(sz*z)});
                    DYABLO_ASSERT_KOKKOS_DEBUG(ns.size()>0, "Neighbor not found");
                    const OctantIndex& iOct_other = ns[0];
                    level_t target_level_n = oct_getlevel(local_octs_coord, ghost_octs_coord, iOct_other);;
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
        DYABLO_ASSERT_KOKKOS_DEBUG( marker==-1 || marker==1, "Invalid marker" );
        // Remove partial octants coarsening        
        if(marker==1 ||  ( marker==-1 && is_full_coarsening(lmesh, local_octs_coord, ghost_octs_coord, markers, iOct)))
        { 
            [[maybe_unused]] auto res = markers_out.insert(iOct_local, marker);
            DYABLO_ASSERT_KOKKOS_DEBUG(res.success(), "Could not insert in Kokkos::UnorderdMap");
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
    DYABLO_ASSERT_KOKKOS_DEBUG(!iOct.isGhost, "need_refine_to_balance : iOct cannot be ghost");

    uint32_t nbOcts = lmesh.getNumOctants();

    level_t new_level = oct_getlevel(local_octs_coord, ghost_octs_coord, iOct);
    uint32_t idm = markers.find(OctantIndex::OctantIndex_to_iOctLocal(iOct, nbOcts));
    if( markers.valid_at(idm) )
        new_level += markers.value_at(idm);

    int nz_max = lmesh.getNdim() == 2? 0:1;
    for( int nz=-nz_max; nz<=nz_max; nz++ )
        for( int ny=-1; ny<=1; ny++ )
            for( int nx=-1; nx<=1; nx++ )
            {
                if( nx!=0 || ny!=0 || nz!=0 )
                {
                    LightOctree_base::NeighborList neighbors = 
                        lmesh.findNeighbors(iOct, {(int8_t)nx,(int8_t)ny,(int8_t)nz});
                    for(uint32_t n=0; n<neighbors.size(); n++)
                    {
                        const OctantIndex& iOct_neighbor = neighbors[n];
                        level_t new_level_neighbor = oct_getlevel(local_octs_coord, ghost_octs_coord, iOct_neighbor);
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

            DYABLO_ASSERT_KOKKOS_DEBUG(marker<=1, "Marker > 1 after need_refine update");
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
                coord_t ix = local_octs_coord(octs_coord_id::IX, iOct_old);
                coord_t iy = local_octs_coord(octs_coord_id::IY, iOct_old);
                coord_t iz = local_octs_coord(octs_coord_id::IZ, iOct_old);
                if( ix%2==0 && iy%2==0 && iz%2==0 )
                    n_new_octants = 1;
                else 
                    n_new_octants = 0;
            }
            else
            {
                n_new_octants = 0;
                DYABLO_ASSERT_KOKKOS_DEBUG(false, "Only +1 -1 allowed in markers");
            }
        }
        
        if(final && n_new_octants>0)
        {
            coord_t ix = local_octs_coord(octs_coord_id::IX, iOct_old);
            coord_t iy = local_octs_coord(octs_coord_id::IY, iOct_old);
            coord_t iz = local_octs_coord(octs_coord_id::IZ, iOct_old);
            level_t level = local_octs_coord(octs_coord_id::LEVEL, iOct_old);

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
        DYABLO_ASSERT_KOKKOS_DEBUG(iOct_new <= new_size, "Too many new octants during adapt::apply scan");
    });

    return local_octs_coord_out;
}

morton_t shift_level( morton_t m, level_t from_level, level_t to_level  )
{
    int16_t ldiff = to_level-from_level;
    if(ldiff > 0)
        return m << 3*ldiff;
    else 
        return m >> -3*ldiff;
}

// Compute morton at level_max;
morton_t get_morton_smaller(coord_t ix, coord_t iy, coord_t iz, level_t level, level_t level_max)
{
    morton_t morton = compute_morton_key( ix,iy,iz );
    return shift_level(morton, level, level_max);
};

morton_t get_morton_smaller(const oct_view_t& local_octs_coord, uint32_t idx, level_t level_max)
{
    coord_t ix = local_octs_coord(octs_coord_id::IX, idx);
    coord_t iy = local_octs_coord(octs_coord_id::IY, idx);
    coord_t iz = local_octs_coord(octs_coord_id::IZ, idx);
    level_t level = local_octs_coord(octs_coord_id::LEVEL, idx);
    return get_morton_smaller(ix,iy,iz,level,level_max);
}

std::set< std::pair<int, uint32_t> > discover_ghosts(
        const std::vector<morton_t>& morton_intervals,
        const AMRmesh_hashmap::oct_view_t& local_octs_coord,
        level_t level_max, uint8_t ndim, const std::array<bool,3>& periodic,
        const MpiComm& mpi_comm)
{
    uint32_t mpi_rank = mpi_comm.MPI_Comm_rank();

    auto find_rank = [&](morton_t morton)
    {
        // upper_bound : first verifying value > morton
        auto it = std::upper_bound( morton_intervals.begin(), morton_intervals.end(), morton );
        // neighbor_rank: last interval value <= morton
        uint32_t neighbor_rank = it - morton_intervals.begin() - 1;
        DYABLO_ASSERT_HOST_DEBUG( (int)neighbor_rank < mpi_comm.MPI_Comm_size(), "find_rank : rank out of range" );
        DYABLO_ASSERT_HOST_DEBUG( morton_intervals[neighbor_rank] <= morton && morton_intervals[neighbor_rank+1] > morton,
            "find_rank : morton out of interval : " << morton << " not in [" << morton_intervals[neighbor_rank] << "," << morton_intervals[neighbor_rank+1] << "[" );
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
        coord_t ix = local_octs_coord(octs_coord_id::IX, iOct);
        coord_t iy = local_octs_coord(octs_coord_id::IY, iOct);
        coord_t iz = local_octs_coord(octs_coord_id::IZ, iOct);
        level_t level = local_octs_coord(octs_coord_id::LEVEL, iOct);
        coord_t max_i = std::pow(2, level);

        DYABLO_ASSERT_HOST_DEBUG(find_rank(get_morton_smaller(ix,iy,iz,level,level_max)) == mpi_rank, "find_rank() should return local rank for local octants");

        int dz_max = (ndim == 2)? 0:1;
        for( int dz=-dz_max; dz<=dz_max; dz++ )
        for( int dy=-1; dy<=1; dy++ )
        for( int dx=-1; dx<=1; dx++ )
        if(   (dx!=0 || dy!=0 || dz!=0)
           && (periodic[IX] || ( /*0<=ix+dx &&*/ ix+dx<max_i ))
           && (periodic[IY] || ( /*0<=iy+dy &&*/ iy+dy<max_i ))
           && (periodic[IZ] || ( /*0<=iz+dz &&*/ iz+dz<max_i )) )
        {
            morton_t m_neighbor = get_morton_smaller((ix+dx+max_i)%max_i,(iy+dy+max_i)%max_i,(iz+dz+max_i)%max_i,level,level_max);
            // Find the first rank where morton_intervals[rank] >= m_neighbor
            uint32_t neighbor_rank = find_rank(m_neighbor);

            // Verify that the whole same-size virtual neighbor is owned by neighbor_rank
            // i.e : last suboctant of same-size neighbor is owned by the same MPI
            morton_t m_next = shift_level(m_neighbor, level_max, level) + 1;
                     m_next = shift_level(m_next, level, level_max);
            if( level<level_max && m_next < morton_intervals[neighbor_rank+1] )
            {// Neighbors are owned by only one MPI
                if(neighbor_rank != mpi_rank)
                    local_octants_to_send_set.insert({neighbor_rank, iOct});
            }
            else
            {// Neighbors may be scattered between multiple MPIs              
                morton_t m_suboctant_origin = shift_level(m_neighbor, level_max, level+1); // Morton of first suboctant at level+1;
                // Apply offset to get a neighbor
                m_suboctant_origin += (dx == -1) << IX; // Other half of suboctant if left of original cell
                m_suboctant_origin += (dy == -1) << IY;
                m_suboctant_origin += (dz == -1) << IZ;
                // Iterate over neighbor suboctants
                int sx_max = (dx==0); // constrained to the same plane as origin if offset in this direction
                int sy_max = (dy==0);
                int sz_max = (ndim==2) ? 0 : (dz==0);
                for( int16_t sz=0; sz<=sz_max; sz++ )
                for( int16_t sy=0; sy<=sy_max; sy++ )
                for( int16_t sx=0; sx<=sx_max; sx++ )
                {
                    morton_t m_suboctant = m_suboctant_origin + (sz << IZ) + (sy << IY) + (sx << IX); // Add current suboctant coordinates
                    m_suboctant = shift_level(m_suboctant, level+1, level_max);         // get morton of suboctant at level_max

                    uint32_t neighbor_rank = find_rank(m_suboctant);
                    if(neighbor_rank != mpi_rank)
                        local_octants_to_send_set.insert({neighbor_rank, iOct});
                }
            }
        }
    }
    return local_octants_to_send_set;
}

oct_view_t exchange_ghosts_octs(AMRmesh_hashmap& mesh, const oct_view_t& local_octs_coord)
{
    // TODO : avoid deep_copies
    ViewCommunicator comm_ghosts(mesh.getBordersPerProc());

    oct_view_device_t local_octs_coord_device("local_octs_coord_device", octs_coord_id::NUM_OCTS_COORDS, local_octs_coord.extent(1));
    Kokkos::deep_copy(local_octs_coord_device, local_octs_coord);
    oct_view_device_t ghost_octs_coord_device("ghost_octs_coord_device", octs_coord_id::NUM_OCTS_COORDS, comm_ghosts.getNumGhosts());

    comm_ghosts.exchange_ghosts<1>(local_octs_coord_device, ghost_octs_coord_device);
    
    oct_view_t ghost_octs_coord("ghost_octs_coord", ghost_octs_coord_device.layout());
    Kokkos::deep_copy(ghost_octs_coord, ghost_octs_coord_device);

    return ghost_octs_coord;
}

/// Modifies update distant octants markers in `markers`
void exchange_markers(AMRmesh_hashmap& mesh, AMRmesh_hashmap::markers_device_t& markers)
{
    ViewCommunicator comm_ghosts(mesh.getBordersPerProc());

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
        Kokkos::realloc(ghost_markers, comm_ghosts.getNumGhosts() );
        comm_ghosts.exchange_ghosts<0>(local_markers, ghost_markers);

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

std::vector<morton_t> compute_current_morton_intervals(const AMRmesh_hashmap& mesh, const oct_view_t& local_octs_coord, level_t level_max)
{
    int mpi_size = mesh.getNproc();
    const MpiComm& mpi_comm = mesh.getMpiComm();

    // Allgather here ensures that old_morton_interval_begin/end for each process forms is a partition of [0,+inf[
    // When there is no local octant, current interval must be determined by using values of neighbor processes, 
    // otherwise we could just recieve old_morton_interval_end from next rank
    uint64_t morton_first = mesh.getNumOctants() == 0 ? 0 : get_morton_smaller(local_octs_coord, 0, level_max);
    std::vector<morton_t> morton_intervals(mpi_size+1);
    mpi_comm.MPI_Allgather( &morton_first, morton_intervals.data(), 1 );
    DYABLO_ASSERT_HOST_RELEASE( morton_intervals[0] == 0, "First interval doesn,'t start at 0" );
    morton_intervals[mpi_size] = std::numeric_limits<morton_t>::max();
    for(int i=mpi_size-1; i>=1; i--)
    {
        if( morton_intervals[i]==0 )
            morton_intervals[i] = morton_intervals[i+1];
        DYABLO_ASSERT_HOST_RELEASE( morton_intervals[i] <= morton_intervals[i+1], 
        "morton interval upper bound is smaller than lower bound. Rank " << i << " : [" << morton_intervals[i] << "," << morton_intervals[i+1] << "[" );
    }   

    return morton_intervals;
}

} // namespace

void AMRmesh_hashmap::adapt(bool dummy)
{
    uint32_t mpi_rank = getRank();
    uint32_t mpi_size = getNproc();
    
    {
        LightOctree_hashmap lmesh(this, level_min, level_max);

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
            mpi_comm.MPI_Allreduce( &modified_local, &modified_global, 1, MpiComm::MPI_Op_t::LOR );
        }

        adapt_clean(lmesh, local_octs_coord_device, ghost_octs_coord_device, markers_device);

        markers.clear();
        Kokkos::deep_copy(markers, markers_device);
    }

    oct_view_t new_octs = adapt_apply( local_octs_coord, markers, dim );

    local_octs_coord = new_octs;    


    this->markers = markers_t(this->getNumOctants());

    uint64_t nb_octs = this->getNumOctants();
    mpi_comm.MPI_Allreduce( &nb_octs, &total_octs_count, 1, MpiComm::MPI_Op_t::SUM );

    // Update ghosts metadata
    // TODO : use markers to build new ghost list instead of complete reconstruction
    {
        // Compute morton intervals to discover ghosts
        // Each process writes morton of first octant then communicate
        std::vector<morton_t> morton_intervals = compute_current_morton_intervals(*this, local_octs_coord, level_max);
        
        // Discover ghosts
        std::set< std::pair<int, uint32_t> > local_octants_to_send_set = discover_ghosts(morton_intervals, local_octs_coord, level_max, dim, periodic, mpi_comm);
        
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
        uint32_t size_local = this->getNumOctants();
        std::vector<uint32_t> sizes(mpi_size);
        mpi_comm.MPI_Allgather(&size_local, sizes.data(), 1); 
        this->global_id_begin = std::accumulate(&sizes[0], &sizes[mpi_rank], 0);
    }

    pmesh_epoch++;

    // {
    //     static int iter;        
    //     debug::output_vtk(std::string("after_adapt_iter")+std::to_string(iter), *this);
    //     iter++;
    // }
}

std::map<int, std::vector<uint32_t>> AMRmesh_hashmap::loadBalance(level_t level)
{
    uint32_t mpi_rank = this->getRank();
    uint32_t mpi_size = this->getNproc();

    std::vector<uint64_t> new_intervals(mpi_size+1); // First global index for rank i
    std::vector<morton_t> morton_intervals(mpi_size+1);
    std::map<int, std::vector<uint32_t>> loadbalance_to_send;
    {
        // Get evenly distributed initial intervals
        int nb_mortons = 0;
        uint64_t iOct_begin = this->global_id_begin;
        uint64_t iOct_end = this->global_id_begin+this->getNumOctants();
        for(uint32_t i=0; i<=mpi_size; i++)
        {
            uint64_t idx = (this->total_octs_count*i)/mpi_size ;
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
            mpi_comm.MPI_Allgatherv_inplace( morton_intervals.data(), nb_mortons );
            morton_intervals[0] = 0;
            morton_intervals[mpi_size] = std::numeric_limits<morton_t>::max();
            for(uint32_t rank=0; rank<mpi_size; rank++)
            {
                // Truncate suboctants to keep `level` levels of suboctants compact
                morton_intervals[rank] = (morton_intervals[rank] >> (3*level)) << (3*level);
            }
            DYABLO_ASSERT_HOST_RELEASE( morton_intervals[mpi_rank] <= morton_intervals[mpi_rank+1], 
        "morton interval upper bound is smaller than lower bound. Rank " << mpi_rank << " : [" << morton_intervals[mpi_rank] << "," << morton_intervals[mpi_rank+1] << "[" );
        }

        std::cout << "Rank " << this->getRank() << ": new morton interval [" << morton_intervals[mpi_rank] << ", " << morton_intervals[mpi_rank+1] << "[" << std::endl;

        // Compute `new_intervals` corresponding to `morton_intervals`
        {
            morton_t old_morton_interval_begin, old_morton_interval_end;
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

            mpi_comm.MPI_Allgatherv_inplace( new_intervals.data(), nb_local_pivots );
            new_intervals[mpi_size] = this->total_octs_count;

        }
        std::cout << "Rank " << this->getRank() << ": iOct interval [" << new_intervals[mpi_rank] << ", " << new_intervals[mpi_rank+1] << "[" << std::endl;

        // List octants to exchange
        for( uint32_t rank=0; rank<mpi_size; rank++ )
        {
	  int32_t to_send_begin = std::max((int64_t)new_intervals[rank]-(int64_t)this->global_id_begin, (int64_t)0);
            int32_t to_send_end   = std::min((int64_t)new_intervals[rank+1]-(int64_t)this->global_id_begin, (int64_t)this->getNumOctants());
            
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
            ViewCommunicator loadbalance_communicator( loadbalance_to_send );
            oct_view_device_t local_octs_coord_old( "local_octs_coord_old", local_octs_coord.layout() );
            oct_view_device_t local_octs_coord_new( "local_octs_coord_new", octs_coord_id::NUM_OCTS_COORDS, loadbalance_communicator.getNumGhosts() );
            Kokkos::deep_copy(local_octs_coord_old , this->local_octs_coord );
            loadbalance_communicator.exchange_ghosts<1>(local_octs_coord_old, local_octs_coord_new);
            Kokkos::realloc(this->local_octs_coord, local_octs_coord_new.layout());
            Kokkos::deep_copy( this->local_octs_coord, local_octs_coord_new );
        }

        this->global_id_begin = new_intervals[mpi_rank];

        if( getNumOctants() != 0 )
        {
            std::cout << "Rank " << this->getRank() << ": actual morton interval [" << get_morton_smaller(local_octs_coord, 0, level_max) << ", " << get_morton_smaller(local_octs_coord, getNumOctants()-1, level_max) << "]" << std::endl;
            DYABLO_ASSERT_HOST_RELEASE( get_morton_smaller(local_octs_coord, 0, level_max) >= morton_intervals[mpi_rank], 
                "First octant not in morton interval. Rank " << mpi_rank << ", morton " << get_morton_smaller(local_octs_coord, 0, level_max) << " : [" << morton_intervals[mpi_rank] << "," << morton_intervals[mpi_rank+1] << "[" );
            DYABLO_ASSERT_HOST_RELEASE( get_morton_smaller(local_octs_coord, getNumOctants()-1, level_max) < morton_intervals[mpi_rank+1], 
                "Last octant not in morton interval. Rank " << mpi_rank << ", morton " << get_morton_smaller(local_octs_coord, getNumOctants()-1, level_max) << " : [" << morton_intervals[mpi_rank] << "," << morton_intervals[mpi_rank+1] << "[" );
        }
        else 
        {
            std::cout << "Rank " << this->getRank() << ": actual morton interval [EMPTY]" << std::endl;
            std::cout << "WARNING : Rank has 0 octant, this is probably not okay" << std::endl;
        }

        sequential_mesh = false;
    }

    std::set< std::pair<int, uint32_t> > local_octants_to_send_set = discover_ghosts(morton_intervals, local_octs_coord, level_max, dim, periodic, mpi_comm);

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

    DYABLO_ASSERT_HOST_RELEASE(this->getNumOctants() > 0, "Rank " << mpi_rank << " has 0 octant");

    pmesh_epoch++;

    return loadbalance_to_send;
}

void AMRmesh_hashmap::loadBalance_userdata( int compact_levels, UserData& U )
{
    auto octs_to_exchange = loadBalance(compact_levels);
    ViewCommunicator lb_comm(octs_to_exchange);
    U.exchange_loadbalance(lb_comm);
}

const std::map<int, std::vector<uint32_t>>& AMRmesh_hashmap::getBordersPerProc() const
{
    return local_octants_to_send;
}

}