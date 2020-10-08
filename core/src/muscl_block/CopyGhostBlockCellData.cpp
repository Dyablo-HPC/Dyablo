#include "muscl_block/CopyGhostBlockCellData.h"

namespace dyablo
{
namespace muscl_block
{

CopyGhostBlockCellDataFunctor::CopyGhostBlockCellDataFunctor(
    std::shared_ptr<AMRmesh> pmesh, HydroParams params, id2index_t fm,
    blockSize_t blockSizes, uint32_t ghostWidth, uint32_t nbOctsPerGroup,
    DataArrayBlock U, DataArrayBlock U_ghost, DataArrayBlock Ugroup,
    uint32_t iGroup, InterfaceFlags interface_flags) :
  pmesh(pmesh),
  params(params),
  fm(fm),
  blockSizes(blockSizes),
  ghostWidth(ghostWidth),
  nbOctsPerGroup(nbOctsPerGroup),
  U(U),
  U_ghost(U_ghost),
  Ugroup(Ugroup),
  iGroup(iGroup),
  interface_flags(interface_flags)
{
  // in 2d, bz and bz_g are not used
  blockSizes[IZ] = (params.dimType == THREE_D) ? blockSizes[IZ] : 1;
  bx_g           = blockSizes[IX] + 2 * ghostWidth;
  by_g           = blockSizes[IY] + 2 * ghostWidth;
  bz_g = (params.dimType == THREE_D) ? blockSizes[IZ] + 2 * ghostWidth : 1;

  // Gravity
  copy_gravity = (params.gravity_type & GRAVITY_FIELD);
  ndim         = (params.dimType == THREE_D ? 3 : 2);
}

void CopyGhostBlockCellDataFunctor::apply(
    std::shared_ptr<AMRmesh> pmesh, ConfigMap configMap, HydroParams params,
    id2index_t fm, blockSize_t blockSizes, uint32_t ghostWidth,
    uint32_t nbOctsPerGroup, DataArrayBlock U, DataArrayBlock U_ghost,
    DataArrayBlock Ugroup, uint32_t iGroup, InterfaceFlags interface_flags)
{
  CopyGhostBlockCellDataFunctor functor(pmesh,
                                        params,
                                        fm,
                                        blockSizes,
                                        ghostWidth,
                                        nbOctsPerGroup,
                                        U,
                                        U_ghost,
                                        Ugroup,
                                        iGroup,
                                        interface_flags);

  // using kokkos team execution policy
  uint32_t nbTeams_ = configMap.getInteger("amr", "nbTeams", 16);
  functor.nbTeams   = nbTeams_;
  // create execution policy
  team_policy_t policy(nbTeams_,
                       Kokkos::AUTO() /* team size chosen by kokkos */);

  // launch computation (parallel kernel)
  Kokkos::parallel_for("dyablo::muscl_block::CopyGhostBlockCellDataFunctor",
                       policy,
                       functor);
}

namespace{

using Functor = CopyGhostBlockCellDataFunctor;

/// replaces std::max({v0,v1,...})
uint32_t max_n(uint32_t v) { return v; }
template <typename T0, typename... Ts>
T0 max_n(T0 v0, T0 v1, Ts... vs)
{
  return v0 > max_n(v1, vs...) ? v0 : max_n(v1, vs...);
}

//! Signed coordinates for cells
using coord_g_t = Kokkos::Array<int32_t, 3>;

/**
 * Compute index in group data (ghosted) corresponding to cell at position `coords`
 * 
 * @param coords cell coordinates inside octant (non-ghosted : (-1,0,0) is in left border)
 * @param bSizes Number of cells in each dimension inside octant
 * @param ghostWidth ghost width in group data
 **/
template <int ndim>
uint32_t coord_g_to_index(  coord_g_t   coords, 
                            blockSize_t bSizes,
                            uint32_t    ghostWidth)
{
    const int32_t i = coords[IX] + ghostWidth;
    const int32_t j = coords[IY] + ghostWidth;
    const int32_t k = coords[IZ] + ghostWidth;

    const uint32_t bx = bSizes[IX];
    const uint32_t by = bSizes[IY];
    //const uint32_t& bz = bSizes[IZ];

    uint32_t res = ndim == 2 ?
        i + (bx+2*ghostWidth)*j : 
        i + (bx+2*ghostWidth)*j + (bx+2*ghostWidth)*(by+2*ghostWidth)*k;

    return res;
}

//! Cell positions inside local and neighbor octants
struct get_pos_t{
    bool in_domain;  //! true if index is inside local domain (coords invalid if false)
    coord_g_t pos_in_local;  //! position in local block ( will be in ghost zone (-1,0,0) is left ghost )
    coord_t pos_in_neighbor;  //! position in neighbor block (cannot be outside neighbor)
    coord_g_t neighbor_pos; //! position of neighbor octant relative to local octant ((-1,0,0) is left neighbor)
};

/**
 * @brief Get position of local cell into local and neighbor octant
 * 
 * @tparam ndim number of dimensions (2 or 3)
 * @tparam dir ghost direction. ex : if dir==DIR_X in 2D, resulting positions cover left AND right ghost cells
 * @param f Ghost copy functor (to fetch simulation parameters)
 * @param index index between 0 and the number of cells in border to be mapped to a position inside border
 * 
 * Translates an index to a position inside the border along the specified axis. 
 * Relative position of neighbor octant and cell position inside neighbor octant is also computed in this function
 * 
 * @note no corners are included when dir=DIR_X, edges along Z are included when dir=DIR_Y, whole Z faces are included when DIR_Z
 * ex : In 2D with a 2x2 block, x are ghosts in DIR_X, y are ghosts in DIR_Y and o are inner cells
 * y y y y
 * x o o x
 * x o o x
 * y y y y
 * 
 * @return See struct get_pos_t
 **/
template< int ndim, DIR_ID dir >
get_pos_t get_pos( const Functor& f, Functor::index_t index )
{
    get_pos_t res = {};

    // Construct border dimensions and border axis
    ComponentIndex3D dim_ghosts; // Ghost dimension
    ComponentIndex3D dim1, dim2; // Other dimensions (in ordrer X < Y < Z)
    uint32_t b1, b2;
    // Border origin differs if DIR_X, DIR_Y, DIR_Z because of corners
    coord_g_t borderOrigin;
    borderOrigin[IX] = -f.ghostWidth;
    borderOrigin[IY] = -f.ghostWidth;
    borderOrigin[IZ] = -f.ghostWidth;
    // using the axies (dim_ghosts, dim1, dim2),
    // border cells are inside [(0,0,0), (ghostWidth,b1,b2)[ 
    if( dir == DIR_X )
    {
        dim_ghosts  = IX;
        dim1        = IY;
        dim2        = IZ;
        b1 = f.blockSizes[dim1];
        b2 = f.blockSizes[dim2];
        // Set origin without corner in dimensions Y and Z
        borderOrigin[dim1] = 0;
        borderOrigin[dim2] = 0;
    }
    else if( dir == DIR_Y )
    {
        dim1        = IX;
        dim_ghosts  = IY;
        dim2        = IZ;
        // b1 is larger than for DIM_X to take corners into account
        b1 = f.blockSizes[dim1] + 2 * f.ghostWidth;
        b2 = f.blockSizes[dim2];
        // Set origin without corner in dimensions Z
        borderOrigin[dim2] = 0; 
    }
    else //if( dir == DIR_Z )
    {
        assert( (dir!=DIR_Z) or (ndim!=2) ); // Cannot use DIR_Z when dim == 2
        dim1        = IX;
        dim2        = IY;
        dim_ghosts  = IZ;
        b1 = f.blockSizes[dim1] + 2 * f.ghostWidth;
        b2 = f.blockSizes[dim2] + 2 * f.ghostWidth;
    }
    uint32_t nbcells_face = b1*b2*f.ghostWidth;
    blockSize_t borderSizes; // current border dimensions
    borderSizes[dim_ghosts] = f.ghostWidth;
    borderSizes[dim1] = b1;
    borderSizes[dim2] = b2;
    blockSize_t ghostWidths; // ghostWidth in different dimensions
    ghostWidths[IX] = f.ghostWidth;
    ghostWidths[IY] = f.ghostWidth;
    ghostWidths[IZ] = (ndim == 2) ? 0 : f.ghostWidth;

    // Skip cell if outside current border (too many compute units in team)
    if( index < nbcells_face*2 )
    {
        res.in_domain = true;

        //Determine if index is in left or right face
        FACE_ID face = FACE_LEFT;
        if( index >= nbcells_face )
        {
            face = FACE_RIGHT;
            index -= nbcells_face;
        }

        coord_t pos_in_border = index_to_coord(index, borderSizes[IX], borderSizes[IY], borderSizes[IZ] );

        // -- Compute position in local grid
        // GLITCHY : unsigned to signed (result should be positive)
        // FACE_LEFT borders originates from (-ghostWidth,-ghostWidth,-ghostWidth)
        res.pos_in_local = {
            static_cast<int32_t>(pos_in_border[IX]) + borderOrigin[IX],
            static_cast<int32_t>(pos_in_border[IY]) + borderOrigin[IY],
            static_cast<int32_t>(pos_in_border[IZ]) + borderOrigin[IZ]
        };
        // Shift coordinates in `dim_ghosts` dimensions for FACE_RIGHT
        if( face == FACE_RIGHT )
            res.pos_in_local[dim_ghosts] += ghostWidths[dim_ghosts] + f.blockSizes[dim_ghosts];
        
        // -- Compute position of neighbor octant + position to read from neighbor grid
        res.neighbor_pos = {0,0,0};
        coord_g_t pos_in_neighbor_aux = res.pos_in_local;
        for( ComponentIndex3D current_dim : {IX, IY, IZ} )
        {
            // "Left" border : shift position inside neighbor to the "right"
            if( pos_in_neighbor_aux[current_dim] < 0 )
            {
                res.neighbor_pos[current_dim] = -1;
                pos_in_neighbor_aux[current_dim] += f.blockSizes[current_dim];
            }
            // "Right" border : shift position inside neighbor to the "left"
            else if( pos_in_neighbor_aux[current_dim] >= (int32_t)f.blockSizes[current_dim] )
            {
                res.neighbor_pos[current_dim] = 1;
                pos_in_neighbor_aux[current_dim] -= f.blockSizes[current_dim];
            }
            // pos_in_neighbor_aux should be inside neighbor domain
            assert( pos_in_neighbor_aux[current_dim] >=0 && pos_in_neighbor_aux[current_dim] < (int32_t)f.blockSizes[current_dim] );
        }
        
        // Cast from signed to unsigned : pos_in_neighbor_aux should be inside domain (see assert above)
        res.pos_in_neighbor = {
            static_cast<uint32_t>(pos_in_neighbor_aux[IX]),
            static_cast<uint32_t>(pos_in_neighbor_aux[IY]),
            static_cast<uint32_t>(pos_in_neighbor_aux[IZ])
        };

        assert( res.pos_in_local[IX] == res.neighbor_pos[IX]*(int32_t)f.blockSizes[IX] + (int32_t)res.pos_in_neighbor[IX] );
        assert( res.pos_in_local[IY] == res.neighbor_pos[IY]*(int32_t)f.blockSizes[IY] + (int32_t)res.pos_in_neighbor[IY] );
        assert( res.pos_in_local[IZ] == res.neighbor_pos[IZ]*(int32_t)f.blockSizes[IZ] + (int32_t)res.pos_in_neighbor[IZ] );
    } 
    else 
        // Cell outside of current border : skip
        res.in_domain = false;

    return res;
}

using CellData = HydroState3d;

/**
 * Fetch data associated to a cell from a neighbor octant when neighbor and local octant have the same size
 * 
 * @param f ghost copy functor
 * @param iOct_neigh index of neighbor octant
 * @param pos_in_neighbor cell position inside neighbor
 * @param is_ghost is neighbor octant a ghost octant?
 **/
template <int ndim>
CellData get_cell_data_same_size( const Functor& f, uint32_t iOct_neigh, coord_t pos_in_neighbor, bool is_ghost)
{
    CellData res;
    uint32_t index_in_neighbor = coord_to_index_g<ndim>( pos_in_neighbor, f.blockSizes, 0 );
    if (is_ghost)
    {
        res[ID] = f.U_ghost(index_in_neighbor, f.fm[ID], iOct_neigh);
        res[IP] = f.U_ghost(index_in_neighbor, f.fm[IP], iOct_neigh);
        res[IU] = f.U_ghost(index_in_neighbor, f.fm[IU], iOct_neigh);
        res[IV] = f.U_ghost(index_in_neighbor, f.fm[IV], iOct_neigh);
        if( ndim == 3 ) res[IW] = f.U_ghost(index_in_neighbor, f.fm[IW], iOct_neigh);        
        // TODO : copy gravity
        // if (copy_gravity) { 
        //     Ugroup(index_cur, fm[IGX], iOct_local) = U_ghost(index_border, fm[IGX], iOct_neigh);
        //     Ugroup(index_cur, fm[IGY], iOct_local) = U_ghost(index_border, fm[IGY], iOct_neigh);
        // }
    } 
    else
    {
        res[ID] = f.U(index_in_neighbor, f.fm[ID], iOct_neigh);
        res[IP] = f.U(index_in_neighbor, f.fm[IP], iOct_neigh);
        res[IU] = f.U(index_in_neighbor, f.fm[IU], iOct_neigh);
        res[IV] = f.U(index_in_neighbor, f.fm[IV], iOct_neigh);
        if( ndim == 3 ) res[IW] = f.U(index_in_neighbor, f.fm[IW], iOct_neigh);
        // TODO : copy gravity
        // if (copy_gravity) { 
        //     Ugroup(index_cur, fm[IGX], iOct_local) = U(index_border, fm[IGX], iOct_neigh);
        //     Ugroup(index_cur, fm[IGY], iOct_local) = U(index_border, fm[IGY], iOct_neigh);
        // }
    }
    return res;
}

/**
 * Fetch data associated to a cell from a neighbor octant when neighbor is larger than local octant
 * 
 * @param f ghost copy functor
 * @param iOct_neigh index of neighbor octant
 * @param pos_in_neighbor cell position inside neighbor
 * @param is_ghost is neighbor octant a ghost octant?
 * @param cell_physical_pos physical position of cell to fetch
 * 
 * ex :  Fetching x for local octant returns the value of X form neighbor,
 *       Fetching y for local octant returns the value of Y form neighbor,
 * Local :            Neighbor:
 * 
 * . . . . . .     
 * . o o o o .      X   O   O   O
 * . o o o o x 
 * . o o o o .      O   O   O   O
 * . o o o o .      
 * . . . . . y      Y   O   O   O
 *                  
 *                  O   O   O   O
 * 
 * @note cell_physical_pos is necessary to determine in which sub-octant to fetch (see y in example above)
 **/
template <int ndim >
CellData get_cell_data_larger( const Functor& f, uint32_t iOct_neigh, coord_t pos_in_neighbor, bool is_ghost, real_t cell_physical_pos[3])
{
    bitpit::darray3 neighbor_center = is_ghost ? f.pmesh->getCenter(f.pmesh->getGhostOctant(iOct_neigh)) : f.pmesh->getCenter(iOct_neigh);

    for( ComponentIndex3D current_dim : {IX, IY, IZ} )
    {
        // Scale because neighbor is bigger 
        pos_in_neighbor[current_dim] /= 2;
        // Shift according to cell position inside neighbor octant
        if( cell_physical_pos[current_dim] > neighbor_center[current_dim] ) 
            pos_in_neighbor[current_dim] += f.blockSizes[current_dim]/2;
    }

    return get_cell_data_same_size<ndim>( f, iOct_neigh, pos_in_neighbor, is_ghost );
}

template< int ndim >
int neighbor_count = (ndim == 2) ? 2 : 4;

/**
 * Fetch data associated to a cell from a neighbor octant when neighbor is smaller than local octant
 * 
 * @param f ghost copy functor
 * @param iOct_neigh indexes of neighbor octants
 * @param pos_in_neighbor cell position inside neighbor
 * @param is_ghost are neighbor octants ghost octants?
 * @param cell_physical_pos physical position of cell to fetch
 * 
 * ex :  Fetching X for local octant returns an interpolated value from x in neighbor,
 *       Fetching Y for local octant returns an interpolated value from y in neighbor,
 *         Local :             Neighbor:
 * 
 *      .   .   .   .
 *                 
 *   .  O   O   O   O  X        x x o o     
 *                              x x o o  
 *   .  O   O   O   O  Y        y y o o     
 *                              y y o o     
 *   .  O   O   O   O  .         
 *                   
 *   .  O   O   O   O  .
 *      
 *      .   .   .   .                
 * 
 * @note cell_physical_pos is necessary to determine in which sub-octant cell is in local octant
 **/
template <int ndim>
CellData get_cell_data_smaller( const Functor& f, const std::vector<uint32_t>& iOcts_neigh, coord_t pos_in_neighbor, const std::vector<bool>& is_ghost, real_t cell_physical_pos[3])
{                
    // Get position of first cell in neighbor (as if it was same size)
    coord_t c0_neighbor = pos_in_neighbor; 
    // Scale in all dimensions because neighbor is smaller
    c0_neighbor[IX] = c0_neighbor[IX] * 2;
    c0_neighbor[IY] = c0_neighbor[IY] * 2;
    c0_neighbor[IZ] = c0_neighbor[IZ] * 2;

    // We Assume block size is pair : copy from only one neighbor suboctant
    assert( f.blockSizes[IX]%2 == 0 && f.blockSizes[IY]%2 == 0 && ( ndim ==2 || f.blockSizes[IZ]%2 == 0 )  );
    // Find which sub-octant contains the cell to fill
    int8_t suboctant = -1;
    for( size_t i=0; i<iOcts_neigh.size(); i++ )
    {        
        real_t npx = is_ghost[i] ? f.pmesh->getXghost(iOcts_neigh[i]) : f.pmesh->getX(iOcts_neigh[i]);
        real_t npy = is_ghost[i] ? f.pmesh->getYghost(iOcts_neigh[i]) : f.pmesh->getY(iOcts_neigh[i]);
        real_t npz = is_ghost[i] ? f.pmesh->getZghost(iOcts_neigh[i]) : f.pmesh->getZ(iOcts_neigh[i]);
        real_t neighbor_size = is_ghost[i] ? f.pmesh->getSizeGhost(iOcts_neigh[i]) : f.pmesh->getSize(iOcts_neigh[i]);

        bitpit::darray3 neighbor_min = {npx,npy,npz};

        if(   ( neighbor_min[IX] < cell_physical_pos[IX] &&  cell_physical_pos[IX] < neighbor_min[IX] + neighbor_size )
          and ( neighbor_min[IY] < cell_physical_pos[IY] &&  cell_physical_pos[IY] < neighbor_min[IY] + neighbor_size )
          and ( (ndim==2) || (neighbor_min[IZ] < cell_physical_pos[IZ] &&  cell_physical_pos[IZ] < neighbor_min[IZ] + neighbor_size )) )
        {
            suboctant = i;
            break;
        }        
    }
    assert( suboctant != -1 ); // Could not find suboctant

    CellData res = {};

    int dimz_size = ndim - 1;
      // loop inside neighbor block(s) (smaller octant) to accumulate values
    for (int8_t iz = 0; iz < dimz_size; ++iz)
    {
        for (int8_t iy = 0; iy < 2; ++iy)
        {
            for (int8_t ix = 0; ix < 2; ++ix)
            {               
                coord_t pos_in_smaller;
                pos_in_smaller[IX] = c0_neighbor[IX] + ix; 
                pos_in_smaller[IY] = c0_neighbor[IY] + iy;
                pos_in_smaller[IZ] = c0_neighbor[IZ] + iz;
                // Offset neighbor coordinates according to position relative to neighbor
                for( ComponentIndex3D current_dim : {IX, IY, IZ} )
                {
                    if( pos_in_smaller[current_dim] >= f.blockSizes[current_dim] ) 
                    {
                        pos_in_smaller[current_dim] -= f.blockSizes[current_dim];
                    }
                }

                // TODO : compute suboctant here to allow odd block sizes
                // and determine which neighbor to copy from.
                // note : Copy from multiple neighbors could be needed if blocksize is odd
                //   _____    ___ __...
                // N|_|_|_|  |   |
                // 0|_|_|_|  |___|_...
                // 1|_|_|_|->|   |

                // N|_|_|_|->|___|_...
                // 0|_|_|_|  |   |
                // 2|_|_|_|  |___|_...

                CellData d = get_cell_data_same_size<ndim>( f, iOcts_neigh[suboctant], pos_in_smaller, is_ghost[suboctant] );

                for( size_t i=0; i<res.size(); i++ )
                    res[i] += d[i];
            }
        }
    }

    for( size_t i=0; i<res.size(); i++ )
        res[i] /= dimz_size*4;
    return res;
}

/**
 * Generate data associated to a boundary cell
 * 
 * @param f ghost copy functor
 * @param iOct_local indexes of local octant
 * @param pos_in_local cell position inside local octant : (-1,0,0) is left ghost
 * @param is_ghost are neighbor octants ghost octants?
 * @param cell_physical_pos physical position of cell to fetch
 **/ 
template < int ndim, DIR_ID dir >
CellData get_cell_data_border( const Functor& f, uint32_t iOct_local, coord_g_t pos_in_local, real_t cell_center_physical[3])
{

    // normal momentum sign
    real_t sign[3] = {1.0,1.0,1.0};

    coord_g_t coord_in = {  pos_in_local[IX],
                            pos_in_local[IY],
                            pos_in_local[IZ]};

    BoundaryConditionType bc_min[3], bc_max[3];
    bc_min[IX] = f.params.boundary_type_xmin;
    bc_max[IX] = f.params.boundary_type_xmax;
    bc_min[IY] = f.params.boundary_type_ymin;
    bc_max[IY] = f.params.boundary_type_ymax;
    bc_min[IZ] = f.params.boundary_type_zmin;
    bc_max[IZ] = f.params.boundary_type_zmax;

    // absorbing border : we just copy/mirror data 
    // from the last inner cells into ghost cells
    for( ComponentIndex3D current_dim : {IX, IY, IZ} )
    {
        if(     bc_min[current_dim] == BC_ABSORBING 
                && cell_center_physical[current_dim] < f.tree_min[current_dim])
        { // Left border
            coord_in[current_dim] = 0;
        }
        else if(bc_max[current_dim] == BC_ABSORBING 
                && cell_center_physical[current_dim] > f.tree_max[current_dim])
        { // Right border
            coord_in[current_dim] = f.blockSizes[current_dim]-1;
        }
    }

    // reflecting border : we just copy/mirror data 
    // from inner cells into ghost cells, and invert
    // normal momentum
    for( ComponentIndex3D current_dim : {IX, IY, IZ} )
    {
        if(bc_min[current_dim] == BC_REFLECTING 
            && cell_center_physical[current_dim] < f.tree_min[current_dim])
        {
            coord_in[current_dim] = -1 - pos_in_local[current_dim];
            sign[current_dim] = -1.0;
        }
        else if(bc_max[current_dim] == BC_REFLECTING 
                && cell_center_physical[current_dim] > f.tree_max[current_dim])
        {
            coord_in[current_dim] = (f.blockSizes[current_dim]-1) - (pos_in_local[current_dim]-f.blockSizes[current_dim]);
            sign[current_dim] = -1.0;
        }
    }

    //if(iOct_local == 26 && f.iGroup==1)
    //std::cout << " Border " << pos_in_local[IX] << " "  << pos_in_local[IY] << " "  << pos_in_local[IZ] << " <- " << coord_in[IX] << " "  << coord_in[IY] << " "  << coord_in[IZ] << std::endl;
    
    // assume inner cells have already been copied 
    uint32_t index = coord_g_to_index<ndim>( coord_in, f.blockSizes, f.ghostWidth );
    CellData res;
    res[ID] = f.Ugroup(index, f.fm[ID], iOct_local);
    res[IP] = f.Ugroup(index, f.fm[IP], iOct_local);
    res[IU] = f.Ugroup(index, f.fm[IU], iOct_local) * sign[IX];
    res[IV] = f.Ugroup(index, f.fm[IV], iOct_local) * sign[IY];
    if(ndim == 3)
        res[IW] = f.Ugroup(index, f.fm[IW], iOct_local) * sign[IZ];

    // if (copy_gravity) {
    //     Ugroup(index_cur, fm[IGX], iOct_local) = Ugroup(index, fm[IGX], iOct_local);
    //     Ugroup(index_cur, fm[IGY], iOct_local) = Ugroup(index, fm[IGY], iOct_local);
    // } // end if copy
    return res;
}


/**
 * Table to convert neighbor relative position to "iface" parameter for findNeighbors()
 * If neighbor octant is at position (x, y, z) relative to local octant (-1<=x,y,z<=1)
 * iface for findNeighbors() is iface_from_pos[z+1][y+1][x+1]-1
 * 
 * In 2D : offset = (x,y,0) => iface = iface_from_pos[0][y+1][x+1]-1
 * 
 * Has been generated by https://gitlab.maisondelasimulation.fr/pkestene/dyablo/-/snippets/3
 **/
constexpr uint8_t iface_from_pos[3][3][3] = {
                        {{ 1, 3, 2 },
                            { 1, 5, 2 },
                            { 3, 4, 4 }},
                        {{ 5, 3, 6 },
                            { 1, 0, 2 },
                            { 7, 4, 8 }},
                        {{ 5, 11, 6 },
                            { 9, 6, 10 },
                            { 7, 12, 8 }}};
constexpr uint8_t CODIM_NODE = 3;
constexpr uint8_t CODIM_EDGE = 2;
constexpr uint8_t CODIM_FACE = 1;

/** 
 * When neighbor is larger, it is returned only in one `findNeighbours()` request
 * Sometimes `findNeighbours()` edge/node returns 0 neighbors because the only neighbor has already been returned by findNeighbour
 * on a lower codimension request. In this case, actual edge/node neighbor has to be searched in lower codimention neighbors
 * 
 * @param iOct_global index of local octant
 * @param neighbor neighbor position relative to local octant ( (-1,0,0) is left neighbor )
 **/
template < int ndim >
void fix_missing_corner_neighbor( const Functor& f, uint32_t iOct_global, const coord_g_t& neighbor, const real_t cellPos[3], std::vector<uint32_t>& res_iOct_neighbors, std::vector<bool>& res_isghost_neighbors)
{
    /// Check if position `cellPos` is inside neighbor designated by search_codim and search_iface
    /// @returns true if cell is inside and sets res_iOct_neighbors and res_isGhost_neighbors accordingly, false otherwise
    auto check_neighbor = [&](uint8_t search_codim, uint8_t search_iface)
    {   
        std::vector<uint32_t> iOct_neighbors;
        std::vector<bool> isghost_neighbors;

        f.pmesh->findNeighbours(iOct_global, search_iface, search_codim, iOct_neighbors, isghost_neighbors);
        if( iOct_neighbors.size() == 1 ) // Looking only for bigger neighbors
        {
            bitpit::darray3 neigh_min = isghost_neighbors[0] ? 
                f.pmesh->getCoordinates(f.pmesh->getGhostOctant(iOct_neighbors[0])) : 
                f.pmesh->getCoordinates(iOct_neighbors[0]);
            real_t oct_size = isghost_neighbors[0] ? 
                f.pmesh->getSize(f.pmesh->getGhostOctant(iOct_neighbors[0])) : 
                f.pmesh->getSize(iOct_neighbors[0]);
            bitpit::darray3 neigh_max = {
                    neigh_min[IX] + oct_size,
                    neigh_min[IY] + oct_size,
                    neigh_min[IZ] + oct_size
            };

            if( neigh_min[IX] < cellPos[IX] && cellPos[IX] < neigh_max[IX] 
            &&  neigh_min[IY] < cellPos[IY] && cellPos[IY] < neigh_max[IY]
            && ((ndim==2) || (neigh_min[IZ] < cellPos[IZ] && cellPos[IZ] < neigh_max[IZ])) )
            {
                res_iOct_neighbors = iOct_neighbors;
                res_isghost_neighbors = isghost_neighbors;
                return true;
            }
        }
        return false;        
    };

    // Determine neighbor codimension (node/edge/face)
    int codim = 0;
    codim += std::abs(neighbor[IX]);
    codim += std::abs(neighbor[IY]);
    codim += std::abs(neighbor[IZ]);

    if(ndim == 2)
    {
        assert( codim == CODIM_EDGE ); // In 2D, only edges can have this issue
        // Search both faces connected to edge:
        uint8_t search_codim = CODIM_FACE; 
        uint8_t search_iface_x = iface_from_pos[0][1][neighbor[IX]+1]-1;
        if( check_neighbor(search_codim, search_iface_x) ) return;
        uint8_t search_iface_y = iface_from_pos[0][neighbor[IY]+1][1]-1;
        if( check_neighbor(search_codim, search_iface_y) ) return;
        assert(false); // Failed to find neighbor...
    }
    else 
    {
        assert( codim == CODIM_NODE || codim == CODIM_EDGE ); // In 3D, edges and node can have this issue

        if(codim == CODIM_NODE) // search direction is a node (corner)
        {
            // Search the 3 edges connected to the node
            for(int i=0; i<3; i++)
            {
                uint8_t search_neighbor[3] = {(uint8_t)(neighbor[IX]+1), (uint8_t)(neighbor[IY]+1), (uint8_t)(neighbor[IZ]+1)};
                search_neighbor[i] = 1; // Search edges that have 2 coordinate in common
                uint8_t search_iface = iface_from_pos[search_neighbor[IZ]][search_neighbor[IY]][search_neighbor[IX]]-1;
                if( check_neighbor(CODIM_EDGE, search_iface) ) return;
            }
        }
        // Search connected faces
        for(int i=0; i<3; i++)
        {
            uint8_t search_neighbor[3] = {1,1,1};
            search_neighbor[i] = neighbor[i]+1; // Search faces that have 1 coordinate in common
            if( search_neighbor[IX] != 1 || search_neighbor[IY] != 1 || search_neighbor[IZ] != 1 ) // Only 2 faces when initially edge (0,0,0 not a neighbor)
            {
                uint8_t search_iface = iface_from_pos[search_neighbor[IZ]][search_neighbor[IY]][search_neighbor[IX]]-1;
                if( check_neighbor(CODIM_FACE, search_iface) ) return;
            }
        }
        assert(false); // Failed to find neighbor...
    }     
}


/**
 * @brief Fetch data associated to a cell from a neighbor octant.
 * 
 * Find neighbor octant and drive call to 
 *  * get_cell_data_same_size
 *  * get_cell_data_larger
 *  * get_cell_data_smaller
 * according to the size of the neighbor
 * 
 * @param f ghost copy functor
 * @param iOct_local index of local octant
 * @param neighbor realtive position of neighbor octant ( (-1,0,0) is left neighbor )
 * @param pos_in_neighbor cell position inside neighbor octant
 * @param pos_in_local cell position inside local octant
 * @return Cell data fetched from neighbor
 **/
template < int ndim, DIR_ID dir >
CellData get_cell_data( const Functor& f, uint32_t iOct_local, coord_g_t neighbor, coord_t pos_in_neighbor, coord_g_t pos_in_local )
{
    uint32_t iOct_global = iOct_local + f.iGroup * f.nbOctsPerGroup;

    // Compute cell physical position
    bitpit::darray3 oct_origin = f.pmesh->getCoordinates(iOct_global);
    real_t oct_size = f.pmesh->getSize(iOct_global);
    real_t cell_size[3] = {
        oct_size/f.blockSizes[IX],
        oct_size/f.blockSizes[IY],
        oct_size/f.blockSizes[IZ]
    };
    real_t cellPos[3] = {
        oct_origin[IX] + pos_in_local[IX] * cell_size[IX] + cell_size[IX]/2,
        oct_origin[IY] + pos_in_local[IY] * cell_size[IY] + cell_size[IY]/2,
        (ndim == 2) ? 0 : oct_origin[IZ] + pos_in_local[IZ] * cell_size[IZ] + cell_size[IZ]/2
    };
    
    // Determine codimension
    int count_dims = 0;
    count_dims += std::abs(neighbor[IX]);
    count_dims += std::abs(neighbor[IY]);
    count_dims += std::abs(neighbor[IZ]);
   
    uint8_t pablo_codim = count_dims;

    BoundaryConditionType bc_min[3], bc_max[3];
    bc_min[IX] = f.params.boundary_type_xmin;
    bc_max[IX] = f.params.boundary_type_xmax;
    bc_min[IY] = f.params.boundary_type_ymin;
    bc_max[IY] = f.params.boundary_type_ymax;
    bc_min[IZ] = f.params.boundary_type_zmin;
    bc_max[IZ] = f.params.boundary_type_zmax;

    // Shift periodic position
    for( ComponentIndex3D current_dim : {IX, IY, IZ} )
    {
        if( bc_min[current_dim] == BC_PERIODIC 
            && cellPos[current_dim] < f.tree_min[current_dim] )
            cellPos[current_dim] += f.tree_max[current_dim] - f.tree_min[current_dim];
        if( bc_max[current_dim] == BC_PERIODIC 
            && cellPos[current_dim] > f.tree_max[current_dim] )
            cellPos[current_dim] -= f.tree_max[current_dim] - f.tree_min[current_dim];
    }


    if( cellPos[IX] < f.tree_min[IX] || f.tree_max[IX] < cellPos[IX] 
     || cellPos[IY] < f.tree_min[IY] || f.tree_max[IY] < cellPos[IY] 
     || cellPos[IZ] < f.tree_min[IZ] || f.tree_max[IZ] < cellPos[IZ] )
    {
        return CellData({0,1.11}); // Boundary conditions are set later in the kernel;
    }

    int neighbor_pos_z = (ndim==2) ? 0 : neighbor[IZ]+1;
    uint8_t pablo_iface = iface_from_pos[neighbor_pos_z][neighbor[IY]+1][neighbor[IX]+1] - 1;

    std::vector<uint32_t> iOct_neighbors;
    std::vector<bool> isghost_neighbors;
    f.pmesh->findNeighbours(iOct_global, pablo_iface, pablo_codim, iOct_neighbors, isghost_neighbors);

    uint32_t oct_level = f.pmesh->getLevel(iOct_global);

    if(iOct_neighbors.size() == 0) 
    {
        fix_missing_corner_neighbor<ndim>(f, iOct_global, neighbor, cellPos, iOct_neighbors, isghost_neighbors);
    }
    
    if(iOct_neighbors.size() == 0) return CellData({0,1.22});
    assert(iOct_neighbors.size() > 0); //Should have at least one neighbor (Boundaries already taken care of)

    // All neighbors are on same level as neighbor[0]
    uint32_t neigh_level = isghost_neighbors[0] ? f.pmesh->getLevel(f.pmesh->getGhostOctant(iOct_neighbors[0])) : f.pmesh->getLevel(iOct_neighbors[0]);

    if( neigh_level == oct_level ) 
    {
        return get_cell_data_same_size<ndim>(f, iOct_neighbors[0], pos_in_neighbor, isghost_neighbors[0]);
    }
    else if( neigh_level > oct_level ) 
    {
        return get_cell_data_smaller<ndim>(f, iOct_neighbors, pos_in_neighbor, isghost_neighbors, cellPos);
    }
    else //if( neigh_level < oct_level ) 
    {
        return get_cell_data_larger<ndim>(f, iOct_neighbors[0], pos_in_neighbor, isghost_neighbors[0], cellPos);
    }
}

/**
 * @brief Write cell data into local octant group data.
 * 
 * @param f ghost copy functor
 * @param iOct_g index in group of local octant
 * @param pos_in_local cell position inside local octant
 * @param data cell data to write
 **/
template <int ndim>
void write_cell_data( const Functor& f, uint32_t iOct_g, coord_g_t pos_in_local, const CellData& data )
{
    uint32_t index_cur = coord_g_to_index<ndim>( pos_in_local, f.blockSizes, f.ghostWidth );
    f.Ugroup(index_cur, f.fm[ID], iOct_g) = data[ID];
    f.Ugroup(index_cur, f.fm[IP], iOct_g) = data[IP];
    f.Ugroup(index_cur, f.fm[IU], iOct_g) = data[IU];
    f.Ugroup(index_cur, f.fm[IV], iOct_g) = data[IV];
    if(ndim == 3)
    {
        f.Ugroup(index_cur, f.fm[IW], iOct_g) = data[IW];
    }
    if(f.copy_gravity)
    {
        f.Ugroup(index_cur, f.fm[IGX], iOct_g) = data[IGX];
        f.Ugroup(index_cur, f.fm[IGY], iOct_g) = data[IGY];
    }
}


/**
 * @brief Sets the value of one cell inside a ghost face of local octant.
 * 
 * Iterate over cells in ghost faces of octant `iOct_g` using `index`
 * 
 * @tparam ndim number of dimensions (2 or 3)
 * @tparam dir ghost direction. ex : if dir==DIR_X in 2D, resulting cell positions cover left AND right ghost cells
 * @param f Ghost copy functor (to fetch simulation parameters)
 * @param iOct_g index in group of local octant
 * @param index index between 0 and the number of cells in border to be mapped to a position inside border 
 *              (see get_pos() for the detail of the mapping index -> cell position)
 **/
template< int ndim, DIR_ID dir >
void fill_ghost_faces( const Functor& f, uint32_t iOct_g, Functor::index_t index )
{
    // Compute positions as if neighbor was same size
    get_pos_t p = get_pos<ndim, dir>(f, index);

    if( p.in_domain )
    {
        // Get data from neighbor position + position inside neighbor
        CellData data = get_cell_data<ndim, dir>( f, iOct_g, p.neighbor_pos, p.pos_in_neighbor, p.pos_in_local );

        // Write data to local cell
        write_cell_data<ndim>( f, iOct_g, p.pos_in_local, data );
    }
}

/**
 * @brief Sets the value af a cell inside a ghost face of local octant when the cell is a boundary ghost
 * 
 * Similar to fill_ghost_faces() but for boundary cells
 **/
template< int ndim, DIR_ID dir >
void fill_boundary_faces( const Functor& f, uint32_t iOct_local, Functor::index_t index )
{
    // Compute positions as if neighbor was same size
    get_pos_t p = get_pos<ndim, dir>(f, index);

    if( p.in_domain )
    {
        uint32_t iOct_global = iOct_local + f.iGroup * f.nbOctsPerGroup;

        bitpit::darray3 oct_origin = f.pmesh->getCoordinates(iOct_global);
        real_t oct_size = f.pmesh->getSize(iOct_global);
        real_t cell_size[3] = {
            oct_size/f.blockSizes[IX],
            oct_size/f.blockSizes[IY],
            oct_size/f.blockSizes[IZ]
        };
        real_t cellPos[3] = {
            oct_origin[IX] + p.pos_in_local[IX] * cell_size[IX] + cell_size[IX]/2,
            oct_origin[IY] + p.pos_in_local[IY] * cell_size[IY] + cell_size[IY]/2,
            (ndim == 2) ? 0 : oct_origin[IZ] + p.pos_in_local[IZ] * cell_size[IZ] + cell_size[IZ]/2
        };
            
        if( cellPos[IX] < f.tree_min[IX] || cellPos[IX] > f.tree_max[IX]
        || cellPos[IY] < f.tree_min[IY] || cellPos[IY] > f.tree_max[IY]
        || (ndim == 3 && (cellPos[IZ] < f.tree_min[IZ] || cellPos[IZ] > f.tree_max[IZ]) ) )
        {
            CellData cellData = get_cell_data_border<ndim, dir>(f, iOct_local, p.pos_in_local, cellPos);
            write_cell_data<ndim>( f, iOct_local, p.pos_in_local, cellData );
        }    
    }
}

/**
 * @brief Set the values of every ghost cells in every octant of the group
 * 
 * Iterate over Octants assigned to the team and copy ghost cells data from neighbor octants. 
 **/
template< int ndim >
void fill_ghosts(const Functor& f, Functor::team_policy_t::member_type member)
{
    static_assert(ndim == 2 || ndim == 3 , "CopyFaceBlockCellData only supports 2D and 3D");

    const uint32_t iGroup = f.iGroup;
    const uint32_t nbOctsPerGroup = f.nbOctsPerGroup;
    //const uint32_t bx = f.blockSizes[IX];
    const uint32_t by = f.blockSizes[IY];
    const uint32_t bz = f.blockSizes[IZ];
    const uint32_t ghostWidth = f.ghostWidth;

    // iOct must span the range [iGroup*nbOctsPerGroup , (iGroup+1)*nbOctsPerGroup [
    uint32_t iOct = member.league_rank() + iGroup * nbOctsPerGroup;
    // octant id inside the Ugroup data array
    uint32_t iOct_g = member.league_rank();
    // total number of octants
    uint32_t nbOcts = f.pmesh->getNumOctants();
    // compute first octant index after current group
    uint32_t iOctNextGroup = (iGroup + 1) * nbOctsPerGroup;   

    uint32_t nbcells_face_X = ghostWidth*by*bz;
    uint32_t nbcells_face_Y = f.bx_g*ghostWidth*bz;
    uint32_t nbcells_face_Z = (ndim==2) ? 0 : f.bx_g*f.by_g*ghostWidth;
    uint32_t nbCells = 2*max_n(nbcells_face_X, nbcells_face_Y, nbcells_face_Z);

    while (iOct < iOctNextGroup and iOct < nbOcts)
    {
      f.interface_flags.resetFlags(iOct_g);

      // perform "vectorized" loop inside a given block data
      Kokkos::parallel_for(
        Kokkos::TeamVectorRange(member, nbCells),
        KOKKOS_LAMBDA(const Functor::index_t index)
        {
          // compute face X,left (left)
          fill_ghost_faces<ndim, DIR_X>(f, iOct_g, index);
          fill_ghost_faces<ndim, DIR_Y>(f, iOct_g, index);
          if(ndim == 3)
            fill_ghost_faces<ndim, DIR_Z>(f, iOct_g, index);          
        }); // end TeamVectorRange
    
        member.team_barrier();
        if( f.pmesh->getBound(iOct) ) //This octant has ghosts outside of global domain
        {
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(member, nbCells),
                KOKKOS_LAMBDA(const Functor::index_t index)
                {
                    // compute face X,left (left)
                    fill_boundary_faces<ndim, DIR_X>(f, iOct_g, index);
                    fill_boundary_faces<ndim, DIR_Y>(f, iOct_g, index);
                    if(ndim == 3)
                        fill_boundary_faces<ndim, DIR_Z>(f, iOct_g, index); 
                });
        }
      
      // increase current octant location both in U and Ugroup
      iOct += f.nbTeams;
      iOct_g += f.nbTeams;

    } // end while iOct inside current group of octants

  } // functor<ndim>()

} //namespace

void CopyGhostBlockCellDataFunctor::operator()(team_policy_t::member_type member) const
{
    if (params.dimType == TWO_D)
      fill_ghosts<2>(*this, member);

    if (params.dimType == THREE_D)
      fill_ghosts<3>(*this, member);
}

} // namespace muscl_block
} // namespace dyablo