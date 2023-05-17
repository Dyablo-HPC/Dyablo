/**
 * This unit test is aimed at testing boundary conditions in dyablo
 * 
 * Summary of the tests : 
 *  1. Testing the getBoundaryPosAndOffset method in CellIndex
 *  2. Testing the values of a reflecting boundary 
 *  3. Testing the values of an absorbing boundary
 */
#include "gtest/gtest.h"

#include "amr/AMRmesh.h"
#include "utils/mpi/GlobalMpiSession.h"
#include "utils/monitoring/Timers.h"
#include "io/IOManager.h"

#include "foreach_cell/ForeachCell.h"
#include "boundary_conditions/BoundaryConditions.h"
#include "enums.h"

using CellIndex = dyablo::ForeachCell::CellIndex;


namespace dyablo {

using TmpView = Kokkos::View<real_t**>;
using TmpViewHost = TmpView::HostMirror;

namespace {

enum VarIndex_test{
  ID,IE,IP=IE,IU,IV,IW,IGX,IGY,IGZ
};

/**
 * Simple Kokkos functor filling a vector of state
 * from a data array and a list of positions using 
 * the boundary conditions manager.
 */
struct FillFunctor {
  ForeachCell::CellArray_global_ghosted U;
  TmpView out;
  ForeachCell::CellMetaData metadata;
  BoundaryConditions bc_manager;
  CellIndex cv[4];

  FillFunctor(ConfigMap &configMap,
              ForeachCell::CellArray_global_ghosted U, 
              TmpView out, 
              ForeachCell::CellMetaData metadata,
              CellIndex cv[],
              BoundaryConditionType bc_type)
    : U(U), out(out), metadata(metadata), bc_manager(configMap) {
      for (int i=0; i < 4; ++i)
        this->cv[i] = cv[i];

      for (auto dir : {IX, IY, IZ}) {
        bc_manager.bc_min[dir] = bc_type;
        bc_manager.bc_max[dir] = bc_type;
      }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    auto fm = U.fm;
    auto v = bc_manager.getBoundaryValue<3, HydroState>(U, cv[i], metadata);
    out(fm[ID], i) = v.rho;
    out(fm[IE], i) = v.e_tot;
    out(fm[IU], i) = v.rho_u;
    out(fm[IV], i) = v.rho_v;
    out(fm[IW], i) = v.rho_w;
  }
};

/**
 * @brief Builds a CellIndex object for a boundary cell
 * 
 * This method returns a CellIndex object that will be constructed from a position
 * inside the domain (i, j, k) and an offset (di, dj, dk) and a block size (bx, by, bz). 
 * The cell is flagged as boundary, however no checks are performed if the resulting 
 * position is really outside the cell.
 * 
 * @note IMPORTANT : The cell is constructed as in AMRBlockForeachCell_CellArray.h
 *                   hence the final index in each direction tells if we're out of
 *                   the cell on the left (eg i in [0; bx[), inside the cell ([bx; 2*bx[)
 *                   or on the right ([2*bx; 3*bx[).
 */
CellIndex make_boundary_cellindex(int i, int j, int k, int di, int dj, int dk, int bx, int by, int bz) {
  return CellIndex{{0, false}, 
                   (uint32_t)i+bx+di, 
                   (uint32_t)j+by+dj, 
                   (uint32_t)k+bz+dk, 
                   (uint32_t)bx, 
                   (uint32_t)by, 
                   (uint32_t)bz, 
                   CellIndex::BOUNDARY};
}

/**
 * @brief Test class deriving from ::testing::Test to allow test fixtures
 */
class TestBoundaryConditions : public ::testing::Test {
public:
  std::shared_ptr<ConfigMap>                 configMap;
  int                                        ndim;
  std::shared_ptr<AMRmesh>                   amr_mesh;
  FieldManager                               fieldMgr;
  ForeachCell::CellArray_global_ghosted      U;

  void SetUp() override {
    std::cout << "// =========================================\n";
    std::cout << "// Testing Boundary conditions ...\n";
    std::cout << "// =========================================\n";

    ndim = 3;

    std::cout << "Create mesh..." << std::endl;
    std::string configmap_str = 
      "[amr]\n"
      "use_block_data=yes\n"
      "bx=4\n"
      "by=5\n"
      "bz=6\n"
      "[mesh]\n"
      "ndim=3\n";
    configMap = std::make_shared<ConfigMap>(configmap_str);

    amr_mesh   = std::make_shared<AMRmesh>(ndim, ndim, std::array<bool,3>{false,false,false}, 3, 5);
    fieldMgr   = FieldManager({ID,IP,IU,IV,IW});

    uint32_t nbOcts = amr_mesh->getNumOctants();

    std::cout << "Initialize User Data..." << std::endl;
    ForeachCell foreach_cell(*amr_mesh, *configMap);
    U = foreach_cell.allocate_ghosted_array("U", fieldMgr);
    int bx = U.bx;
    int by = U.by;
    int bz = U.bz;
    uint32_t nbCellsPerOct = bx*by*bz;
    auto fm = fieldMgr.get_id2index();
    { 
      // Initialize U
      auto U_host = Kokkos::create_mirror_view(U.U);
      for( uint32_t iOct=0; iOct<nbOcts; iOct++ )
      { 
        for( uint32_t c=0; c<nbCellsPerOct; c++ )
        {
          U_host(c, fm[ID], iOct) = c;
          U_host(c, fm[IP], iOct) = c;
          U_host(c, fm[IU], iOct) = c;
          U_host(c, fm[IV], iOct) = c;
          U_host(c, fm[IW], iOct) = c;
        }
      }
      Kokkos::deep_copy( U.U, U_host );
      U.exchange_ghosts(GhostCommunicator(amr_mesh));
    }
  }

  ~TestBoundaryConditions() {}
};

/**
 * This test checks if the CellIndex method getBoundaryPosAndOffset works correctly
 * This method takes as input a CellIndex in the boundary (outside the domain), and
 * returns the index of the closest cell in the domain (cell_inside) as well as an offset. 
 * This offset, if applied to cell_inside should return the original cell outside of the 
 * domain.
 */
TEST_F(TestBoundaryConditions, getBoundaryPosAndOffset) {
  using offset_t = CellIndex::offset_t;

  // Defining cells outside of the boundary
  // For treating boundary conditions, indices are shifted by the size of a block
  // Hence, the inside of the block starts at i,j,k = (bx,by,bz) and ends at (2bx-1,2by-1,2bz-1) 
  // included
  int bx=U.bx;
  int by=U.by;
  int bz=U.bz;
  CellIndex c1 = make_boundary_cellindex(0, 0, 0, -1, -1, -1, bx, by, bz);
  CellIndex c2 = make_boundary_cellindex(0, 0, 0, -2, -2, -2, bx, by, bz);
  CellIndex c3 = make_boundary_cellindex(bx-1, by-1, bz-1, 1, 1, 1, bx, by, bz);
  CellIndex c4 = make_boundary_cellindex(bx-1, by-1, bz-1, 2, 2, 2, bx, by, bz);
  
  // Retrieving boundary pos and offset
  CellIndex bc1, bc2, bc3, bc4;
  offset_t off1, off2, off3, off4;

  c1.getBoundaryPosAndOffset(bc1, off1);
  c2.getBoundaryPosAndOffset(bc2, off2);
  c3.getBoundaryPosAndOffset(bc3, off3);
  c4.getBoundaryPosAndOffset(bc4, off4);

  // Testing offsets
  EXPECT_EQ(off1[IX], -1);
  EXPECT_EQ(off1[IY], -1);
  EXPECT_EQ(off1[IZ], -1);
  EXPECT_EQ(off2[IX], -2);
  EXPECT_EQ(off2[IY], -2);
  EXPECT_EQ(off2[IZ], -2);
  EXPECT_EQ(off3[IX],  1);
  EXPECT_EQ(off3[IY],  1);
  EXPECT_EQ(off3[IZ],  1);
  EXPECT_EQ(off4[IX],  2);
  EXPECT_EQ(off4[IY],  2);
  EXPECT_EQ(off4[IZ],  2);

  // Testing indices
  EXPECT_EQ(bc1.i, 0);
  EXPECT_EQ(bc1.j, 0);
  EXPECT_EQ(bc1.k, 0);
  EXPECT_EQ(bc2.i, 0);
  EXPECT_EQ(bc2.j, 0);
  EXPECT_EQ(bc2.k, 0);
  EXPECT_EQ(bc3.i, 3);
  EXPECT_EQ(bc3.j, 4);
  EXPECT_EQ(bc3.k, 5);
  EXPECT_EQ(bc4.i, 3);
  EXPECT_EQ(bc4.j, 4);
  EXPECT_EQ(bc4.k, 5);
}

/**
 * This test checks if the boundary conditions class manages properly the absorbing boundary
 * conditions. Especially, the boundary condition should reproduce the values in the domain of
 * the cell that is placed symmetrically to the boundary interface.
 */
TEST_F(TestBoundaryConditions, absorbingBoundaries) {
  // Defining indices
  int bx = U.bx;
  int by = U.by;
  int bz = U.bz;
  using pos_t = Kokkos::Array<int, 3>;

  // Cells from which the boundary cells are created
  pos_t cp[4] {{0, 2, 1},
               {0, 1, 1},
               {bx-2, by-3, bz-1},
               {bx-2, by-1, bz-2}};
  // Symmetric cells
  pos_t sp[4] {{0, 2, 1},
               {1, 1, 1},
               {bx-2, by-3, bz-1},
               {bx-2, by-2, bz-2}};

  // Linear ids of symmetric cells
  int lids[4];
  for (int i=0; i < 4; ++i)
    lids[i] = sp[i][IX] + bx*sp[i][IY] + bx*by*sp[i][IZ];

  CellIndex c1 = make_boundary_cellindex(cp[0][IX], cp[0][IY], cp[0][IZ], -1, 0, 0, bx, by, bz);
  CellIndex c2 = make_boundary_cellindex(cp[1][IX], cp[1][IY], cp[1][IZ], -2, 0, 0, bx, by, bz);
  CellIndex c3 = make_boundary_cellindex(cp[2][IX], cp[2][IY], cp[2][IZ],  0, 0, 1, bx, by, bz);
  CellIndex c4 = make_boundary_cellindex(cp[3][IX], cp[3][IY], cp[3][IZ],  0, 2, 0, bx, by, bz);

  TmpView out("test_BC", 5, 4);
  CellIndex cv[4] {c1, c2, c3, c4};
  ForeachCell foreach_cell(*amr_mesh, *configMap);
  auto U = this->U;
  auto metadata = foreach_cell.getCellMetaData();

  FillFunctor f{*configMap, U, out, metadata, cv, BC_ABSORBING};
  Kokkos::parallel_for(4, f);

  TmpViewHost out_host = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(out_host, out);

  auto fm = U.fm;

  // Testing values
  EXPECT_NEAR(out_host(fm[IU],0), lids[0], 1e-3);
  EXPECT_NEAR(out_host(fm[IV],0), lids[0], 1e-3);
  EXPECT_NEAR(out_host(fm[IW],0), lids[0], 1e-3);
  EXPECT_NEAR(out_host(fm[IU],1), lids[1], 1e-3);
  EXPECT_NEAR(out_host(fm[IV],1), lids[1], 1e-3);
  EXPECT_NEAR(out_host(fm[IW],1), lids[1], 1e-3);
  EXPECT_NEAR(out_host(fm[IU],2), lids[2], 1e-3);
  EXPECT_NEAR(out_host(fm[IV],2), lids[2], 1e-3);
  EXPECT_NEAR(out_host(fm[IW],2), lids[2], 1e-3);
  EXPECT_NEAR(out_host(fm[IU],3), lids[3], 1e-3);
  EXPECT_NEAR(out_host(fm[IV],3), lids[3], 1e-3);
  EXPECT_NEAR(out_host(fm[IW],3), lids[3], 1e-3);
}

/**
 * This test checks if the boundary conditions class manages properly the reflecting boundary
 * conditions. Especially, the boundary condition should :
 *  1. Reproduce the values of the symmetrical cell in the domain
 *  2. Invert the normal velocity wrt the limit.
 */
TEST_F(TestBoundaryConditions, reflectingBoundaries) {
  // Defining indices
  int bx = U.bx;
  int by = U.by;
  int bz = U.bz;
  using pos_t = Kokkos::Array<int, 3>;

  // Cells from which the boundary cells are created
  pos_t cp[4] {{0, 2, 1},
               {0, 1, 1},
               {bx-2, by-3, bz-1},
               {bx-2, by-1, bz-2}};
  // Symmetric cells
  pos_t sp[4] {{0, 2, 1},
               {1, 1, 1},
               {bx-2, by-3, bz-1},
               {bx-2, by-2, bz-2}};

  // Linear ids of symmetric cells
  int lids[4];
  for (int i=0; i < 4; ++i)
    lids[i] = sp[i][IX] + bx*sp[i][IY] + bx*by*sp[i][IZ];

  CellIndex c1 = make_boundary_cellindex(cp[0][IX], cp[0][IY], cp[0][IZ], -1, 0, 0, bx, by, bz);
  CellIndex c2 = make_boundary_cellindex(cp[1][IX], cp[1][IY], cp[1][IZ], -2, 0, 0, bx, by, bz);
  CellIndex c3 = make_boundary_cellindex(cp[2][IX], cp[2][IY], cp[2][IZ],  0, 0, 1, bx, by, bz);
  CellIndex c4 = make_boundary_cellindex(cp[3][IX], cp[3][IY], cp[3][IZ],  0, 2, 0, bx, by, bz);

  // Setting up boundary conditions
  BoundaryConditions bc_manager(*configMap);
  bc_manager.bc_min[IX] = BC_REFLECTING;
  bc_manager.bc_min[IY] = BC_REFLECTING;
  bc_manager.bc_min[IZ] = BC_REFLECTING;
  bc_manager.bc_max[IX] = BC_REFLECTING;
  bc_manager.bc_max[IY] = BC_REFLECTING;
  bc_manager.bc_max[IZ] = BC_REFLECTING;

  using TmpView = Kokkos::View<real_t**>;
  using TmpViewHost = TmpView::HostMirror;
  TmpView out("test_BC", 5, 4);
  ForeachCell foreach_cell(*amr_mesh, *configMap);
  auto metadata = foreach_cell.getCellMetaData();
  CellIndex cv[4] {c1, c2, c3, c4};

  FillFunctor f{*configMap, U, out, metadata, cv, BC_REFLECTING};
  Kokkos::parallel_for(4, f);

  TmpViewHost out_host = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(out_host, out);

  auto fm = U.fm;

  // Testing values
  EXPECT_NEAR(out_host(fm[IU],0), -lids[0], 1e-3);
  EXPECT_NEAR(out_host(fm[IV],0), lids[0], 1e-3);
  EXPECT_NEAR(out_host(fm[IW],0), lids[0], 1e-3);
  EXPECT_NEAR(out_host(fm[IU],1), -lids[1], 1e-3);
  EXPECT_NEAR(out_host(fm[IV],1), lids[1], 1e-3);
  EXPECT_NEAR(out_host(fm[IW],1), lids[1], 1e-3);
  EXPECT_NEAR(out_host(fm[IU],2), lids[2], 1e-3);
  EXPECT_NEAR(out_host(fm[IV],2), lids[2], 1e-3);
  EXPECT_NEAR(out_host(fm[IW],2), -lids[2], 1e-3);
  EXPECT_NEAR(out_host(fm[IU],3), lids[3], 1e-3);
  EXPECT_NEAR(out_host(fm[IV],3), -lids[3], 1e-3);
  EXPECT_NEAR(out_host(fm[IW],3), lids[3], 1e-3);
}
}
}