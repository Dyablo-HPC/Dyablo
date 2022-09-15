/**
 * Testing the states implementation and conversion
 **/

#include "utils/mpi/GlobalMpiSession.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"
#include "compute_dt/Compute_dt.h"
#include "init/InitialConditions.h"
#include "states/State_forward.h"
#include "states/State_Nd.h"
#include "io/IOManager.h"
#include "foreach_cell/ForeachCell.h"
using blockSize_t    = Kokkos::Array<uint32_t, 3>;

using Device = Kokkos::DefaultExecutionSpace;

#include "gtest/gtest.h"

using namespace dyablo;

TEST( Test_StateNd, StateNd_manipulation )
{
  StateNd<3> offset {-2.0, 1.0, 0.1};
  StateNd<3> offset2 {5.3, 0.1, -0.1};

  auto add = offset + offset2;
  for (size_t i=0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(add[i], offset[i]+offset2[i]);

  real_t q1 = 3.0;
  auto mul = offset * q1;
  for (size_t i=0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(mul[i], offset[i]*q1);

  auto mul2 = offset * offset2;
  for (size_t i=0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(mul2[i], offset[i]*offset2[i]);

  real_t q2 = 2.0;
  auto div = offset / q2;
  for (size_t i=0; i < 3; ++i) 
    EXPECT_DOUBLE_EQ(div[i], offset[i] / q2);

  StateNd<3> offset3{offset};
  real_t q3 = 0.1;
  offset3 /= q3;
  for (size_t i=0; i < 3; ++i)
    EXPECT_DOUBLE_EQ(offset3[i], offset[i]/q3);
}

TEST( Test_Hydro_legacy, HydroState2d ) 
{
  HydroState2d state1 {1.0, 0.1, 0.1, 0.2};
  HydroState2d state2 {0.12, 0.22, 1.54, -2.3};

  auto add = state1 + state2;
  for (size_t i=0; i < 4; ++i) 
    EXPECT_DOUBLE_EQ(add[i], state1[i]+state2[i]);

  real_t q1 = 6.4;
  auto mul = state1 * q1;
  for (size_t i=0; i < 4; ++i)
    EXPECT_DOUBLE_EQ(mul[i], state1[i]*q1);

  auto mul2 = state1 * state2;
  for (size_t i=0; i < 4; ++i)
    EXPECT_DOUBLE_EQ(mul2[i], state1[i]*state2[i]);

  real_t q2 = 3.8;
  auto div = state1 / q2;
  for (size_t i=0; i < 4; ++i)
    EXPECT_DOUBLE_EQ(div[i], state1[i]/q2);

  HydroState2d state3{state1};
  for (size_t i=0; i < 4; ++i)
    EXPECT_DOUBLE_EQ(state3[i], state1[i]);

  real_t q3 = 3.0;
  state3 /= q3;
  for (size_t i=0; i < 4; ++i)
    EXPECT_DOUBLE_EQ(state3[i], state1[i]/q3);
}

TEST( Test_Hydro_legacy, HydroState3d ) 
{
  HydroState3d state1 {6.4, 0.23, -0.23, 0.72, 4.15};
  HydroState3d state2 {0.0, 1.02, 1.24, -1.0, 4.2};

  auto add = state1 + state2;
  for (size_t i=0; i < 5; ++i)
    EXPECT_DOUBLE_EQ(add[i], state1[i]+state2[i]);
  
  real_t q1 = 6.4;
  auto mul = state1 * q1;
  for (size_t i=0; i < 5; ++i)
    EXPECT_DOUBLE_EQ(mul[i], state1[i]*q1);

  auto mul2 = state1 * state2;
  for (size_t i=0; i < 5; ++i)
    EXPECT_DOUBLE_EQ(mul2[i], state1[i]*state2[i]);

  real_t q2 = 0.23;
  auto div = state1 / q2;
  for (size_t i=0; i < 5; ++i)
    EXPECT_DOUBLE_EQ(div[i], state1[i]/q2);

  real_t q3 = 8.1;
  HydroState3d state3{state1};
  for (size_t i=0; i < 5; ++i)
    EXPECT_DOUBLE_EQ(state1[i], state3[i]);

  state3 /= q3;
  for (size_t i=0; i < 5; ++i)
    EXPECT_DOUBLE_EQ(state3[i], state1[i]/q3);
}

////// Hydro

TEST( Test_Hydro_States, PrimHydroState )
{
  auto test_eq = [&](const PrimHydroState& state, 
                     real_t rho, 
                     real_t p,
                     real_t u,
                     real_t v,
                     real_t w) {
    EXPECT_DOUBLE_EQ(state.rho, rho);
    EXPECT_DOUBLE_EQ(state.p, p);
    EXPECT_DOUBLE_EQ(state.u, u);
    EXPECT_DOUBLE_EQ(state.v, v);
    EXPECT_DOUBLE_EQ(state.w, w);
  };

  auto test_eq_st = [&](const PrimHydroState& state1,
                        const PrimHydroState& state2) {
    EXPECT_DOUBLE_EQ(state1.rho, state2.rho);
    EXPECT_DOUBLE_EQ(state1.p,   state2.p);
    EXPECT_DOUBLE_EQ(state1.u,   state2.u);
    EXPECT_DOUBLE_EQ(state1.v,   state2.v);
    EXPECT_DOUBLE_EQ(state1.w,   state2.w);
  };

  // Default constructor
  PrimHydroState state0;
  test_eq(state0, 0.0, 0.0, 0.0, 0.0, 0.0);

  // Init constructor
  PrimHydroState state1 {1.0, 2.56, 2.1, -7.8, 0.01};
  test_eq(state1, 1.0, 2.56, 2.1, -7.8, 0.01);

  // Copy constructor
  PrimHydroState state2 {0.1, 1.5, -1.0, -0.2, -9.2};
  PrimHydroState state3{state2};
  test_eq_st(state3, state2);

  // operator +
  auto add = state1 + state2;
  test_eq(add, 
          state1.rho + state2.rho,
          state1.p   + state2.p,
          state1.u   + state2.u,
          state1.v   + state2.v,
          state1.w   + state2.w);

  real_t q1 = 2.942;
  auto add2 = state1 + q1;
  test_eq(add2,
          state1.rho + q1,
          state1.p   + q1,
          state1.u   + q1,
          state1.v   + q1,
          state1.w   + q1);

  // operator +=
  state3 += state1;
  test_eq_st(state3, add);

  state3 = state1;
  state3 += q1;
  test_eq_st(state3, add2);

  // operator -
  auto sub = state1 - state2;
  test_eq(sub,
          state1.rho - state2.rho,
          state1.p   - state2.p,
          state1.u   - state2.u,
          state1.v   - state2.v,
          state1.w   - state2.w);

  // operator*
  auto mul = state1 * state2;
  test_eq(mul,
          state1.rho * state2.rho,
          state1.p   * state2.p,
          state1.u   * state2.u,
          state1.v   * state2.v,
          state1.w   * state2.w);

  real_t q3 = 7.123927;
  auto mul2 = state2 * q3;
  test_eq(mul2,
          state2.rho * q3,
          state2.p * q3,
          state2.u * q3,
          state2.v * q3,
          state2.w * q3);

  auto mul3 = q3 * state2;
  test_eq_st(mul3, mul2);

  real_t q4 = -137.22;
  auto state4{state1};
  state4 *= q4;
  test_eq_st(state4, state1*q4);

  // operator/
  real_t q5 = 0.289;
  auto div = state1 / q5;
  test_eq(div,
          state1.rho / q5,
          state1.p / q5,
          state1.u / q5,
          state1.v / q5,
          state1.w / q5);

  state1 /= q5;
  test_eq_st(state1, div);
}

TEST( Test_Hydro_States, ConsHydroState )
{
auto test_eq = [&](const ConsHydroState& state, 
                     real_t rho, 
                     real_t e_tot,
                     real_t rho_u,
                     real_t rho_v,
                     real_t rho_w) {
    EXPECT_DOUBLE_EQ(state.rho,   rho);
    EXPECT_DOUBLE_EQ(state.e_tot, e_tot);
    EXPECT_DOUBLE_EQ(state.rho_u, rho_u);
    EXPECT_DOUBLE_EQ(state.rho_v, rho_v);
    EXPECT_DOUBLE_EQ(state.rho_w, rho_w);
  };

  auto test_eq_st = [&](const ConsHydroState& state1,
                        const ConsHydroState& state2) {
    EXPECT_DOUBLE_EQ(state1.rho,   state2.rho);
    EXPECT_DOUBLE_EQ(state1.e_tot, state2.e_tot);
    EXPECT_DOUBLE_EQ(state1.rho_u, state2.rho_u);
    EXPECT_DOUBLE_EQ(state1.rho_v, state2.rho_v);
    EXPECT_DOUBLE_EQ(state1.rho_w, state2.rho_w);
  };

  // Default constructor
  ConsHydroState state0;
  test_eq(state0, 0.0, 0.0, 0.0, 0.0, 0.0);

  // Init constructor
  ConsHydroState state1 {393.23, 9.11, 0.023, 0.0001, -1.78e3};
  test_eq(state1, 393.23, 9.11, 0.023, 0.0001, -1.78e3);

  // Copy constructor
  ConsHydroState state2 {0.001, 2.3e1, 0.323, 0.87, -1.3e2};
  ConsHydroState state3{state2};
  test_eq_st(state3, state2);

  // operator +
  auto add = state1 + state2;
  test_eq(add, 
          state1.rho + state2.rho,
          state1.e_tot + state2.e_tot,
          state1.rho_u + state2.rho_u,
          state1.rho_v + state2.rho_v,
          state1.rho_w + state2.rho_w);

  real_t q1 = 837.222;
  auto add2 = state1 + q1;
  test_eq(add2,
          state1.rho + q1,
          state1.e_tot + q1,
          state1.rho_u + q1,
          state1.rho_v + q1,
          state1.rho_w + q1);

  // operator +=
  state3 += state1;
  test_eq_st(state3, add);

  state3 = state1;
  state3 += q1;
  test_eq_st(state3, add2);

  // operator -
  auto sub = state1 - state2;
  test_eq(sub,
          state1.rho - state2.rho,
          state1.e_tot - state2.e_tot,
          state1.rho_u - state2.rho_u,
          state1.rho_v - state2.rho_v,
          state1.rho_w - state2.rho_w);

  // operator*
  auto mul = state1 * state2;
  test_eq(mul,
          state1.rho * state2.rho,
          state1.e_tot * state2.e_tot,
          state1.rho_u * state2.rho_u,
          state1.rho_v * state2.rho_v,
          state1.rho_w * state2.rho_w);

  real_t q3 = -1.2394;
  auto mul2 = state2 * q3;
  test_eq(mul2,
          state2.rho * q3,
          state2.e_tot * q3,
          state2.rho_u * q3,
          state2.rho_v * q3,
          state2.rho_w * q3);

  auto mul3 = q3 * state2;
  test_eq_st(mul3, mul2);

  real_t q4 = 13.22;
  auto state4{state1};
  state4 *= q4;
  test_eq_st(state4, state1*q4);

  // operator/
  real_t q5 = 1.0e2;
  auto div = state1 / q5;
  test_eq(div,
          state1.rho / q5,
          state1.e_tot / q5,
          state1.rho_u / q5,
          state1.rho_v / q5,
          state1.rho_w / q5);

  state1 /= q5;
  test_eq_st(state1, div);
}

////// MHD

TEST( Test_MHD_States, PrimMHDState )
{
  auto test_eq = [&](const PrimMHDState& state, 
                     real_t rho, 
                     real_t p,
                     real_t u,
                     real_t v,
                     real_t w,
                     real_t Bx,
                     real_t By,
                     real_t Bz) {
    EXPECT_DOUBLE_EQ(state.rho, rho);
    EXPECT_DOUBLE_EQ(state.p,  p);
    EXPECT_DOUBLE_EQ(state.u,  u);
    EXPECT_DOUBLE_EQ(state.v,  v);
    EXPECT_DOUBLE_EQ(state.w,  w);
    EXPECT_DOUBLE_EQ(state.Bx, Bx);
    EXPECT_DOUBLE_EQ(state.By, By);
    EXPECT_DOUBLE_EQ(state.Bz, Bz);
  };

  auto test_eq_st = [&](const PrimMHDState& state1,
                        const PrimMHDState& state2) {
    EXPECT_DOUBLE_EQ(state1.rho, state2.rho);
    EXPECT_DOUBLE_EQ(state1.p,   state2.p);
    EXPECT_DOUBLE_EQ(state1.u,   state2.u);
    EXPECT_DOUBLE_EQ(state1.v,   state2.v);
    EXPECT_DOUBLE_EQ(state1.w,   state2.w);
    EXPECT_DOUBLE_EQ(state1.Bx,  state2.Bx);
    EXPECT_DOUBLE_EQ(state1.By,  state2.By);
    EXPECT_DOUBLE_EQ(state1.Bz,  state2.Bz);
  };

  // Default constructor
  PrimMHDState state0;
  test_eq(state0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

  // Init constructor
  PrimMHDState state1 {1.0, 2.56, 2.1, -7.8, 0.01, 1.2, -0.3, 24.3};
  test_eq(state1, 1.0, 2.56, 2.1, -7.8, 0.01, 1.2, -0.3, 24.3);

  // Copy constructor
  PrimMHDState state2 {0.1, 1.5, -1.0, -0.2, -9.2, -1.0, 3.0, -2.0};
  PrimMHDState state3{state2};
  test_eq_st(state3, state2);

  // operator +
  auto add = state1 + state2;
  test_eq(add, 
          state1.rho + state2.rho,
          state1.p   + state2.p,
          state1.u   + state2.u,
          state1.v   + state2.v,
          state1.w   + state2.w,
          state1.Bx  + state2.Bx,
          state1.By  + state2.By,
          state1.Bz  + state2.Bz);

  real_t q1 = 2.942;
  auto add2 = state1 + q1;
  test_eq(add2,
          state1.rho + q1,
          state1.p   + q1,
          state1.u   + q1,
          state1.v   + q1,
          state1.w   + q1,
          state1.Bx  + q1,
          state1.By  + q1,
          state1.Bz  + q1);

  // operator +=
  state3 += state1;
  test_eq_st(state3, add);

  state3 = state1;
  state3 += q1;
  test_eq_st(state3, add2);

  // operator -
  auto sub = state1 - state2;
  test_eq(sub,
          state1.rho - state2.rho,
          state1.p   - state2.p,
          state1.u   - state2.u,
          state1.v   - state2.v,
          state1.w   - state2.w,
          state1.Bx  - state2.Bx,
          state1.By  - state2.By,
          state1.Bz  - state2.Bz);

  // operator*
  auto mul = state1 * state2;
  test_eq(mul,
          state1.rho * state2.rho,
          state1.p   * state2.p,
          state1.u   * state2.u,
          state1.v   * state2.v,
          state1.w   * state2.w,
          state1.Bx  * state2.Bx,
          state1.By  * state2.By,
          state1.Bz  * state2.Bz);

  real_t q3 = 7.123927;
  auto mul2 = state2 * q3;
  test_eq(mul2,
          state2.rho * q3,
          state2.p   * q3,
          state2.u   * q3,
          state2.v   * q3,
          state2.w   * q3,
          state2.Bx  * q3,
          state2.By  * q3,
          state2.Bz  * q3);

  auto mul3 = q3 * state2;
  test_eq_st(mul3, mul2);

  real_t q4 = -137.22;
  auto state4{state1};
  state4 *= q4;
  test_eq_st(state4, state1*q4);

  // operator/
  real_t q5 = 0.289;
  auto div = state1 / q5;
  test_eq(div,
          state1.rho / q5,
          state1.p   / q5,
          state1.u   / q5,
          state1.v   / q5,
          state1.w   / q5,
          state1.Bx  / q5,
          state1.By  / q5,
          state1.Bz  / q5);

  state1 /= q5;
  test_eq_st(state1, div);
}

TEST( Test_MHD_States, ConsMHDState )
{
auto test_eq = [&](const ConsMHDState& state, 
                     real_t rho, 
                     real_t e_tot,
                     real_t rho_u,
                     real_t rho_v,
                     real_t rho_w,
                     real_t Bx,
                     real_t By,
                     real_t Bz) {
    EXPECT_DOUBLE_EQ(state.rho,   rho);
    EXPECT_DOUBLE_EQ(state.e_tot, e_tot);
    EXPECT_DOUBLE_EQ(state.rho_u, rho_u);
    EXPECT_DOUBLE_EQ(state.rho_v, rho_v);
    EXPECT_DOUBLE_EQ(state.rho_w, rho_w);
    EXPECT_DOUBLE_EQ(state.Bx,    Bx);
    EXPECT_DOUBLE_EQ(state.By,    By);
    EXPECT_DOUBLE_EQ(state.Bz,    Bz);
  };

  auto test_eq_st = [&](const ConsMHDState& state1,
                        const ConsMHDState& state2) {
    EXPECT_DOUBLE_EQ(state1.rho,   state2.rho);
    EXPECT_DOUBLE_EQ(state1.e_tot, state2.e_tot);
    EXPECT_DOUBLE_EQ(state1.rho_u, state2.rho_u);
    EXPECT_DOUBLE_EQ(state1.rho_v, state2.rho_v);
    EXPECT_DOUBLE_EQ(state1.rho_w, state2.rho_w);
    EXPECT_DOUBLE_EQ(state1.Bx,    state2.Bx);
    EXPECT_DOUBLE_EQ(state1.By,    state2.By);
    EXPECT_DOUBLE_EQ(state1.Bz,    state2.Bz);
  };

  // Default constructor
  ConsMHDState state0;
  test_eq(state0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

  // Init constructor
  ConsMHDState state1 {393.23, 9.11, 0.023, 0.0001, -1.78e3, -1.0, 5.2e3, 1.2343};
  test_eq(state1, 393.23, 9.11, 0.023, 0.0001, -1.78e3, -1.0, 5.2e3, 1.2343);

  // Copy constructor
  ConsMHDState state2 {0.001, 2.3e1, 0.323, 0.87, -1.3e2, 0.1, 0.2, 0.75};
  ConsMHDState state3{state2};
  test_eq_st(state3, state2);

  // operator +
  auto add = state1 + state2;
  test_eq(add, 
          state1.rho   + state2.rho,
          state1.e_tot + state2.e_tot,
          state1.rho_u + state2.rho_u,
          state1.rho_v + state2.rho_v,
          state1.rho_w + state2.rho_w,
          state1.Bx    + state2.Bx,
          state1.By    + state2.By,
          state1.Bz    + state2.Bz);

  real_t q1 = 837.222;
  auto add2 = state1 + q1;
  test_eq(add2,
          state1.rho + q1,
          state1.e_tot + q1,
          state1.rho_u + q1,
          state1.rho_v + q1,
          state1.rho_w + q1,
          state1.Bx + q1,
          state1.By + q1,
          state1.Bz + q1);

  // operator +=
  state3 += state1;
  test_eq_st(state3, add);

  state3 = state1;
  state3 += q1;
  test_eq_st(state3, add2);

  // operator -
  auto sub = state1 - state2;
  test_eq(sub,
          state1.rho - state2.rho,
          state1.e_tot - state2.e_tot,
          state1.rho_u - state2.rho_u,
          state1.rho_v - state2.rho_v,
          state1.rho_w - state2.rho_w,
          state1.Bx    - state2.Bx,
          state1.By    - state2.By,
          state1.Bz    - state2.Bz);

  // operator*
  auto mul = state1 * state2;
  test_eq(mul,
          state1.rho * state2.rho,
          state1.e_tot * state2.e_tot,
          state1.rho_u * state2.rho_u,
          state1.rho_v * state2.rho_v,
          state1.rho_w * state2.rho_w,
          state1.Bx    * state2.Bx,
          state1.By    * state2.By,
          state1.Bz    * state2.Bz);

  real_t q3 = -1.2394;
  auto mul2 = state2 * q3;
  test_eq(mul2,
          state2.rho * q3,
          state2.e_tot * q3,
          state2.rho_u * q3,
          state2.rho_v * q3,
          state2.rho_w * q3,
          state2.Bx    * q3,
          state2.By    * q3,
          state2.Bz    * q3);

  auto mul3 = q3 * state2;
  test_eq_st(mul3, mul2);

  real_t q4 = 13.22;
  auto state4{state1};
  state4 *= q4;
  test_eq_st(state4, state1*q4);

  // operator/
  real_t q5 = 1.0e2;
  auto div = state1 / q5;
  test_eq(div,
          state1.rho / q5,
          state1.e_tot / q5,
          state1.rho_u / q5,
          state1.rho_v / q5,
          state1.rho_w / q5,
          state1.Bx    / q5,
          state1.By    / q5,
          state1.Bz    / q5);

  state1 /= q5;
  test_eq_st(state1, div);
}

/// Test if get/setConservativeState(ConsHydroState) work correctly
void test_ConsHydroState_setget()
{
  using CellIndex = ForeachCell::CellIndex;
  using CellArray_global = ForeachCell::CellArray_global;

  uint32_t bx=1, by=1, bz=1;
  uint32_t nbOcts = 10;

  FieldManager field_manager({ID, IE, IU, IV, IW});

  DataArrayBlock U_("U_", bx*by*bz, field_manager.nbfields(), nbOcts);
  CellArray_global U{U_, bx, by, bz, nbOcts, field_manager.get_id2index()};

  auto make_iCell = KOKKOS_LAMBDA( uint32_t iOct, uint32_t i, uint32_t j, uint32_t k )
  {
    return CellIndex{ {iOct, false}, i, j, k, bx, by, bz, CellIndex::Status::LOCAL_TO_BLOCK };
  };

  Kokkos::parallel_for("ConsHydroState_set", nbOcts,
    KOKKOS_LAMBDA(uint32_t i)
  {
    ConsHydroState s{i+0.1,i+0.2,i+0.3,i+0.4,i+0.5};
    setConservativeState<3>( U, make_iCell(i, 0,0,0), s );
  });

  int errors = 0;
  Kokkos::parallel_reduce("ConsHydroState_get", nbOcts,
    KOKKOS_LAMBDA(uint32_t i, int& errors)
  {
    ConsHydroState s;
    getConservativeState<3>( U, make_iCell(i, 0,0,0), s );
    
    auto compare = [&]( real_t expected, real_t actual )
    {
      if( expected != actual )
      {
        errors ++;
        printf("Error : %f != %f \n", expected, actual);
      }
    };
    compare(i+0.1, s.rho);
    compare(i+0.2, s.e_tot);
    compare(i+0.3, s.rho_u);
    compare(i+0.4, s.rho_v);
    compare(i+0.5, s.rho_w);
  }, errors);

  EXPECT_EQ(0, errors);
}

TEST( Test_MHD_States, ConsHydroState_setget )
{
  test_ConsHydroState_setget();
}