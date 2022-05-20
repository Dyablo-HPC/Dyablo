/**
 * Testing the states implementation and conversion
 **/

#include "utils/mpi/GlobalMpiSession.h"
#include "utils/monitoring/Timers.h"
#include "foreach_cell/ForeachCell.h"
#include "compute_dt/Compute_dt.h"
#include "init/InitialConditions.h"
#include "HydroState.h"
#include "io/IOManager.h"
using blockSize_t    = Kokkos::Array<uint32_t, 3>;

using Device = Kokkos::DefaultExecutionSpace;

#include "gtest/gtest.h"

using namespace dyablo;

TEST( Test_HydroStates, StateNd_manipulation )
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

TEST( Test_HydroStates, HydroState2d_legacy ) 
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

TEST( Test_HydroStates, HydroState3d_legacy ) 
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

TEST( Test_HydroStates, PrimHydroState )
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

TEST( Test_HydroStates, ConsHydroState )
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