/**
 * @file morton_utils_test.cpp
 * @date May 25, 2020
 * @author pkestene
 * @note
 *
 * Contributors: pkestene
 *
 * This file is part of the dyablo software project.
 *
 * @copyright © Commissariat a l'Energie Atomique et aux Energies Alternatives (CEA)
 *
 */

#include "morton_utils.h"

#include "gtest/gtest.h"

#include <cstdint>

TEST(Test_morton_utils, test_6)
{
  uint32_t value = 2 + 4; // 0x10  + 0x100
  auto v1 = dyablo::splitBy3<3>(value);
  EXPECT_EQ(v1, 72);
}

TEST(Test_morton_utils, test_20)
{
  uint32_t value = 20;
  auto v1 = dyablo::splitBy3<3>(value);
  EXPECT_EQ(v1, 4160);
}

