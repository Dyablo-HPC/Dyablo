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

#include "shared/morton_utils.h"

#include <boost/test/auto_unit_test.hpp>
namespace utf = boost::unit_test;

#include <cstdint>

BOOST_AUTO_TEST_SUITE(dyablo)

BOOST_AUTO_TEST_SUITE(shared)

BOOST_AUTO_TEST_CASE(morton_utils, * utf::tolerance(0.00001))
{

  uint32_t value = 2 + 4; // 0x10  + 0x100
  auto v1 = dyablo::splitBy3<3>(value);
  BOOST_CHECK_EQUAL(v1, 72);

  value = 20;
  v1 = dyablo::splitBy3<3>(value);
  BOOST_CHECK_EQUAL(v1, 4160);

} // morton_utils


BOOST_AUTO_TEST_SUITE_END() /* shared */

BOOST_AUTO_TEST_SUITE_END() /* dyablo */
