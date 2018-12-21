/*---------------------------------------------------------------------------*\
 *
 *  bitpit
 *
 *  Copyright (C) 2015-2017 OPTIMAD engineering Srl
 *
 *  -------------------------------------------------------------------------
 *  License
 *  This file is part of bitpit.
 *
 *  bitpit is free software: you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License v3 (LGPL)
 *  as published by the Free Software Foundation.
 *
 *  bitpit is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with bitpit. If not, see <http://www.gnu.org/licenses/>.
 *
\*---------------------------------------------------------------------------*/

#ifndef __BITPIT_PABLO_MORTON_HPP__
#define __BITPIT_PABLO_MORTON_HPP__

namespace bitpit {

namespace PABLO {

/**
* Seperate bits from a given integer 3 positions apart.
*
* The function comes from libmorton (see https://github.com/Forceflow/libmorton).
*
* \param a is an integer position
* \result Separated bits.
*/
inline uint64_t splitBy3(uint32_t a)
{
    uint64_t x = a & 0x1fffff; // we only look at the first 21 bits
    x = (x | x << 32) & 0x1f00000000ffff;  // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
    x = (x | x << 16) & 0x1f0000ff0000ff;  // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
    x = (x | x << 8) & 0x100f00f00f00f00f; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
    x = (x | x << 4) & 0x10c30c30c30c30c3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
    x = (x | x << 2) & 0x1249249249249249;

    return x;
}

/**
* Seperate bits from a given integer 2 positions apart.
*
* The function comes from libmorton (see https://github.com/Forceflow/libmorton).
*
* \param a is an integer position
* \result Separated bits.
*/
inline uint64_t splitBy2(uint32_t a)
{
    uint64_t x = a;
    x = (x | x << 16) & 0xFFFF0000FFFF;  // shift left 16 bits, OR with self, and 0000000000000000111111111111111100000000000000001111111111111111
    x = (x | x << 8) & 0xFF00FF00FF00FF;  // shift left 8 bits, OR with self, and 0000000011111111000000001111111100000000111111110000000011111111
    x = (x | x << 4) & 0xF0F0F0F0F0F0F0F; // shift left 4 bits, OR with self, and 0000111100001111000011110000111100001111000011110000111100001111
    x = (x | x << 2) & 0x3333333333333333; // shift left 2 bits, OR with self, and 0011001100110011001100110011001100110011001100110011001100110011
    x = (x | x << 1) & 0x5555555555555555; // shift left 1 bits, OR with self, and 0101010101010101010101010101010101010101010101010101010101010101

    return x;
}

/**
* Compute the Morton number of the given set of coordinates.
*
* The function uses the "magic bits" algorithm of the libmorton library
* (see https://github.com/Forceflow/libmorton).
*
* \param x is the integer x position
* \param y is the integer y position
* \param z is the integer z position
* \result The Morton number.
*/
inline uint64_t computeMorton(uint32_t x, uint32_t y, uint32_t z)
{
    uint64_t morton = splitBy3(x) | splitBy3(y) << 1 | splitBy3(z) << 2;

    return morton;
}

/**
* Compute the Morton number of the given set of coordinates.
*
* The function uses the "magic bits" algorithm of the libmorton library
* (see https://github.com/Forceflow/libmorton).
*
* \param x is the integer x position
* \param y is the integer y position
* \result The Morton number.
*/
inline uint64_t computeMorton(uint32_t x, uint32_t y)
{
    uint64_t morton = splitBy2(x) | splitBy2(y) << 1;

    return morton;
}

/**
* Compute the XYZ key of the given set of coordinates.
*
* \param x is the integer x position
* \param y is the integer y position
* \param z is the integer z position
* \param maxLevel is the Maximum allowed refinement level of octree
* \result The Morton number.
*/
inline uint64_t computeXYZKey(uint64_t x, uint64_t y, uint64_t z, int8_t maxLevel)
{
    uint64_t key = x | (y << (maxLevel+1)) | (z << 2*(maxLevel+1));

    return key;
}

/**
* Compute the XYZ key number of the given set of coordinates.
*
* \param x is the integer x position
* \param y is the integer y position
* \param maxLevel is the Maximum allowed refinement level of octree
* \result The Morton number.
*/
inline uint64_t computeXYZKey(uint64_t x, uint64_t y, int8_t maxLevel)
{
    uint64_t key = x | (y << (maxLevel+1));

    return key;
}

}

}

#endif
