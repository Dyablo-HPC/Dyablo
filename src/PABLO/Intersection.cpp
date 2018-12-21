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

// =================================================================================== //
// INCLUDES                                                                            //
// =================================================================================== //
#include "Intersection.hpp"

namespace bitpit {

// =================================================================================== //
// CLASS IMPLEMENTATION                                                                    //
// =================================================================================== //

// =================================================================================== //
// CONSTRUCTORS AND OPERATORS
// =================================================================================== //
/*! Default constructor of Intersection.
 * It build a void 2D, no boundary neither process boundary intersection.
 * Every member set to a default value 0.
 */
Intersection::Intersection(){
	m_owners[0] = 0;
	m_owners[1] = 0;
	m_iface = 0;
	m_isnew = false;
	m_isghost = false;
	m_finer = 0;
	m_out = 0;
	m_outisghost = false;
	m_bound = m_pbound = false;
	m_dim = 2;
};

/*! Custom constructor of Intersection with user set dimension.
 * It build a void, no boundary neither process boundary intersection.
 * Every member set to a default value 0.
 * \param[in] dim The space dimension of the octree. 2D is the default value.
 */
Intersection::Intersection(uint8_t dim){
	m_owners[0] = 0;
	m_owners[1] = 0;
	m_iface = 0;
	m_isnew = false;
	m_isghost = false;
	m_finer = 0;
	m_out = 0;
	m_outisghost = false;
	m_bound = m_pbound = false;
	m_dim = dim;
};

/*! Copy constructor of Intersection.
 * \param[in] intersection Intersection to be copied.
 */
Intersection::Intersection(const Intersection & intersection){
	m_owners[0] = intersection.m_owners[0];
	m_owners[1] = intersection.m_owners[1];
	m_iface = intersection.m_iface;
	m_isnew = intersection.m_isnew;
	m_isghost = intersection.m_isghost;
	m_finer = intersection.m_finer;
	m_out = intersection.m_out;
	m_outisghost = intersection.m_outisghost;
	m_bound = intersection.m_bound;
	m_pbound = intersection.m_pbound;
	m_dim = intersection.m_dim;
};

/*! Assigmnement operator of Intersection.
 * \param[in] intersection Intersection to be assigned.
 */
Intersection& Intersection::operator =(const Intersection & intersection){
	m_owners[0] = intersection.m_owners[0];
	m_owners[1] = intersection.m_owners[1];
	m_iface = intersection.m_iface;
	m_isnew = intersection.m_isnew;
	m_isghost = intersection.m_isghost;
	m_finer = intersection.m_finer;
	m_out = intersection.m_out;
	m_outisghost = intersection.m_outisghost;
	m_bound = intersection.m_bound;
	m_pbound = intersection.m_pbound;
	m_dim = intersection.m_dim;
	return *this;
};

/*! Equality operator of Intersection.
 * \param[in] intersection Intersection to be compared.
 */
bool Intersection::operator ==(const Intersection & intersection){
	bool check = true;
	check = check && (m_owners[0] == intersection.m_owners[0]);
	check = check && (m_owners[1] == intersection.m_owners[1]);
	check = check && (m_iface == intersection.m_iface);
	check = check && (m_isnew == intersection.m_isnew);
	check = check && (m_isghost == intersection.m_isghost);
	check = check && (m_out == intersection.m_out);
	check = check && (m_outisghost == intersection.m_outisghost);
	check = check && (m_finer == intersection.m_finer);
	check = check && (m_bound == intersection.m_bound);
	check = check && (m_pbound == intersection.m_pbound);
	check = check && (m_dim == intersection.m_dim);
	return check;

};

// =================================================================================== //
// METHODS
// =================================================================================== //

// =================================================================================== //
// BASIC GET/SET METHODS
// =================================================================================== //
/*!Get the owner with exiting normal;
 */
uint32_t Intersection::getOut() const {
	return m_owners[m_out];
};

/*!Get the owner with entering normal;
 */
uint32_t Intersection::getIn() const {
	return m_owners[!m_out];
};

/*!Get the owner with exiting normal;
 */
bool Intersection::getOutIsGhost() const {
	return m_outisghost;
};

/*!Get the owner with smaller size;
 */
uint32_t Intersection::getFiner() const {
	return m_owners[m_finer];
};

/*!Get the direction of the exiting normal;
 * \param[out] normal Components of the exiting normal.
 * \param[in] normals Basic matrix with components of the elementary normals.
 */
void Intersection::getNormal(int8_t normal[3], int8_t normals[6][3]) const {
	for (int i=0; i<m_dim; i++){
		normal[i] = normals[m_iface][i];
	}
};

/*!Get the boundary condition of the intersection;
 * \return Boolean true/false if the intersection is/is not a boundary intersection
 */
bool Intersection::getBound() const {
	return m_bound;
};

/*!Get the ghost information about the intersection;
 * \return Boolean true/false if the intersection is/is not a ghost intersection
 */
bool Intersection::getIsGhost() const {
	return m_isghost;
};

/*!Get the partition boundary condition of the intersection;
 * \return Boolean true/false if the intersection is/is not a process boundary intersection
 */
bool Intersection::getPbound() const {
	return m_pbound;
};

}
