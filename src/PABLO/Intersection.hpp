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

#ifndef __BITPIT_PABLO_INTERSECTION_HPP__
#define __BITPIT_PABLO_INTERSECTION_HPP__

// =================================================================================== //
// INCLUDES                                                                            //
// =================================================================================== //
#include <stdint.h>

namespace bitpit {

// =================================================================================== //
// NAME SPACES                                                                         //
// =================================================================================== //

// =================================================================================== //
// CLASS DEFINITION                                                                    //
// =================================================================================== //
/*!
 *	\ingroup		PABLO
 *	\date			16/dec/2015
 *	\authors		Edoardo Lombardi
 *	\authors		Marco Cisternino
 *	\copyright		Copyright 2014 Optimad engineering srl. All rights reserved.
 *
 *	\brief Intersection class definition
 *
 *	The intersection is the face (edge in 2D) or portion of face shared by two octants.
 *	An intersection is defined by :
 *	- the owner octants, i.e. the octants sharing the intersection,
 *	identified by a couple (array[2]) of indices;
 *	- the index of the face, that contains the intersection, of the first owner;
 *	- an identifier of the octant in the couple with higher
 *	level of refinement (0/1) [if same level identifier =0];
 *	- a flag stating if an owner is ghost;
 *	- a flag to communicate if the intersection is new after a mesh refinement.
 *
 */
class Intersection{

	// =================================================================================== //
	// FRIENDSHIPS
	// =================================================================================== //

	friend class LocalTree;
	friend class ParaTree;

	// =================================================================================== //
	// TYPEDEFS
	// =================================================================================== //

	// =================================================================================== //
	// MEMBERS
	// =================================================================================== //
private:
	uint32_t 	m_owners[2];		/**< Owner octants of the intersection (first is the internal octant) */
	uint8_t   	m_iface;			/**< Index of the face of the outer owner */
	bool		m_out;				/**< 0/1 octant with exiting normal (if boundary =0) */
	bool		m_outisghost;		/**< 0/1 if octant with exiting normal is a ghost octant */
	bool		m_finer;			/**< 0/1 finer octant (if same level =0) */
	bool		m_isghost;			/**< The intersection has a member ghost */
	bool		m_isnew;			/**< The intersection is new after a mesh adapting? */
	bool		m_bound;			/**< The intersection is a boundary intersection of the whole domain */
	bool		m_pbound;			/**< The intersection is a boundary intersection of a process domain */
	uint8_t		m_dim;				/**< Dimension of intersection (2D/3D) */

	// =================================================================================== //
	// CONSTRUCTORS AND OPERATORS
	// =================================================================================== //
public:
	Intersection();
	Intersection(const Intersection & intersection);
	Intersection & operator =(const Intersection & intersection);
private:
	Intersection(uint8_t dim);
	bool operator ==(const Intersection & intersection);

	// =================================================================================== //
	// METHODS
	// =================================================================================== //

	// =================================================================================== //
	// BASIC GET/SET METHODS
	// =================================================================================== //

	uint32_t getOut() const;
	bool getOutIsGhost() const;
	uint32_t getIn() const;
	uint32_t getFiner() const;
	void getNormal(int8_t normal[3], int8_t normals[6][3]) const;
	bool getBound() const;
	bool getIsGhost() const;
	bool getPbound() const;

};

}

#endif /* __BITPIT_PABLO_INTERSECTION_HPP__ */
