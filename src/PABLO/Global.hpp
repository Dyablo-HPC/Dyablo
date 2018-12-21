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

#ifndef __BITPIT_PABLO_GLOBAL_HPP__
#define __BITPIT_PABLO_GLOBAL_HPP__

// =================================================================================== //
// INCLUDES                                                                            //
// =================================================================================== //
#include <stdint.h>

namespace bitpit {

/*!
 *	\ingroup		PABLO
 *	\date			23/apr/2014
 *	\authors		Edoardo Lombardi
 *	\authors		Marco Cisternino
 *	\copyright		Copyright 2014 Optimad engineering srl. All rights reserved.
 *
 *	\brief Global variables used in PABLO
 *
 *	Global variables are used in PABLO everywhere and they are public, i.e. each
 *	global variable can be used asant by external codes.
 *
 *	Class Global is a class with static members initialized during the construction
 *	of a paratree object.
 *
 */
class Global{

	// =================================================================================== //
	// FRIENDSHIPS
	// =================================================================================== //
	friend class ParaTree;
	friend class LocalTree;
	friend class Map;
	friend class Octant;

	// =================================================================================== //
	// MEMBERS
	// =================================================================================== //
private:
	static const int8_t   m_maxLevel;		/**< Maximum allowed refinement level of octree */
	static const uint32_t m_maxLength;		/**< Length of the logical domain */

	uint8_t  m_nchildren;			/**< Number of children of an octant */
	uint8_t  m_nfaces;				/**< Number of faces of an octant */
	uint8_t  m_nedges;				/**< Number of edges of an octant */
	uint8_t  m_nnodes;				/**< Number of nodes of an octant */
	uint8_t  m_nnodesPerFace;		/**< Number of nodes per face of an octant */
	uint8_t  m_oppFace[6];			/**< oppface[i] = Index of the face of an octant neighbour through the i-th face of the current octant */
	uint8_t  m_nodeFace[8][3];		/**< nodeface[i][0:1] = local indices of faces sharing the i-th node of an octant */
	uint8_t  m_nodeEdge[8][3];		/**< nodeedge[i][0:1] = local indices of edges sharing the i-th node of an octant */
	uint8_t  m_faceNode[6][4];		/**< facenode[i][0:1] = local indices of nodes of the i-th face of an octant */
	uint8_t  m_edgeFace[12][2];		/**< edgeface[i][0:1] = local indices of faces sharing the i-th edge of an octant */
	uint8_t  m_edgeNode[12][2];		/**< edgeNode[i][0:1] = local indices of nodes of the i-th edge of an octant */
	int8_t   m_normals[6][3];		/**< Components (x,y,z) of the normals per face (z=0 in 2D) */
	int8_t   m_edgeCoeffs[12][3];	/**< Components (x,y,z) of the "normals" per edge */
	int8_t   m_nodeCoeffs[8][3];	/**< Components (x,y,z) of the "normals" per node */
	uint8_t  m_parallelEdges[12][3];/**< Parallel edges per edge */

	// =================================================================================== //
	// METHODS
	// =================================================================================== //

	// =================================================================================== //
	// BASIC GET/SET METHODS
	// =================================================================================== //
	static int8_t 		getMaxLevel();
	static uint32_t 	getMaxLength();

	void 		getEdgecoeffs(int8_t edgecoeffs[12][3]) const;
	void 		getEdgeface(uint8_t edgeface[12][2]) const;
	void 		getEdgenode(uint8_t edgeNode[12][2]) const;
	void 		getEdgenode(uint8_t edge, uint8_t edgeNode[2]) const;
	void 		getFacenode(uint8_t facenode[6][4]) const;
	void 		getNodeedge(uint8_t nodeegde_[8][3]) const;
	uint8_t 	getNchildren() const;
	uint8_t 	getNedges() const;
	uint8_t 	getNfaces() const;
	uint8_t 	getNnodes() const;
	uint8_t 	getNnodesperface() const;
	void 		getNodecoeffs(int8_t nodecoeffs[8][3]) const;
	void 		getNodeface(uint8_t nodeface[8][3]) const;
	void 		getNormals(int8_t normals[6][3]) const;
	uint8_t 	getOctantBytes() const;
	void 		getOppface(uint8_t oppface[6]) const;
	void 		getParallelEdges(uint8_t parallelEdges[12][3]) const;
	void 		getParallelEdges(uint8_t edge, uint8_t parallelEdges[]) const;

	void 		initialize(uint8_t dim);

};

}

#endif /* __BITPIT_PABLO_GLOBAL_HPP__ */
