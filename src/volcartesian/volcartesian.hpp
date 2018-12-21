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

#ifndef __BITPIT_VOLCARTESIAN_HPP__
#define __BITPIT_VOLCARTESIAN_HPP__

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "bitpit_patchkernel.hpp"

namespace bitpit {

class VolCartesian : public VolumeKernel {

public:
	using VolumeKernel::isPointInside;
	using PatchKernel::locatePoint;
	using PatchKernel::getCellType;
	using PatchKernel::getInterfaceType;

	enum MemoryMode {
		MEMORY_NORMAL,
		MEMORY_LIGHT
	};

	VolCartesian();
	VolCartesian(const int &dimension, const std::array<double, 3> &origin,
			   const std::array<double, 3> &lengths, const std::array<int, 3> &nCells);
	VolCartesian(const int &id, const int &dimension, const std::array<double, 3> &origin,
			   const std::array<double, 3> &lengths, const std::array<int, 3> &nCells);
	VolCartesian(const int &dimension, const std::array<double, 3> &origin,
			   double length, int nCells1D);
	VolCartesian(const int &id, const int &dimension, const std::array<double, 3> &origin,
			   double length, int nCells1D);
	VolCartesian(const int &dimension, const std::array<double, 3> &origin,
			   double length, double dh);
	VolCartesian(const int &id, const int &dimension, const std::array<double, 3> &origin,
			   double length, double dh);
	VolCartesian(std::istream &stream);

	std::unique_ptr<PatchKernel> clone() const override;

	void reset() override;

	void setDiscretization(const std::array<int, 3> &nCells);

	long getVertexCount() const override;

	long getCellCount() const override;
	ElementType getCellType() const;
	ElementType getCellType(const long &id) const override;

	long getInterfaceCount() const override;
	ElementType getInterfaceType() const;
	ElementType getInterfaceType(const long &id) const override;

	double evalCellVolume(const long &id) const override;
	double evalCellSize(const long &id) const override;
	std::array<double, 3> evalCellCentroid(const long &id) const override;

	double evalInterfaceArea(const long &id) const override;
	std::array<double, 3> evalInterfaceNormal(const long &id) const override;

	std::array<double, 3> getSpacing() const;
	double getSpacing(const int &direction) const;

	void switchMemoryMode(MemoryMode mode);
	MemoryMode getMemoryMode();

	bool isPointInside(const std::array<double, 3> &point) override;
	bool isPointInside(const long &id, const std::array<double, 3> &point) override;
	long locatePoint(const std::array<double, 3> &point) override;
	std::array<int, 3> locatePointCartesian(const std::array<double, 3> &point);
	long locateClosestVertex(std::array<double,3> const &point) const;
	std::array<int, 3> locateClosestVertexCartesian(std::array<double,3> const &point) const;
	long locateClosestCell(std::array<double,3> const &point);
	std::array<int, 3> locateClosestCellCartesian(std::array<double,3> const &point);

	std::vector<long> extractCellSubSet(std::array<int, 3> const &ijkMin, std::array<int, 3> const &ijkMax);
	std::vector<long> extractCellSubSet(int const &idxMin, int const &idxMax);
	std::vector<long> extractCellSubSet(std::array<double, 3> const &pointMin, std::array<double, 3> const &pointMax);
	std::vector<long> extractVertexSubSet(std::array<int, 3> const &ijkMin, std::array<int, 3> const &ijkMax);
	std::vector<long> extractVertexSubSet(int const &idxMin, int const &idxMax);
	std::vector<long> extractVertexSubSet(std::array<double, 3> const &pointMin, std::array<double, 3> const &pointMax);

	std::array<double, 3> getOrigin() const;
	void setOrigin(const std::array<double, 3> &origin);
	void translate(std::array<double, 3> translation) override;
	std::array<double, 3> getLengths() const;
	void setLengths(const std::array<double, 3> &lengths);
	void scale(std::array<double, 3> scaling) override;

	std::vector<double> convertToVertexData(const std::vector<double> &cellData) const;
	std::vector<double> convertToCellData(const std::vector<double> &nodeData) const;

	int linearCellInterpolation(std::array<double,3> &point,
		std::vector<int> &stencil, std::vector<double> &weights);
	int linearVertexInterpolation(std::array<double,3> &point,
		std::vector<int> &stencil, std::vector<double> &weights);

	long getCellLinearId(const int &i, const int &j, const int &k) const;
	long getCellLinearId(const std::array<int, 3> &ijk) const;
	std::array<int, 3> getCellCartesianId(long const &idx) const;
	bool isCellCartesianIdValid(const std::array<int, 3> &ijk) const;
	long getVertexLinearId(const int &i, const int &j, const int &k) const;
	long getVertexLinearId(const std::array<int, 3> &ijk) const;
	std::array<int, 3> getVertexCartesianId(long const &idx) const;
	std::array<int, 3> getVertexCartesianId(long const &cellIdx, int const &vertex) const;
	std::array<int, 3> getVertexCartesianId(const std::array<int, 3> &cellIjk, int const &vertex) const;
	bool isVertexCartesianIdValid(const std::array<int, 3> &ijk) const;

protected:
	VolCartesian(const VolCartesian &other) = default;

	std::vector<adaption::Info> _spawn(bool trackSpawn) override;

	int _getDumpVersion() const override;
	void _dump(std::ostream &stream) const override;
	void _restore(std::istream &stream) override;

	void _findCellFaceNeighs(const long &id, const int &face, const std::vector<long> &blackList, std::vector<long> *neighs) const override;
	void _findCellEdgeNeighs(const long &id, const int &edge, const std::vector<long> &blackList, std::vector<long> *neighs) const override;
	void _findCellVertexNeighs(const long &id, const int &vertex, const std::vector<long> &blackList, std::vector<long> *neighs) const override;

private:
	MemoryMode m_memoryMode;

	std::array<double, 3> m_cellSpacings;
	std::array<double, 3> m_minCoords;
	std::array<double, 3> m_maxCoords;

	std::array<std::vector<double>, 3> m_vertexCoords;
	std::array<std::vector<double>, 3> m_cellCenters;

	std::array<int, 3> m_nCells1D;
	std::array<int, 3> m_nVertices1D;

	long m_nVertices;
	long m_nCells;
	long m_nInterfaces;

	double m_cellVolume;
	std::array<double, 3> m_interfaceArea;
	std::array<std::array<double, 3>, 6> m_normals;

	std::vector<std::array<int, 3>> m_vertexNeighDeltas;
	std::vector<std::array<int, 3>> m_edgeNeighDeltas;
	std::vector<std::array<int, 2>> m_edgeFaces;

	void initialize();
	void initializeInterfaceArea();
	void initializeCellVolume();

	void setMemoryMode(MemoryMode mode);

	void addVertices();
	std::array<double, 3> evalVertexCoords(const long &id);

	void addCells();

	void addInterfaces();
	std::array<int, 3> getInterfaceCountDirection(const int &direction);
	void addInterfacesDirection(const int &direction);

};

}

#endif
