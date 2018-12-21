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

#ifndef __BITPIT_VOLOCTREE_HPP__
#define __BITPIT_VOLOCTREE_HPP__

#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "bitpit_PABLO.hpp"
#include "bitpit_patchkernel.hpp"

namespace bitpit {

class VolOctree : public VolumeKernel {

public:
	using VolumeKernel::isPointInside;
	using PatchKernel::locatePoint;

	struct OctantInfo {
		OctantInfo() : id(0), internal(true) {};
		OctantInfo(uint32_t _id, bool _internal) : id(_id), internal(_internal) {};

		bool operator==(const OctantInfo &other) const
		{
			return (id == other.id && internal == other.internal);
		}

		uint32_t id;
		bool internal;
	};

	struct OctantInfoHasher
	{
		// We can just use the id for the hash, beacuse only two different
		// octants can be assigned to the same id: the internal octant and
		// the ghost octant. It's not worth including the ghost flag in the
		// hash.
		std::size_t operator()(const OctantInfo& k) const
		{
			return std::hash<uint32_t>()(k.id);
		}
	};

	VolOctree();
	VolOctree(const int &dimension, std::array<double, 3> origin, double length, double dh);
	VolOctree(const int &id, const int &dimension, std::array<double, 3> origin, double length, double dh);
	VolOctree(std::unique_ptr<PabloUniform> &&tree, std::unique_ptr<PabloUniform> *adopter = nullptr);
	VolOctree(const int &id, std::unique_ptr<PabloUniform> &&tree, std::unique_ptr<PabloUniform> *adopter = nullptr);
	VolOctree(std::istream &stream);

	~VolOctree();

	std::unique_ptr<PatchKernel> clone() const override;

	void reset() override;
	void setDimension(int dimension) override;

	void settleAdaptionMarkers() override;

	double evalCellVolume(const long &id) const override;
	double evalCellSize(const long &id) const override;
	std::array<double, 3> evalCellCentroid(const long &id) const override;
	void evalCellBoundingBox(long id, std::array<double,3> *minPoint, std::array<double,3> *maxPoint) const override;

	double evalInterfaceArea(const long &id) const override;
	std::array<double, 3> evalInterfaceNormal(const long &id) const override;

	OctantInfo getCellOctant(const long &id) const;
	int getCellLevel(const long &id) const;

	long getOctantId(const OctantInfo &octantInfo) const;
	Octant * getOctantPointer(const OctantInfo &octantInfo);
	const Octant * getOctantPointer(const OctantInfo &octantInfo) const;

	PabloUniform & getTree();
	const PabloUniform & getTree() const;
	void setTreeAdopter(std::unique_ptr<PabloUniform> *entruster);

	bool isPointInside(const std::array<double, 3> &point) override;
	bool isPointInside(const long &id, const std::array<double, 3> &point) override;
	long locatePoint(const std::array<double, 3> &point) override;

	std::array<double, 3> getOrigin() const;
	void setOrigin(const std::array<double, 3> &origin);
	void translate(std::array<double, 3> translation) override;
	double getLength() const;
	void setLength(double length);
	void scale(std::array<double, 3> scaling) override;

	void updateAdjacencies(const std::vector<long> &cellIds, bool resetAdjacencies = true) override;

#if BITPIT_ENABLE_MPI==1
	void setCommunicator(MPI_Comm communicator) override;

	int getCellHaloLayer(const long &id) const override;
#endif

protected:
	VolOctree(const VolOctree &other);

	std::vector<adaption::Info> _spawn(bool trackSpawn) override;

	std::vector<adaption::Info> _adaptionPrepare(bool trackAdaption) override;
	std::vector<adaption::Info> _adaptionAlter(bool trackAdaption) override;
	void _adaptionCleanup() override;
	bool _markCellForRefinement(const long &id) override;
	bool _markCellForCoarsening(const long &id) override;
	bool _resetCellAdaptionMarker(const long &id) override;
	adaption::Marker _getCellAdaptionMarker(const long &id) override;
	bool _enableCellBalancing(const long &id, bool enabled) override;
	void _setTol(double tolerance) override;
	void _resetTol() override;

	int _getDumpVersion() const override;
	void _dump(std::ostream &stream) const override;
	void _restore(std::istream &stream) override;

	long _getCellNativeIndex(long id) const override;

	void _findCellEdgeNeighs(const long &id, const int &edge, const std::vector<long> &blackList, std::vector<long> *neighs) const override;
	void _findCellVertexNeighs(const long &id, const int &vertex, const std::vector<long> &blackList, std::vector<long> *neighs) const override;

#if BITPIT_ENABLE_MPI==1
	std::size_t _getMaxHaloSize() override;
	void _setHaloSize(std::size_t haloSize) override;

	std::vector<adaption::Info> _partitioningPrepare(bool trackPartitioning) override;
	std::vector<adaption::Info> _partitioningAlter(bool trackPartitioning) override;
	void _partitioningCleanup() override;

	std::vector<long> _findGhostExchangeSources(int rank) override;
#endif

private:
	using PatchKernel::setBoundingBox;

	typedef std::bitset<72> OctantHash;

	struct RenumberInfo {
		RenumberInfo()
			: cellId(Cell::NULL_ID), newTreeId(0)
		{
		};

		RenumberInfo(long _cellId, uint32_t _newTreeId)
			: cellId(_cellId), newTreeId(_newTreeId)
		{
		};

		long cellId;
		uint32_t newTreeId;
	};

	struct DeleteInfo {
		DeleteInfo()
			: cellId(Element::NULL_ID), trigger(adaption::TYPE_UNKNOWN), rank(-1)
		{
		};

		DeleteInfo(long _cellId, adaption::Type _trigger, int _rank = -1)
			: cellId(_cellId), trigger(_trigger), rank(_rank)
		{
		};

		long cellId;
		adaption::Type trigger;
		int rank;
	};

	struct FaceInfo {
		FaceInfo() : id(Element::NULL_ID), face(-1) {};
		FaceInfo(long _id, int _face) : id(_id), face(_face) {};

		bool operator==(const FaceInfo &other) const
		{
			return (id == other.id && face == other.face);
		}

		long id;
		int face;
	};

	struct FaceInfoHasher
	{
		// We can just use the id for the hash, because a cell can have only
		// a limited amount of faces. It's not worth including the face index
		// in the hash.
		std::size_t operator()(const FaceInfo& k) const
		{
			return std::hash<long>()(k.id);
		}
	};

	typedef std::unordered_set<FaceInfo, FaceInfoHasher> FaceInfoSet;

	typedef std::unordered_map<uint64_t, long> StitchInfo;

	std::vector<std::vector<int>> m_octantLocalFacesOnVertex;
	std::vector<std::vector<int>> m_octantLocalFacesOnEdge;
	std::vector<std::vector<int>> m_octantLocalEdgesOnVertex;

	const ReferenceElementInfo *m_cellTypeInfo;
	const ReferenceElementInfo *m_interfaceTypeInfo;

	std::unordered_map<long, uint32_t, Element::IdHasher> m_cellToOctant;
	std::unordered_map<long, uint32_t, Element::IdHasher> m_cellToGhost;
	std::unordered_map<uint32_t, long> m_octantToCell;
	std::unordered_map<uint32_t, long> m_ghostToCell;

	std::unique_ptr<PabloUniform> m_tree;
	std::unique_ptr<PabloUniform> *m_treeAdopter;

	void initialize();

	void setBoundingBox();

	void __reset(bool resetTree);
	void __setDimension(int dimension);

	bool setMarker(const long &id, const int8_t &value);

	OctantHash evaluateOctantHash(const OctantInfo &octantInfo);

	StitchInfo deleteCells(const std::vector<DeleteInfo> &deletedOctants);
	void renumberCells(const std::vector<RenumberInfo> &renumberedOctants);
	std::vector<long> importCells(const std::vector<OctantInfo> &octantTreeIds, StitchInfo &stitchInfo, std::istream *stream = nullptr);

	std::vector<adaption::Info> sync(bool trackChanges);

	void findCellCodimensionNeighs(const long &id, const int &index,
		const int &codimension, const std::vector<long> &blackList, std::vector<long> *neighs) const;
};

}

#endif
