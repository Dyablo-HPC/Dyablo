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

#include "patch_kernel.hpp"
#include "patch_manager.hpp"

namespace bitpit {

/*!
	\class PatchManager
	\ingroup patchkernel

	\brief The PatchManager oversee the handling of the patches.
*/

const int PatchManager::AUTOMATIC_ID = IndexGenerator<int>::NULL_ID;

/*
    Initialize logger manager instance.
*/
std::unique_ptr<PatchManager> PatchManager::m_manager = nullptr;

/*!
    Returns an instance of the patch manager.

    \result An instance of the patch manager.
*/
PatchManager & PatchManager::manager()
{
    if (!m_manager) {
        m_manager = std::unique_ptr<PatchManager>(new PatchManager());
    }

    return *m_manager;
}

/*!
	Get the patch with the specified id.

	\param id is the id of the patch
*/
PatchKernel * PatchManager::get(int id)
{
	for (const auto &entry : m_patchIds) {
		long patchId = entry.second;
		if (patchId == id) {
			return entry.first;
		}
	}

	return nullptr;
}

/*!
	Registers a patch in the manager

	\param patch is a pointer to the patch to be registered
	\param id is the id that will be assigned to the patch
	\result The id assigned to the patch.
*/
int PatchManager::registerPatch(PatchKernel *patch, int id)
{
	if (id >= 0) {
		if (m_idGenerator.isAssigned(id)) {
			throw std::runtime_error ("A patch with the same id already exists");
		}

		m_idGenerator.setAssigned(id);
	} else {
		id = m_idGenerator.generate();
	}

	patch->setId(id);
	m_patchIds[patch] = id;
	m_patchOrder.push_back(patch);

	return id;
}

/*!
	Un-registers a patch in the manager

	\param patch is a pointer to the patch to be un-registered
*/
void PatchManager::unregisterPatch(PatchKernel *patch)
{
	auto iterator = m_patchIds.find(patch);
	if (iterator == m_patchIds.end()) {
		throw std::runtime_error ("The patch to be unregistered does not exist");
	}

	int id = iterator->second;
	m_idGenerator.trash(id);

	m_patchIds.erase(iterator);
	for (auto itr = m_patchOrder.begin(); itr != m_patchOrder.end(); ++itr) {
		if (*itr == patch) {
			m_patchOrder.erase(itr);
			break;
		}
	}
}

/*!
	Creates a new patch manager.
*/
PatchManager::PatchManager()
{
}

/*!
 *  Write the patch manager data to the specified stream.
 *
 *  \param stream is the stream to write to
 */
void PatchManager::dump(std::ostream &stream)
{
	m_idGenerator.dump(stream);
}

/*!
 *  Restore the patch manager data from the specified stream.
 *
 *  \param stream is the stream to read from
 */
void PatchManager::restore(std::istream &stream)
{
	m_idGenerator.restore(stream);
}

/*!
 *  Write the registered patches and the patch manager data to the specified
 *  stream.
 *
 *  \param stream is the stream to write to
 */
void PatchManager::dumpAll(std::ostream &stream)
{
	for (PatchKernel *patch : m_patchOrder) {
		patch->dump(stream);
	}

	dump(stream);
}

/*!
 *  Restore the registered patches and the patch manager data from the
 *  specified stream.
 *
 *  \param stream is the stream to read from
 */
void PatchManager::restoreAll(std::istream &stream)
{
	m_idGenerator.reset();

	for (PatchKernel *patch : m_patchOrder) {
		patch->restore(stream);
	}

	restore(stream);
}

// Patch manager global functions
namespace patch {

    // Generic global functions

    /*!
        Returns the logger manager.

        \result The logger manager.
    */
    PatchManager & manager()
    {
        return PatchManager::manager();
    }

}

}
