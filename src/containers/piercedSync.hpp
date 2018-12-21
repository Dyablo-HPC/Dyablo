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

#ifndef __BITPIT_PIERCED_SYNC_HPP__
#define __BITPIT_PIERCED_SYNC_HPP__

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

namespace bitpit {

/**
* \ingroup containers
*
* \brief Action for pierced synchronization.
*/
class PiercedSyncAction {

public:
    /**
    * Synchronization action type
    */
    enum ActionType {
        TYPE_UNDEFINED = -1,
        TYPE_NOOP,
        TYPE_CLEAR,
        TYPE_RESERVE,
        TYPE_RESIZE,
        TYPE_SHRINK_TO_FIT,
        TYPE_REORDER,
        TYPE_APPEND,
        TYPE_INSERT,
        TYPE_OVERWRITE,
        TYPE_MOVE_APPEND,
        TYPE_MOVE_INSERT,
        TYPE_MOVE_OVERWRITE,
        TYPE_SWAP,
        TYPE_PIERCE
    };

    /**
    * Synchronization action info
    */
    enum ActionInfo {
        INFO_POS        = 0,
        INFO_POS_FIRST  = 0,
        INFO_POS_SECOND = 1,
        INFO_POS_NEXT   = 1,
        INFO_SIZE       = 1,
        INFO_COUNT      = 2
    };

    PiercedSyncAction(ActionType _type = TYPE_UNDEFINED);
    PiercedSyncAction(const PiercedSyncAction &other);

    PiercedSyncAction & operator=(const PiercedSyncAction &other);
    PiercedSyncAction & operator=(PiercedSyncAction&&) = default;

    void swap(PiercedSyncAction &other) noexcept;

    void importData(const std::vector<std::size_t> &values);

    // Dump and restore
    void restore(std::istream &stream);
    void dump(std::ostream &stream) const;

    // Data
    ActionType type;
    std::array<std::size_t, INFO_COUNT> info;
    std::unique_ptr<std::vector<std::size_t>> data;

};

/**
* \ingroup containers
*
* \brief Base class for defining an object that acts like a slave in pierced
* synchronization.
*/
class PiercedSyncSlave {

friend class PiercedSyncMaster;

protected:
    PiercedSyncSlave();

    void swap(PiercedSyncSlave &x) noexcept;

    virtual void commitSyncAction(const PiercedSyncAction &action) = 0;

};

/**
* \ingroup containers
*
* \brief Base class for defining an object that acts like a master in pierced
* synchronization.
*/
class PiercedSyncMaster {

public:
    /**
    * Slave synchronization group definition
    */
    typedef std::vector<PiercedSyncSlave *> SyncGroup;

    /**
    * Synchronization mode
    */
    enum SyncMode {
        SYNC_MODE_CONCURRENT,
        SYNC_MODE_JOURNALED,
        SYNC_MODE_DISABLED,
        SYNC_MODE_ITR_COUNT = SYNC_MODE_DISABLED + 1,
        SYNC_MODE_ITR_BEGIN = 0,
        SYNC_MODE_ITR_END = SYNC_MODE_ITR_BEGIN + SYNC_MODE_ITR_COUNT
    };

protected:
    /**
    * Hash function for SyncMode enum
    */
    struct SyncModeHasher
    {
        template <typename T>
        std::size_t operator()(T t) const
        {
            return static_cast<std::size_t>(t);
        }
    };

    /**
    * Slaves
    */
    std::unordered_map<PiercedSyncSlave *, SyncMode> m_slaves;

    /**
    * Slave synchronization groups
    */
    std::unordered_map<SyncMode, SyncGroup, SyncModeHasher> m_syncGroups;

    /**
    * Slave synchronization groups
    */
    std::vector<PiercedSyncAction> m_syncJournal;

    PiercedSyncMaster();

    void registerSlave(PiercedSyncSlave *slave, PiercedSyncMaster::SyncMode syncMode);
    void unregisterSlave(const PiercedSyncSlave *slave);
    bool isSlaveRegistered(const PiercedSyncSlave *slave) const;
    PiercedSyncMaster::SyncMode getSlaveSyncMode(const PiercedSyncSlave *slave) const;

    void setSyncEnabled(bool enabled);
    bool isSyncEnabled() const;

    void sync();

    void swap(PiercedSyncMaster &x) noexcept;

    void processSyncAction(const PiercedSyncAction &action);

    // Dump and restore
    void restore(std::istream &stream);
    void dump(std::ostream &stream) const;

private:
    bool m_syncEnabled;

    void commitSyncAction(PiercedSyncSlave *slave, const PiercedSyncAction &action);
    void journalSyncAction(const PiercedSyncAction &action);

};

}

#endif
