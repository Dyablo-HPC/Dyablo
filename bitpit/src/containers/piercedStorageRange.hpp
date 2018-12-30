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

#ifndef __BITPIT_PIERCED_STORAGE_RANGE_HPP__
#define __BITPIT_PIERCED_STORAGE_RANGE_HPP__

#include <stdexcept>

#include "piercedKernel.hpp"
#include "piercedKernelRange.hpp"

namespace bitpit {

template<typename PKR_id_t>
class PiercedKernelRange;

template<typename PS_value_t, typename PS_id_t>
class PiercedStorage;

/*!
    @brief The PiercedStorageRange allow to iterate using range-based loops over
    a PiercedStorage.
*/
template<typename value_t, typename id_t = long,
         typename value_no_cv_t = typename std::remove_cv<value_t>::type>
class PiercedStorageRange : protected PiercedKernelRange<id_t>
{

friend class PiercedStorageRange<value_no_cv_t, id_t, value_no_cv_t>;

template<typename PS_value_t, typename PS_id_t>
friend class PiercedStorage;

private:
    /**
    * Storage.
    */
    template<typename PS_value_t, typename PS_id_t>
    using Storage = PiercedStorage<PS_value_t, PS_id_t>;

    /**
    * Storage type
    *
    * When building a const_iterator the pointer to the storage has to be
    * declared const.
    */
    typedef
        typename std::conditional<std::is_const<value_t>::value,
            const Storage<value_no_cv_t, id_t>,
            Storage<value_no_cv_t, id_t>
        >::type

        storage_t;

    /*
    * Iterator type
    *
    * When building a const_iterator the pointer to the container has to
    * be declared const.
    */
    typedef
        typename std::conditional<std::is_const<value_t>::value,
            typename storage_t::const_iterator,
            typename storage_t::iterator
        >::type

        iterator_t;

    /*
    * Const iterator type
    *
    * When building a const_iterator the pointer to the container has to
    * be declared const.
    */
    typedef typename storage_t::const_iterator const_iterator_t;

public:
    /*! Type of container */
    typedef storage_t storage_type;

    /*! Type of data stored in the container */
    typedef value_t value_type;

    /*! Type of ids stored in the container */
    typedef id_t id_type;

    /*! Type of iterator */
    typedef iterator_t iterator;

    /*! Type of constant iterator */
    typedef const_iterator_t const_iterator;

    // Constructors
    PiercedStorageRange();
    PiercedStorageRange(storage_t *storage);
    PiercedStorageRange(storage_t *storage, id_t first, id_t last);
    PiercedStorageRange(const iterator &begin, const iterator &end);

    // General methods
    using PiercedKernelRange<id_t>::evalSize;

    void swap(PiercedStorageRange &other) noexcept;

    const PiercedKernelRange<id_t> & getKernelRange() const;

    // Methods to get begin and end
    template<typename U = value_t, typename U_no_cv = value_no_cv_t,
             typename std::enable_if<std::is_same<U, U_no_cv>::value, int>::type = 0>
    iterator begin() noexcept;

    template<typename U = value_t, typename U_no_cv = value_no_cv_t,
             typename std::enable_if<std::is_same<U, U_no_cv>::value, int>::type = 0>
    iterator end() noexcept;

    const_iterator begin() const noexcept;
    const_iterator end() const noexcept;

    const_iterator cbegin() const noexcept;
    const_iterator cend() const noexcept;

    /*!
        Two-way comparison.
    */
    template<typename other_value_t, typename other_id_t = long>
    bool operator==(const PiercedStorageRange<other_value_t, other_id_t> &rhs) const
    {
        if (PiercedKernelRange<id_t>::operator!=(rhs)) {
            return false;
        }

        return true;
    }

    /*!
    * Two-way comparison.
    */
    template<typename other_value_t, typename other_id_t = long>
    bool operator!=(const PiercedStorageRange<other_value_t, other_id_t> &rhs) const
    {
        if (PiercedKernelRange<id_t>::operator!=(rhs)) {
            return true;
        }

        return false;
    }

private:
    iterator m_begin;
    iterator m_end;

};

}

// Include the implementation
#include "piercedStorageRange.tpp"

#endif
