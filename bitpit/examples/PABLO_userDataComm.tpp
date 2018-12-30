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

/*
 * UserDataComm.tpp
 *
 *  Created on: 18/mar/2014
 *      Author: Marco Cisternino
 */

// #include "bitpit_common.hpp"

/*!
 * \cond
 */
template<class Data>
UserDataComm<Data>::UserDataComm(Data & data_, Data & ghostData_) : data(data_), ghostData(ghostData_){};

template<class Data>
UserDataComm<Data>::~UserDataComm() {};

template<class Data>
inline size_t UserDataComm<Data>::fixedSize() const {
	return 0;
};

template<class Data>
inline size_t UserDataComm<Data>::size(const uint32_t e) const {
	BITPIT_UNUSED(e);
	return sizeof(double)+sizeof(float);
};

template<class Data>
template<class Buffer>
inline void UserDataComm<Data>::gather(Buffer& buff, const uint32_t e) {
	buff << data.doubleData[e];
	buff << data.floatData[e];
};

template<class Data>
template<class Buffer>
inline void UserDataComm<Data>::scatter(Buffer& buff,	const uint32_t e) {
	buff >> ghostData.doubleData[e];
	buff >> ghostData.floatData[e];
};

/*!
 * \endcond
 */
