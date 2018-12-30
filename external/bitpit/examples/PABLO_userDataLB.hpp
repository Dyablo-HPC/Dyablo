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
 * User_data_LB.hpp
 *
 *  Created on: 27/mar/2014
 *      Author: Marco Cisternino
 */

#ifndef __BITPIT_USER_DATA_LB_HPP__
#define __BITPIT_USER_DATA_LB_HPP__
/*!
\cond HIDDEN_SYMBOLS
*/
#include "DataCommInterface.hpp"
template <class D>
class UserDataLB : public bitpit::DataLBInterface<UserDataLB<D> >{
public:

	typedef D Data;

	Data& data;
	Data& ghostdata;

	size_t fixedSize() const;
	size_t size(const uint32_t e) const;
	void move(const uint32_t from, const uint32_t to);

	template<class Buffer>
	void gather(Buffer & buff, const uint32_t e);

	template<class Buffer>
	void scatter(Buffer & buff, const uint32_t e);

	void assign(uint32_t stride, uint32_t length);
	void resize(uint32_t newSize);
	void resizeGhost(uint32_t newSize);
	void shrink();

	UserDataLB(Data& data_, Data& ghostdata_);
	~UserDataLB();
};

#include "PABLO_userDataLB.tpp"
/*
  \endcond
 */
#endif /* __BITPIT_USER_DATA_LB_HPP__ */
