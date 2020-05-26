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
 * UserDataLB.tpp
 *
 *  Created on: 27/mar/2014
 *      Author: Marco Cisternino
 */

/* \cond
 */
template<class D>
inline size_t WeightDataLB<D>::fixedSize() const {
	return 0;
}

template<class D>
inline size_t WeightDataLB<D>::size(const uint32_t e) const {
	BITPIT_UNUSED(e);
	return sizeof(double);
}

template<class D>
inline void WeightDataLB<D>::move(const uint32_t from, const uint32_t to) {
	data[to] = data[from];
}

template<class D>
template<class Buffer>
inline void WeightDataLB<D>::gather(Buffer& buff, const uint32_t e) {
	buff << data[e];
}

template<class D>
template<class Buffer>
inline void WeightDataLB<D>::scatter(Buffer& buff, const uint32_t e) {
	buff >> data[e];
}

template<class D>
inline void WeightDataLB<D>::assign(uint32_t stride, uint32_t length) {
	Data dataCopy = data;
	typename Data::iterator first = dataCopy.begin() + stride;
	typename Data::iterator last = first + length;
	data.assign(first,last);
#if defined(__INTEL_COMPILER)
#else
	data.shrink_to_fit();
#endif
	first = dataCopy.end();
	last = dataCopy.end();
};

template<class D>
inline void WeightDataLB<D>::resize(uint32_t newSize) {
	data.resize(newSize);
}

template<class D>
inline void WeightDataLB<D>::resizeGhost(uint32_t newSize) {
	ghostdata.resize(newSize);
}

template<class D>
inline void WeightDataLB<D>::shrink() {
#if defined(__INTEL_COMPILER)
#else
	data.shrink_to_fit();
#endif
}

template<class D>
inline WeightDataLB<D>::WeightDataLB(Data& data_, Data& ghostdata_) : data(data_), ghostdata(ghostdata_){}

template<class D>
inline WeightDataLB<D>::~WeightDataLB() {}
/* \endcond
 */
