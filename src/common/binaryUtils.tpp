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

#ifndef __BITPIT_COMMON_BINARY_UTILS_TPP__
#define __BITPIT_COMMON_BINARY_UTILS_TPP__

namespace bitpit {

namespace utils {

namespace binary {

/*!
    \ingroup common_binary

    Write the given container to the specified stream in binary format.

    \param stream is the stream to write to
    \param container is the container to write
*/
template<typename T, typename std::enable_if<std::is_pod<T>::value>::type*>
void write(std::ostream &stream, const std::vector<T> &container)
{
    size_t size = container.size();
    write(stream, size);

    write(stream, container.data(), size * sizeof(T));
}

/*!
    \ingroup common_binary

    Write the given container to the specified stream in binary format.

    \param stream is the stream to write to
    \param container is the container to write
*/
template<typename T, std::size_t dim, typename std::enable_if<std::is_pod<T>::value>::type*>
void write(std::ostream &stream, const std::array<T, dim> &container)
{
    write(stream, container.data(), dim * sizeof(T));
}

/*!
    \ingroup common_binary

    Write the given container to the specified stream in binary format.

    \param stream is the stream to write to
    \param container is the container to write
*/
template<typename T, std::size_t dim, typename std::enable_if<!std::is_pod<T>::value>::type*>
void write(std::ostream &stream, const std::array<T, dim> &container)
{
    for (std::size_t i = 0; i < dim; ++i) {
        write(stream, container[i]);
    }
}

/*!
    \ingroup common_binary

    Write the given container to the specified stream in binary format.

    \param stream is the stream to write to
*/
template<typename T, typename std::enable_if<utils::is_iterable<T>::value>::type*>
void write(std::ostream &stream, const T &value)
{
    write(stream, value.size());

    for (const auto &item : value) {
        write(stream, item);
    }
}

/*!
    \ingroup common_binary

    Write the given data to the specified stream in binary format.

    \param stream is the stream to write to
    \param value is the data to write
*/
template<typename T, typename std::enable_if<std::is_pod<T>::value>::type*>
void write(std::ostream &stream, const T &value)
{
    write(stream, &value, sizeof(T));
}

/*!
    \ingroup common_binary

    Write the given data to the specified stream in binary format.

    \param stream is the stream to write to
    \param value is the data to write
    \param size is the size, expressed in bytes, of the data to write
*/
template<typename T>
void write(std::ostream &stream, const T &value, size_t size)
{
    stream.write(&value, size);
}

/*!
    \ingroup common_binary

    Write the given data to the specified stream in binary format.

    \param stream is the stream to write to
    \param value is the data to write
    \param size is the size, expressed in bytes, of the data to write
*/
template<typename T>
void write(std::ostream &stream, const T *value, size_t size)
{
    stream.write(reinterpret_cast<const char*>(value), size);
}

/*!
    \ingroup common_binary

    Read the given container to the specified stream in binary format.

    \param stream is the stream to write to
    \param container is the container to write
*/
template<typename T, typename std::enable_if<std::is_pod<T>::value>::type*>
void read(std::istream &stream, std::vector<T> &container)
{
    size_t size;
    read(stream, size);

    container.resize(size);
    read(stream, container.data(), size * sizeof(T));
}

/*!
    \ingroup common_binary

    Read the given container to the specified stream in binary format.

    \param stream is the stream to write to
    \param container is the container to write
*/
template<typename T, std::size_t dim, typename std::enable_if<std::is_pod<T>::value>::type*>
void read(std::istream &stream, std::array<T, dim> &container)
{
    read(stream, container.data(), dim * sizeof(T));
}

/*!
    \ingroup common_binary

    Read the given container to the specified stream in binary format.

    \param stream is the stream to write to
    \param container is the container to write
*/
template<typename T, std::size_t dim, typename std::enable_if<!std::is_pod<T>::value>::type*>
void read(std::istream &stream, std::array<T, dim> &container)
{
    for (std::size_t i = 0; i < dim; ++i) {
        read(stream, container[i]);
    }
}

/*!
    \ingroup common_binary

    Read the given container to the specified stream in binary format.

    \param stream is the stream to write to
*/
template<typename T, typename std::enable_if<utils::is_iterable<T>::value>::type*>
void read(std::istream &stream, T &value)
{
    size_t size;
    read(stream, size);

    value.resize(size);
    for (auto &item : value) {
        read(stream, item);
    }
}

/*!
    \ingroup common_binary

    Read the given data to the specified stream in binary format.

    \param stream is the stream to read from
    \param[out] value on output it will contain the read value
*/
template<typename T, typename std::enable_if<std::is_pod<T>::value>::type*>
void read(std::istream &stream, T &value)
{
    read(stream, &value, sizeof(T));
}

/*!
    \ingroup common_binary

    Read the given data to the specified stream in binary format.

    \param stream is the stream to read from
    \param[out] value on output it will contain the read value
    \param size is the size, expressed in bytes, of the data to read
*/
template<typename T>
void read(std::istream &stream, T &value, size_t size)
{
    read(stream, &value, size);
}

/*!
    \ingroup common_binary

    Read the given data to the specified stream in binary format.

    \param stream is the stream to read from
    \param[out] value on output it will contain the read value
    \param size is the size, expressed in bytes, of the data to read
*/
template<typename T>
void read(std::istream &stream, T *value, size_t size)
{
    stream.read(reinterpret_cast<char*>(value), size);
}

}

}

}

#endif
