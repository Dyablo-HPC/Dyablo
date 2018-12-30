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

#include "bitpit_operators.hpp"

namespace bitpit {

namespace genericIO {

/*!
 * Writes a POD data type to file stream in ascii format using scientific format
 * @tparam  data_T  type of POD data
 * @param[in]   str     file stream
 * @param[in]   data    data to be written
 */
template<class data_T>
void flushASCII(std::fstream &str, const data_T &data)
{
    std::ios::fmtflags streamFlags(str.flags());

    str << std::setprecision(8) << std::scientific;
    str << data << " ";

    str.flags(streamFlags);
}

/*!
 * Writes a vector of POD data type to file stream in ascii format using scientific format on a single line
 * @tparam  data_T  type of POD data
 * @param[in]   str     file stream
 * @param[in]   data    data to be written
 */
template<class data_T>
void flushASCII(std::fstream &str, const std::vector<data_T> &data)
{
    flushASCII(str, data.size(), data);
}

/*!
 * Writes a vector of POD data type to file stream in ascii format using scientific format putting a fixed number of elements per line
 * @tparam  data_T  type of POD data
 * @param[in]   str     file stream
 * @param[in]   elements_per_line     number of entries per line
 * @param[in]   data    data to be written
 */
template<class data_T>
void flushASCII(std::fstream &str, int elements_per_line, const std::vector<data_T> &data)
{
    int i(0), j(0), k(0);
    int nr;

    int lines, this_line;

    bool next(true);

    nr = data.size();
    lines = (nr - 1) /elements_per_line + 1;

    std::ios::fmtflags streamFlags(str.flags());

    str << std::setprecision(8) << std::scientific;

    while (next) {
        this_line = std::min(elements_per_line, nr - k);

        for (j=0; j<this_line; j++) {
            flushASCII(str, data[k]);
            ++k;
        }

        i++;
        if (i<lines) {
            str << std::endl;
        } else {
            next = false;
        }
    }

    str.flags(streamFlags);
}

/*!
 * Writes a array of POD data type to file stream in ascii format using scientific format on one single line
 * @tparam  data_T  type of POD data
 * @tparam  d       size of array
 * @param[in]       str     file stream
 * @param[in]       data    data to be written
 */
template< class data_T, size_t d >
void flushASCII(std::fstream &str, const std::array<data_T,d> &data)
{
    flushASCII(str, d, data);
}

/*!
 * Writes a std::array of POD data type to file stream in ascii format using scientific format putting a fixed number of elements per line
 * @tparam          data_T  type of POD data
 * @tparam          d       size of array
 * @param[in]       str     file stream
 * @param[in]       elements_per_line     number of entries per line
 * @param[in]       data    data to be written
 */
template< class data_T, size_t d >
void flushASCII(std::fstream &str, int elements_per_line, const std::array<data_T,d> &data)
{
    int i(0), j(0), k(0);
    int nr;

    int lines, this_line;

    bool next(true);

    nr = d;
    lines = nr /elements_per_line;

    std::ios::fmtflags streamFlags(str.flags());

    str << std::setprecision(8) << std::scientific;

    while (next) {
        this_line = std::min(elements_per_line, nr - k);
        for (j=0; j<this_line; j++) {
            flushASCII(str, data[k]);
            ++k;
        }

        i++;
        if (i<lines) {
            str << std::endl;
        } else {
            next = false;
        }
    }

    str.flags(streamFlags);
}

/*!
 * Writes a C array of POD data type to file stream in ascii format using scientific format putting a fixed number of elements per line
 * @tparam          data_T  type of POD data
 * @param[in]       str     file stream
 * @param[in]       elements_per_line     number of entries per line
 * @param[in]       data    data to be written
 * @param[in]       nr      size of the C array
 */
template<class data_T>
void flushASCII(std::fstream &str, int elements_per_line, const data_T *data, int nr)
{
    int i(0), j(0), k(0);

    int lines, this_line;

    bool next(true);

    lines = nr /elements_per_line;

    std::ios::fmtflags streamFlags(str.flags());

    str << std::setprecision(8) << std::scientific;

    while (next) {
        this_line = std::min(elements_per_line, nr - k);
        for (j = 0; j < this_line; j++) {
            flushASCII(str, data[k]);
            ++k;
        }

        i++;
        if (i<lines) {
            str << std::endl;
        } else {
            next = false;
        }
    }

    str.flags(streamFlags);
}

/*!
 * Writes a bitpit::PiercedVector file stream in ascii format.
 * It assumes that flushASCIIY is available for data_T
 * @tparam          data_T  type data
 * @param[in]       str     file stream
 * @param[in]       elements_per_line     number of entries per line
 * @param[in]       data    data to be written
 * @param[in]       writeIndex if indices should be written too
 */
template<class data_T>
void flushASCII(std::fstream &str, int elements_per_line, const PiercedVector<data_T> &data, bool writeIndex)
{
    typename bitpit::PiercedVector<data_T>::const_iterator dataItr =data.begin();

    int i(0), j(0), k(0);
    int nr;

    int lines, this_line;

    bool next(true);

    nr = data.size();
    lines = (nr-1) /elements_per_line + 1;

    std::ios::fmtflags streamFlags(str.flags());

    str << std::setprecision(8) << std::scientific;

    while (next) {

        this_line = std::min(elements_per_line, nr - k);

        for (j=0; j<this_line; j++) {
            if (writeIndex) {
                flushASCII(str, dataItr.getId());
            }
            flushASCII(str, *dataItr);
            ++k;
            ++dataItr;
        }

        i++;
        if (i<lines) {
            str << std::endl;
        } else {
            next = false;
        }
    }

    str.flags(streamFlags);
}

/*!
 * Writes a POD data type to file stream in binary format.
 * The function makes uses of memory contigiuty.
 * @tparam          data_T  type of POD data
 * @param[in]       str     file stream
 * @param[in]       data    data to be written
 */
template<class data_T>
void flushBINARY(std::fstream &str, const data_T &data)
{
    int nbytes;
    nbytes = sizeof(data_T);

    str.write(reinterpret_cast<const char*>(&data), nbytes);
}

/*!
 * Writes a std::vector of POD data type to file stream in binary format.
 * The function makes uses of memory contigiuty.
 * @tparam          data_T  type of POD data
 * @param[in]       str     file stream
 * @param[in]       data    data to be written
 */
template<class data_T>
void flushBINARY(std::fstream &str, const std::vector<data_T> &data)
{
    int nbytes, nr;
    nr = data.size();
    nbytes = sizeof(data_T) * nr;

    str.write(reinterpret_cast<const char*>(&data[0]), nbytes);
}

/*!
 * Writes a std::vector<std::vector> of POD data type to file stream in binary format.
 * @tparam          data_T  type of POD data
 * @param[in]       str     file stream
 * @param[in]       data    data to be written
 */
template<class data_T>
void flushBINARY(std::fstream &str, const std::vector<std::vector<data_T>> &data)
{
    for (const auto &item : data) {
        flushBINARY(str, item);
    }
}

/*!
 * Writes a std::vector<std::array> of POD data type to file stream in binary format.
 * The function makes uses of memory contigiuty.
 * @tparam          data_T  type of POD data
 * @tparam          d       size of the array
 * @param[in]       str     file stream
 * @param[in]       data    data to be written
 */
template< class data_T, size_t d >
void flushBINARY(std::fstream &str, const std::vector<std::array<data_T,d> > &data)
{
    int nbytes, nr;
    nr = data.size();
    nbytes = sizeof(data_T) * nr * d;

    str.write(reinterpret_cast<const char*>(&data[0]), nbytes);
}

/*!
 * Writes a std::array of POD data type to file stream in binary format 
 * @tparam          data_T  type of POD data
 * @tparam          d       size of the array
 * @param[in]       str     file stream
 * @param[in]       data    data to be written
 */
template< class data_T, size_t d >
void flushBINARY(std::fstream &str, const std::array<data_T,d> &data)
{
    int nbytes;
    nbytes = sizeof(data_T)*d;

    str.write(reinterpret_cast<const char*>(&data[0]), nbytes);
}

/*!
 * Writes a C array of POD data type to file stream in binary format 
 * @tparam          data_T  type of POD data
 * @param[in]       str     file stream
 * @param[in]       data    data to be written
 * @param[in]       nr      size of the C array
 */
template<class data_T>
void flushBINARY(std::fstream &str, const data_T *data, int nr)
{
    int nbytes;
    nbytes = sizeof(data_T) *nr;

    str.write(reinterpret_cast<const char*>(data), nbytes);
}

/*!
 * Writes a bitpit::PiercedVector file stream in binary format.
 * It assumes that flushBINARY is available for data_T
 * @tparam          data_T  type data
 * @param[in]       str     file stream
 * @param[in]       data    data to be written
 * @param[in]       writeIndex if indices should be written
 */
template<class data_T>
void flushBINARY(std::fstream &str, const PiercedVector<data_T> &data, bool writeIndex)
{
    typename PiercedVector<data_T>::const_iterator dataItr, dataEnd = data.end();

    if (writeIndex) {
        for (dataItr = data.begin(); dataItr != dataEnd; ++dataItr) {
            flushBINARY(str, dataItr.getId());
            flushBINARY(str, *dataItr);
        }
    } else {
        for (dataItr = data.begin(); dataItr != dataEnd; ++dataItr) {
            flushBINARY(str, *dataItr);
        }
    }
}

/*!
 * Reads one line into templated data type.
 * Relies on operator ">>" of the templated data type.
 * In case the information on the line is not sufficient or exceeds the data type an error message is displayed
 * @tparam          data_T  type of POD data
 * @param[in]       str     file stream
 * @param[in]       data    data to be written
 */
template<class data_T>
void  lineStream(std::fstream &str, data_T &data)
{
    std::vector<data_T> temp;
    data_T         x_;
    std::string         line;
    int            expected, read(0);

    expected = 1;

    getline(str, line);
    bitpit::utils::string::trim(line);

    std::stringstream ss(line);

    while (ss.good()) {
        ss >> x_;
        temp.push_back(x_);
        read++;
    }

    if (read != expected) {
        log::cout() << " Not expected nr of element in line" << std::endl;
        log::cout() << " Expected number: "<< expected << std::endl;
        log::cout() << " Actual number: "<< read << std::endl;
    } else {
        data=temp[0];
    }
}

/*!
 * Reads one line into std::vector of templated data type.
 * Relies on operator ">>" of the templated data type.
 * In case data.size() == 0, data will be resized to match all information available on the line.
 * Otherwise the information on the line should fit exactly the size of data.
 * If not an error message is displayed log::cout()
 * @tparam          data_T  type of POD data
 * @param[in]       str     file stream
 * @param[in]       data    data to be written
 */
template<class data_T>
void  lineStream(std::fstream &str, std::vector<data_T> &data)
{
    std::string           line;
    int                   expected(data.size());

    getline(str, line);
    bitpit::utils::string::trim(line);

    std::stringstream ss(line);

    if (expected == 0) {
        data_T x_;
        while (ss.good()) {
            ss >> x_;
            data.push_back(x_);
        }
    } else {
        int read(0);
        while (ss.good() && read<expected) {
            ss >> data[read];
            read++;
        }

        if (expected != read) {
            log::cout() << " Not expected nr of element in line" << std::endl;
            log::cout() << " Expected number: "<< expected << std::endl;
            log::cout() << " Actual number: "<< read << std::endl;
        }
    }
}

/*!
 * Reads one line into std::array of templated data type and templated size
 * Relies on operator ">>" of the templated data type.
 * The information on the line should fit exactly the size of the array.
 * If not an error message is displayed log::cout()
 * @tparam          data_T  type of POD data
 * @tparam          d       size of std::array
 * @param[in]       str     file stream
 * @param[in]       data    data to be written
 */
template< class data_T, size_t d >
void  lineStream(std::fstream &str, std::array<data_T,d> &data)
{
    std::vector<data_T> temp;
    data_T              x_;
    std::string         line;
    int                 expected, read(0), i;

    expected = d;

    getline(str, line);
    bitpit::utils::string::trim(line);

    std::stringstream ss(line);

    while (ss.good()) {
        ss >> x_;
        temp.push_back(x_);
        read++;
    }

    if (expected == read) {
        for (i=0; i<read; i++) data[i] = temp[i];
    } else {
        log::cout() << " Not expected nr of element in line" << std::endl;
        log::cout() << " Expected number: "<< expected << std::endl;
        log::cout() << " Actual number: "<< read << std::endl;
    }
}

/*!
 * Reads one line into C array of templated data type and given size.
 * Relies on operator ">>" of the templated data type.
 * The information on the line should fit exactly the size .
 * If not an error message is displayed log::cout()
 * @tparam          data_T  type of POD data
 * @param[in]       str     file stream
 * @param[in]       data    data to be written
 * @param[in]       nr      number of elements to be read
 */
template<class data_T>
void  lineStream(std::fstream &str, data_T *data, int nr)
{
    std::vector<data_T> temp;
    data_T              x_;
    std::string         line;
    int                 expected, read(0), i;

    expected = nr;

    getline(str, line);
    bitpit::utils::string::trim(line);

    std::stringstream ss(line);

    while (ss.good()) {
        ss >> x_;
        temp.push_back(x_);
        read++;
    }

    if (expected == read) {
        for (i=0; i<read; i++) data[i] = temp[i];
    } else {
        log::cout() << " Not expected nr of element in line" << std::endl;
        log::cout() << " Expected number: "<< expected << std::endl;
        log::cout() << " Actual number: "<< read << std::endl;
    }
}

/*!
 * Reads a templated data type from file stream in ascii 
 * Relies on the ">>" operator.
 * @tparam  data_T  type of POD data
 * @param[in]   str     file stream
 * @param[in]   data    data to be read
 */
template<class data_T>
void absorbASCII(std::fstream &str, data_T &data)
{
    str >> data;
}

/*!
 * Reads a std::vector of data type from file stream in ascii.
 * The size of the vector defines the number of elements to be read.
 * If not enough elements are present in the file an error message is displayed on log::cout()
 * Relies on the function lineStream.
 * @tparam  data_T  type of POD data
 * @param[in]   str     file stream
 * @param[in]   data    data to be read
 */
template<class data_T>
void absorbASCII(std::fstream &str, std::vector<data_T> &data)
{
    std::vector<data_T>             temp;

    typename std::vector<data_T>::iterator   itrData, begData, endData;
    typename std::vector<data_T>::iterator   itrTemp, begTemp, endTemp;

    begData = data.begin();
    endData = data.end();

    itrData = begData;

    while (str.good() && itrData!=endData) {
        temp.clear();
        lineStream(str, temp);

        begTemp = temp.begin();
        endTemp = temp.end();

        for (itrTemp=begTemp; (itrTemp!=endTemp && itrData!=endData);) {
            *itrData = *itrTemp;

            ++itrTemp;
            ++itrData;
        }
    }

    if (itrData != endData) {
        log::cout() << "Not enough elements found to fill vector" << std::endl;
    }
}

/*!
 * Reads a std::vector of std::vector of templated POD data type from file stream in ascii format
 * @tparam data_T type of data
 * @param[in] str file stream
 * @param[in] data data to be read
 */
template<class data_T>
void absorbASCII(std::fstream &str, std::vector<std::vector<data_T>> &data)
{

    size_t dim1= data.size();
    if(dim1 == 0){
        return;
    }

    size_t row(0);

    typename std::vector<data_T>::iterator itrRow = data[row].begin();
    typename std::vector<data_T>::iterator endRow = data[row].end();

    std::vector<data_T> temp;
    typename std::vector<data_T>::iterator itrTemp;

    while(itrRow==endRow){
        ++row;

        if(row==dim1){
            log::cout() << "all internal vectors have 0 size in absorbASCII " << std::endl;
            return;
        }

        itrRow = data[row].begin();
        endRow = data[row].end();
    }

    while(str.good()) {

        temp.clear();
        lineStream(str, temp);

        for (itrTemp=temp.begin(); itrTemp!=temp.end(); ++itrTemp) {
            *itrRow = *itrTemp;

            ++itrRow;
            while(itrRow==endRow){
                ++row;

                if(row==dim1){
                    return;
                }

                itrRow = data[row].begin();
                endRow = data[row].end();
            }
        }
    }

    log::cout() << "Not enough elements found to fill vector" << std::endl;
}

/*!
 * Reads a std::vector of std::array of templated POD data type of templated size from file stream in asci format
 * @tparam data_T type of data
 * @tparam d  size of array
 * @param[in] str file stream
 * @param[in]   data    data to be read
 */
template< class data_T, size_t d >
void absorbASCII(std::fstream &str, std::vector<std::array<data_T,d> > &data)
{
    if(d == 0){
        return;
    }

    size_t nArrays = data.size();
    if(nArrays == 0){
        return;
    }

    absorbASCII(str, &data[0][0], nArrays * d);
}

/*!
 * Reads a std::array of data type from file stream in ascii.
 * The size of the array defines the number of elements to be read.
 * If not enough elements are present in the file an error message is displayed on log::cout()
 * Relies on the function lineStream.
 * @tparam      data_T  class stored in std::array
 * @tparam      d       size of std::array
 * @param[in]   str     file stream
 * @param[in]   data    data to be read
 */
template< class data_T, size_t d >
void absorbASCII(std::fstream &str, std::array<data_T,d> &data)
{
    std::vector<data_T>             temp;

    typename std::array<data_T,d>::iterator  itrData, begData, endData;
    typename std::vector<data_T>::iterator   itrTemp, begTemp, endTemp;

    begData = data.begin();
    endData = data.end();

    itrData = begData;

    while (str.good() && itrData!=endData) {
        temp.clear();
        lineStream(str, temp);

        begTemp = temp.begin();
        endTemp = temp.end();

        for (itrTemp = begTemp; (itrTemp!=endTemp && itrData!=endData);) {
            *itrData = *itrTemp;

            ++itrTemp;
            ++itrData;
        }
    }

    if (itrData != endData) {
        log::cout() << "Not enough elements found to fill array" << std::endl;
    }
}

/*!
 * Reads a C array of data type from file stream in ascii.
 * If not enough elements are present in the file an error message is displayed on log::cout()
 * Relies on the function lineStream.
 * @tparam      data_T  class stored in C array
 * @tparam      d       size of std::array
 * @param[in]   str     file stream
 * @param[in]   data    pointer to C array
 * @param[in]   nr      number of elements
 */
template<class data_T>
void absorbASCII(std::fstream &str, data_T *data, int nr)
{
    std::vector<data_T>                      temp;

    data_T                                   *itrData, *begData, *endData;
    typename std::vector<data_T>::iterator   itrTemp, begTemp, endTemp;

    begData = &data[0];
    endData = &data[nr];

    itrData = begData;

    while (str.good() && itrData!=endData) {
        temp.clear();
        lineStream(str, temp);

        begTemp = temp.begin();
        endTemp = temp.end();

        for (itrTemp=begTemp; (itrTemp!=endTemp && itrData!=endData);) {
            *itrData = *itrTemp;

            ++itrTemp;
            ++itrData;
        }
    }

    if (itrData != endData) {
        log::cout() << "Not enough elements found to fill array" << std::endl;
    }
}

/*!
 * Reads only the values of the elemnts of a bitpit::PiercedVector from file stream in ascii format.
 * @tparam  data_T  type of POD data
 * @param[in]   str     file stream
 * @param[in]   data    data to be read
 */
template<class data_T>
void absorbASCII(std::fstream &str, bitpit::PiercedVector<data_T> &data)
{
    bool    read(true);
    std::string line;

    typename bitpit::PiercedVector<data_T>::iterator  dataItr = data.begin(), dataEnd = data.end();

    while (str.good() && read) {
        getline(str, line);
        bitpit::utils::string::trim(line);

        std::stringstream ss(line);

        while (ss.good() && read) {
            ss >> *dataItr;
            ++dataItr;

            read = dataItr != dataEnd;
        }
    }
}

/*!
 * Reads a bitpit::PiercedVector from file stream in ascii format, both indices and values of its elements.
 * Relies on the function lineStream.
 * @tparam  data_T  type of POD data
 * @param[in]   str     file stream
 * @param[in]   data    data to be read
 * @param[in]   N number of elements to br read
 */
template<class data_T>
void absorbASCII(std::fstream &str, bitpit::PiercedVector<data_T> &data, long N)
{
    bool    read(true);
    long    n(0), index;
    data_T  value;
    std::string line;

    while (str.good() && read) {
        getline(str, line);
        bitpit::utils::string::trim(line);

        std::stringstream ss(line);

        while (ss.good() && read) {
            ss >> index;
            ss >> value;
            data.insert(index,value);
            ++n;
            read = n<N;
        }
    }
}

/*!
 * Reads a templated data type from file stream in binary
 * @tparam      data_T  type of data
 * @param[in]   str     file stream
 * @param[in]   data    data to be read
 */
template<class data_T>
void absorbBINARY(std::fstream &str, data_T &data)
{
    int nbytes;
    nbytes = sizeof(data_T);

    str.read(reinterpret_cast<char*>(&data), nbytes);
}

/*!
 * Reads a std::vector of templated POD data type from file stream in binary format
 * @tparam      data_T  type of data
 * @param[in]   str     file stream
 * @param[in]   data    data to be read
 */
template<class data_T>
void absorbBINARY(std::fstream &str, std::vector<data_T> &data)
{
    int nbytes, nr;
    nr = data.size();
    nbytes = sizeof(data_T) *nr;

    str.read(reinterpret_cast<char*>(&data[0]), nbytes);
}

/*!
 * Reads a std::vector of std::vector of templated POD data type from file stream in binary format
 * @tparam      data_T  type of data
 * @param[in]   str     file stream
 * @param[in]   data    data to be read
 */
template<class data_T>
void absorbBINARY(std::fstream &str, std::vector<std::vector<data_T>> &data)
{
    for (auto &item: data) {
        absorbBINARY(str, item);
    }
}

/*!
 * Reads a std::vector of std::array of templated POD data type of templated size from file stream in binary
 * @tparam      data_T  type of data
 * @tparam      d       size of array
 * @param[in]   str     file stream
 * @param[in]   data    data to be read
 */
template< class data_T, size_t d >
void absorbBINARY(std::fstream &str, std::vector<std::array<data_T,d> > &data)
{
    int  nbytes, nr;
    nr = data.size();
    nbytes = sizeof(data_T) *nr *d;

    str.read(reinterpret_cast<char*>(&data[0]), nbytes);
}

/*!
 * Reads a std::array of templated POD data type of templated size from file stream in binary format
 * @tparam      data_T  type of data
 * @tparam      d       size of array
 * @param[in]   str     file stream
 * @param[in]   data    data to be read
 */
template< class data_T, size_t d >
void absorbBINARY(std::fstream &str, std::array<data_T,d> &data)
{
    int nbytes;
    nbytes = sizeof(data_T) *d;

    str.read(reinterpret_cast<char*>(&data[0]), nbytes);
}

/*!
 * Reads a C array of templated data type from file stream in binary format
 * @tparam      data_T  type of data
 * @param[in]   str     file stream
 * @param[in]   data    data to be read
 * @param[in]   nr      number of elements to be read
 */
template<class data_T>
void absorbBINARY(std::fstream &str, data_T *data, int nr)
{
    int nbytes;
    nbytes = sizeof(data_T) *nr;

    str.read(reinterpret_cast<char*>(data), nbytes);
}

/*!
 * Reads a bitpit::PiercedVector file stream in binary format.
 * The PiercedVector should already contain the indices of its elements and the data of all elements will be read.
 * It assumes that absorbBINARY is available for data_T; this implies that data_T is of known size.
 * @tparam          data_T  type data
 * @param[in]       str     file stream
 * @param[in]       data    data to be written
 */
template<class data_T>
void absorbBINARY(std::fstream &str, PiercedVector<data_T> &data)
{
    typename bitpit::PiercedVector<data_T>::iterator dataItr, dataEnd = data.end();

    for (dataItr = data.begin(); dataItr != dataEnd; ++dataItr) {
        absorbBINARY(str, *dataItr);
    }
}

/*!
 * Reads a bitpit::PiercedVector file stream in binary format, both the indices and data of its elements.
 * The user must specify how many elements should be read from the stream.
 * It assumes that absorbBINARY is available for data_T; this implies that data_T is of sized size
 * @tparam          data_T  type data
 * @param[in]       str     file stream
 * @param[in]       data    data to be written
 * @param[in]       nElements the number of elements that should be read
 */
template<class data_T>
void absorbBINARY(std::fstream &str, PiercedVector<data_T> &data, long nElements)
{
    long n, index;
    data_T value;

    data.reserve(nElements);

    for (n=0; n<nElements; ++n) {
        absorbBINARY(str,index);
        absorbBINARY(str,value);

        data.insert(index,value);
    }
}

}

}
