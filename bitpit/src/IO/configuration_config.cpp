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

#include <stdexcept>

#include "configuration_config.hpp"

namespace bitpit {

/*!
    \class Config
    \ingroup Configuration

    \brief Configuration storage

    This class implements a configuration storage.
*/

/*!
    Create a new configuration.

    \param multiSections if set to true the configuration will allow multiple
    sections with the same name
*/
Config::Config(bool multiSections)
    : m_multiSections(multiSections),
      m_options(std::unique_ptr<Options>(new Options())),
      m_sections(std::unique_ptr<Sections>(new Sections()))
{
}

/*!
    Copy constructor.

    \param other is the object to be copied
*/
Config::Config(const Config &other)
    : m_multiSections(other.m_multiSections),
      m_options(std::unique_ptr<Options>(new Options(*(other.m_options)))),
      m_sections(std::unique_ptr<Sections>(new Sections()))
{
    for (const auto &entry : *(other.m_sections)) {
        std::unique_ptr<Config> config = std::unique_ptr<Config>(new Config(*(entry.second)));
        m_sections->insert(std::make_pair(entry.first, std::move(config)));
    }
}

/*!
    Copy assigment operator.

    \param other is the object to be copied
*/
Config & Config::operator=(Config other)
{
    this->swap(other);

    return *this;
}

/*!
    Destructor.
*/
Config::~Config()
{
}

/*!
    Swap operator.

    \param other is another config whose content is swapped with that of
    this config
*/
void Config::swap(Config &other)
{
    std::swap(m_multiSections, other.m_multiSections);
    std::swap(m_options, other.m_options);
    std::swap(m_sections, other.m_sections);
}


/*!
    Returns true if multi-sections are enabled.

    \result Returns true if multi-sections are enabled.
*/
bool Config::isMultiSectionsEnabled() const
{
    return m_multiSections;
}

/*!
    Count the number of options.

    \result The number of options stored.
*/
int Config::getOptionCount() const
{
    return m_options->size();
}

/*!
    Gets a reference to the stored options.

    \result A reference to the stored options.
*/
Config::Options & Config::getOptions()
{
    return const_cast<Options &>(static_cast<const Config &>(*this).getOptions());
}

/*!
    Gets a constant reference to the stored options.

    \result A constant reference to the stored options.
*/
const Config::Options & Config::getOptions() const
{
    return *m_options;
}

/*!
    Checks if the specified option exists.

    \param key is the name of the option
    \result True is the option exists, false otherwise.
*/
bool Config::hasOption(const std::string &key) const
{
    return (m_options->count(key) > 0);
}

/*!
    Gets the specified option.

    If the option does not exists an exception is thrown.

    \param key is the name of the option
    \result The specified option.
*/
std::string Config::get(const std::string &key) const
{
    return m_options->at(key);
}

/*!
    Gets the specified option.

    If the option does not exists the fallback value is returned.

    \param key is the name of the option
    \param fallback is the value that will be returned if the specified
    options does not exist
    \result The specified option or the fallback value if the specified
    options does not exist.
*/
std::string Config::get(const std::string &key, const std::string &fallback) const
{
    if (hasOption(key)) {
        return get(key);
    } else {
        return fallback;
    }
}

/*!
    Set the given option to the specified value

    \param key is the name of the option
    \param value is the value of the option
*/
void Config::set(const std::string &key, const std::string &value)
{
    (*m_options)[key] = value;
}

/*!
    Remove the specified option.

    \param key is the name of the option
    \result Returns true if the option existed, otherwise it returns false.
*/
bool Config::removeOption(const std::string &key)
{
    return (m_options->erase(key) != 0);
}

/*!
    Count the number of sections.

    \result The number of sections stored.
*/
int Config::getSectionCount() const
{
    return m_sections->size();
}

/*!
    Count the number of sections with the specified name.

    \param key is the name of the section
    \result The number of sections with the specified name.
*/
int Config::getSectionCount(const std::string &key) const
{
    return m_sections->count(key);
}

/*!
    Gets a reference to the stored sections.

    \result A reference to the stored sections.
*/
Config::Sections & Config::getSections()
{
    return const_cast<Sections &>(static_cast<const Config &>(*this).getSections());
}

/*!
    Gets a constant reference to the stored options.

    \result A constant reference to the stored options.
*/
const Config::Sections & Config::getSections() const
{
    return *m_sections;
}

/*!
    Gets a list of pointers to the sections with the specified name.

    \param key is the name of the section
    \result A list of pointers to the sections with the specified name.
*/
Config::MultiSection Config::getSections(const std::string &key)
{
    MultiSection sections;
    sections.reserve(getSectionCount(key));

    auto range = m_sections->equal_range(key);
    for (auto itr = range.first; itr != range.second; ++itr) {
        sections.push_back(itr->second.get());
    }

    return sections;
}

/*!
    Gets a list of constant pointers to the sections with the specified name.

    \param key is the name of the section
    \result A list of constant pointers to the sections with the specified name.
*/
Config::ConstMultiSection Config::getSections(const std::string &key) const
{
    ConstMultiSection sections;
    sections.reserve(getSectionCount(key));

    auto range = m_sections->equal_range(key);
    for (auto itr = range.first; itr != range.second; ++itr) {
        sections.push_back(const_cast<const Section *>(itr->second.get()));
    }

    return sections;
}

/*!
    Checks if the specified section exists.

    \param key is the name of the section
    \result True is the section exists, false otherwise.
*/
bool Config::hasSection(const std::string &key) const
{
    return (getSectionCount(key) > 0);
}

/*!
    Gets a reference to the specified section.

    If the section does not exists an exception is thrown.

    \param key is the name of the section
    \result A reference to the specified section.
*/
Config::Section & Config::getSection(const std::string &key)
{
    return const_cast<Section &>(static_cast<const Config &>(*this).getSection(key));
}

/*!
    Gets a constant reference to the specified section.

    If the section does not exists an exception is thrown.

    \param key is the name of the section
    \result A constant reference to the specified section.
*/
const Config::Section & Config::getSection(const std::string &key) const
{
    auto sectionItr = m_sections->find(key);
    if (sectionItr == m_sections->end()) {
        throw std::runtime_error("The section named \"" + key + "\" does not esists");
    }

    return *(sectionItr->second);
}

/*!
    Add a section with the specified name.

    If a section with the given name already exists, an exception is raised.

    \param key is the name of the section
    \return A reference to the newly added section.
*/
Config::Section & Config::addSection(const std::string &key)
{
    if (!m_multiSections && hasSection(key)) {
        throw std::runtime_error("A section named \"" + key + "\" already esists");
    }

    std::unique_ptr<Section> section = std::unique_ptr<Section>(new Section(m_multiSections));
    auto sectionItr = m_sections->insert(std::make_pair(key, std::move(section)));

    return *(sectionItr->second);
}

/*!
    Remove the specified section.

    \param key is the name of the section
    \result Returns true if the section existed, otherwise it returns false.
*/
bool Config::removeSection(const std::string &key)
{
    return (m_sections->erase(key) != 0);
}

/*!
    Clear the configuration.
*/
void Config::clear()
{
    m_options->clear();
    m_sections->clear();
}

/*!
    Get a constant reference of the specified section.

    If the section does not exists an exception is thrown.

    \param key is the name of the section
    \result A constant reference to the specified section.
*/
const Config::Section & Config::operator[](const std::string &key) const
{
    return getSection(key);
}

/*!
    Get a reference of the specified section.

    If the section does not exists an exception is thrown.

    \param key is the name of the section
    \result A reference to the specified section.
*/
Config::Section & Config::operator[](const std::string &key)
{
    return getSection(key);
}

/*!
    Write the specified configuration to screen.

    \param out is the stream where the configuration will be written
    \param indentLevel is the indentation level
*/
void Config::dump(std::ostream &out, int indentLevel) const
{
    const int INDENT_SIZE = 2;

    std::string padding(INDENT_SIZE, ' ');
    std::string indent(INDENT_SIZE * indentLevel, ' ');

    out << std::endl;
    out << indent << "Options..." << std::endl;
    if (getOptionCount() > 0) {
        for (auto &entry : getOptions()) {
            out << indent << padding << entry.first << " = " << entry.second << std::endl;
        }
    } else {
        out << indent << padding << "No options." << std::endl;
    }

    ++indentLevel;
    for (auto &entry : getSections()) {
        out << std::endl;
        out << indent << padding << "::: Section " << entry.first << " :::" << std::endl;
        entry.second->dump(out, indentLevel);
    }
}

}
