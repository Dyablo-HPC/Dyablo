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

#ifndef __BITPIT_POD_HPP__
#define __BITPIT_POD_HPP__

#if BITPIT_ENABLE_MPI
#    include <mpi.h>
#endif
#include <string>
#include <vector>
#include <unordered_map>

#include "mesh_mapper.hpp"
#include "pod_kernel.hpp"
#include "pod_voloctree.hpp"

namespace bitpit {

class PODKernel;

class POD : public VTKBaseStreamer {

public:

    /*!
     *\enum MemoryMode
     *\brief Memory Mode of the POD object. It defines the use of the memory resources.
     */
    enum class MemoryMode {
        MEMORY_NORMAL /**<Normal use of the memory. The POD modes are stored in memory till the POD object is destroyed.*/,
        MEMORY_LIGHT /**<Light use of memory. The POD modes are not stored but they are read from file when needed.*/
    };

    /*!
     *\enum RunMode
     *\brief Run Mode of the POD object. It defines if the POD basis has to be computed or restored.
     */
    enum class RunMode {
        RESTORE /**<Restore the POD basis from dumped file.*/,
        COMPUTE /**<Compute the POD basis.*/
    };

    /*!
     *\enum WriteMode
     *\brief Output Write Mode of the POD object. It defines the amount of information written by the POD object.
     */
    enum class WriteMode {
        DUMP /**<Write (dump) only the files needed to restore the POD instance.*/,
        DEBUG /**<Write files to debug the run (POD dumping files, POD basis in .vtu format, database fields in .vtu format, reconstrucetd fields in .vtu format...) .*/,
        NONE /**<Write none.*/
    };

    /*!
     *\enum ReconstructionMode
     *\brief Mode of Reconstruction of fields by using the POD basis.
     */
    enum class ReconstructionMode {
        PROJECTION /**<Orthogonal projection of a field over the POD basis. Note: if the POD modes are not orthogonal on the active domain this is an approximation.*/,
        MINIMIZATION /**<Non-orthogonal projection of a field over the POD basis.Note: if the POD modes are orthogonal on the active domain the result obtained by using PROJECTION is the same.*/
    };

    /*!
     *\enum ErrorMode
     *\brief Mode of Error evaluation of a reconstructed fields by the POD basis.
     */
    enum class ErrorMode {
        COMBINED /**<Maximum of reconstruction errors.*/,
        SINGLE /**<Reconstruction errors one at a time.*/,
        NONE /**<No error evaluation.*/
    }; 

    /*!
     *\enum MeshType
     *\brief Type of the Mesh used to compute the POD basis.
     */
    enum class MeshType {
        UNDEFINED /**<Undefined mesh type. Note: not allowed to run POD computing.*/,
        VOLOCTREE /**<VolOctree mesh type.*/
    };

public:
# if BITPIT_ENABLE_MPI
    POD(MPI_Comm comm = MPI_COMM_WORLD);
# else
    POD();
# endif

    ~POD();

    /**
     * Default copy constructor.
     * \param[in] other Input POD object
     */
    POD(POD&& other) = default;

    void clear();

    void setDirectory(const std::string &directory);
    const std::string & getDirectory();
    void setName(const std::string &name);
    const std::string & getName();
    void addSnapshot(const std::string &directory, const std::string &name);
    void addSnapshot(const pod::SnapshotFile &file);
    void setSnapshots(const std::vector<pod::SnapshotFile> &database);
    void removeLeave1outSnapshot(const std::string &directory, const std::string &name);
    void removeLeave1outSnapshot(const pod::SnapshotFile &file);    
    void unsetLeave1outSnapshots();    
    void addReconstructionSnapshot(const std::string &directory, const std::string &name);
    void addReconstructionSnapshot(const pod::SnapshotFile &file);
    void setModeCount(std::size_t nmodes);
    std::size_t getModeCount();
    void setEnergyLevel(double energy);
    double getEnergyLevel();
    void setErrorThreshold(double threshold);
    double getErrorThreshold(); 
    void setTargetErrorFields(std::vector<std::string> &namesf, std::vector<std::array<std::string,3>> &namevf);

    void setMeshType(MeshType type);
    void setMesh(const std::string &directory, const std::string &name);
    void setMesh(const pod::SnapshotFile &file);
    void setMesh(VolumeKernel* mesh);
    MeshType getMeshType();
    void setStaticMesh(bool flag);
    void setUseMean(bool flag);    

    void setMemoryMode(MemoryMode mode);
    MemoryMode getMemoryMode();
    void setRunMode(RunMode mode);
    RunMode getRunMode();
    void setWriteMode(WriteMode mode);
    WriteMode getWriteMode();
    void setReconstructionMode(ReconstructionMode mode);
    ReconstructionMode getReconstructionMode();
    void setErrorMode(ErrorMode mode);
    ErrorMode getErrorMode();       
    void setExpert(bool mode = true);

    void setSensorMask(const PiercedStorage<bool> & mask, VolumeKernel * mesh = nullptr);

    std::size_t getSnapshotCount();
    std::vector<std::string> getScalarNames();
    std::vector<std::array<std::string,3>> getVectorNames();
    std::vector<std::string> getFieldsNames();

    const VolumeKernel* getMesh();
    const pod::PODMode & getMean();
    const std::vector<pod::PODMode> & getModes();
    std::vector<std::vector<double> > getReconstructionCoeffs();
    const std::unordered_set<long int> & getListActiveIDs();
    std::size_t getListIDInternalCount();
    std::unique_ptr<PODKernel> & getKernel();

    void run();
    void dump();
    void restore();
    void leave1out();    

    void evalMeanMesh();
    void fillListActiveIDs(const PiercedStorage<bool> &bfield);
    void evalCorrelation();
    void evalModes();
    void evalEigen();
    void evalReconstruction();
    void evalErrorBoundingBox();
    void computeMapper(VolumeKernel * mesh);
    void prepareMapper(const std::vector<adaption::Info> & info);
    void updateMapper(const std::vector<adaption::Info> & info);

    void reconstructFields(pod::PODField &field, pod::PODField &recon);
    void dumpField(const std::string &name, const pod::PODField &field) const;

    void reconstructFields(PiercedStorage<double> &fields, VolumeKernel *mesh,
            std::map<std::string, std::size_t> targetFields,
            const std::unordered_set<long> *targetCells);

private:
    std::unique_ptr<PODKernel>              m_podkernel;                /**< POD computational kernel */
    MeshType                                m_meshType;                 /**< Type of POD mesh*/
    bool                                    m_staticMesh;               /**< If true the mesh is unique and the same for each snapshot and for POD modes [it is read one time together with the first snapshot].*/
    bool                                    m_useMean;                  /**< If true the POD is computed by subtracting the mean fields from the snapshots.*/  
    std::string                             m_directory;                /**< Input/output directory.*/
    std::string                             m_name;                     /**< POD session name.*/
    std::vector<pod::SnapshotFile>          m_database;                 /**< Vector of snapshots (directory and file name structure) */
    std::vector<pod::SnapshotFile>          m_reconstructionDatabase;   /**< Vector of snapshots to be reconstructed (directory and file name structure) */
    std::vector<pod::SnapshotFile>          m_leave1outOffDatabase;     /**< Vector of snapshots (directory and file name structure) not used in the leave-1-out method*/    
    std::size_t                             m_nSnapshots;               /**< Number of snapshots*/
    std::size_t                             m_nReconstructionSnapshots; /**< Number of snapshots to be reconstructed*/
    std::size_t                             m_nScalarFields;            /**< Number of scalar fields (note. first fields in dumped file)*/
    std::size_t                             m_nVectorFields;            /**< Number of vector fields (note. last fields in dumped file)*/
    std::size_t                             m_nFields;                  /**< Number of total fields*/
    std::vector<std::string>                m_nameScalarFields;         /**< Names of scalar fields. */
    std::vector<std::array<std::string,3>>  m_nameVectorFields;         /**< Names of vector fields. */
    std::map<std::string, std::size_t>      m_nameTargetErrorFields;    /**< Map of target fields used in error bounding box evaluation. */    
    bool                                    m_toUpdate;                 /**< If true the pod structures need to be updated.*/
    PiercedStorage<bool>                    m_filter;                   /**< Filter field (!=0 fluid cell, ==0 solid cell) used to compute POD modes (no POD on solid cells).*/
    PiercedStorage<bool>                    m_sensorMask;               /**< Sensor mask field (!=0 solve cell, ==0 no-solve cell) used to project (orthogonally and non-orthogonally) on POD modes.*/
    pod::PODMode                            m_mean;                     /**< Mean field of the snapshots database.*/
    pod::PODField                           m_errorMap;                 /**< Error field.*/  
    std::vector<pod::PODMode>               m_modes;                    /**< POD Modes*/
    std::size_t                             m_nModes;                   /**< Number of retained POD modes*/
    double                                  m_energyLevel;              /**< Level of percentage energy of the retained POD modes*/
    double                                  m_errorThreshold;           /**< Minimum error threshold for bounding box computation*/

    std::vector<std::vector<double>>               m_correlationMatrices;   /**< Correlation matrices (internal use)*/
    std::vector<std::vector<double>>               m_minimizationMatrices;  /**< Least-squares minimization matrices (internal use)*/
    std::vector<std::vector<double>>               m_lambda;                /**< Eigenvalue of correlation matrix. */
    std::vector<std::vector<std::vector<double>>>  m_podCoeffs;             /**< Eigenvectors of correlation matrix (i.e. pod coefficients of database snapshots).*/
    std::vector<std::vector<double>>               m_reconstructionCoeffs;  /**< Pod coefficients of last reconstructed snapshot.*/

    std::unordered_set<long int>                   m_listActiveIDs;           /**<List of ID of active cells [to be updated when filter/mask change]. */
    std::vector<std::size_t>                       m_listActiveIDsLeave1out;  /**<List of the active snapshots used in the leave-1-out method*/  
    std::size_t                                    m_sizeInternal;            /**<Number of internal cells in the list of ID of active cells [the internal cells are placed first in the list of active IDs].*/

#if BITPIT_ENABLE_MPI
    MPI_Comm            m_communicator; /**< MPI communicator */
#endif
    int                 m_rank;         /**< Local rank of process. */
    int                 m_nProcs;       /**< Number of processes. */

    //pod options
    MemoryMode          m_memoryMode;           /**<Memory mode: MEMORY_NORMAL - pod modes always in memory, MEMORY_LIGHT - pod modes read from file. */
    RunMode             m_runMode;              /**<Restore or compute pod modes, mean field and pod mesh. */
    WriteMode           m_writeMode;            /**<Write mode: dump write pod info, modes, mean field and pod mesh on dump files only, DEBUG write even vtu files and NONE to dump/write nothing. [Default = DUMP] */
    ReconstructionMode  m_reconstructionMode;   /**<Evaluate reconstruction by PROJECTION or by MINIMIZATION. [Default = MINIMIZATION] */
    ErrorMode           m_errorMode;            /**<Error mode: COMBINED - evaluate the of maximum reconstruction errors, SINGLE - evaluate reconstruction error, NONE - do nothing. [Default = NONE, Default in leave-1-out = COMBINED] */
    bool                m_expert;               /**<Expert mode. Main features: 1 - In expert mode (true) the mapper between POD mesh and input mesh
                                                    during reconstruction of a field has to be updated manually,
                                                    otherwise is recomputed at each call. */

    std::vector<std::size_t>    _m_nr;  /**<Temporary number of modes to track the energy level of retained number of modes for different fields.*/

    const static int    ARCHIVE_VERSION = 0;

    const double    m_tol = 1.0e-12;  /**<Tolerance for energy check.*/

    void _evalMeanMesh();
    void checkModeCount(double *alambda, std::size_t ifield);
    void _evalModes();
    void initCorrelation();
    void evalCorrelationTerm(int i, pod::PODField &snapi, int j, pod::PODField &snapj);
    void evalReconstructionCoeffs(pod::PODField &snapi);
    void _evalReconstructionCoeffs(pod::PODField &snapi);
    void buildFields(pod::PODField &recon);
    void initErrorMaps();
    void buildErrorMaps(pod::PODField &snap, pod::PODField &recon);    
    void evalMinimizationMatrices();
    void initMinimization();
    void solveMinimization(std::vector<std::vector<double>> &rhs);

    void dumpMode(std::size_t ir);

    void readSnapshot(pod::SnapshotFile snap, pod::PODField &fieldr);
    void readMode(std::size_t ir);

    double getCellVolume(long id);
    double getRawCellVolume(long rawIndex);

    void diff(pod::PODField * _a, const pod::PODMode &b);
    void sum(pod::PODField * _a, const pod::PODMode &b);
    std::vector<double> fieldsl2norm(pod::PODField &snap);
    std::vector<double> fieldsMax(pod::PODField &snap);    

#if BITPIT_ENABLE_MPI
    void initializeCommunicator(MPI_Comm communicator);
    MPI_Comm getCommunicator() const;
    bool isCommunicatorSet() const;
    void freeCommunicator();
#endif

    void evalReconstructionCoeffs(PiercedStorage<double> &fields,
            const std::vector<std::size_t> &scalarIds, const std::vector<std::size_t> &podscalarIds,
            const std::vector<std::array<std::size_t, 3>> &vectorIds, const std::vector<std::size_t> &podvectorIds);
    void _evalReconstructionCoeffs(PiercedStorage<double> &fields,
            const std::vector<std::size_t> &scalarIds, const std::vector<std::size_t> &podscalarIds,
            const std::vector<std::array<std::size_t, 3>> &vectorIds, const std::vector<std::size_t> &podvectorIds);
    void buildFields(PiercedStorage<double> &fields,
            const std::vector<std::size_t> &scalarIds, const std::vector<std::size_t> &podscalarIds,
            const std::vector<std::array<std::size_t, 3>> &vectorIds, const std::vector<std::size_t> &podvectorIds,
            const std::unordered_set<long> *targetCells = nullptr);
    void _buildFields(PiercedStorage<double> &fields,
            const std::vector<std::size_t> &scalarIds, const std::vector<std::size_t> &podscalarIds,
            const std::vector<std::array<std::size_t, 3>> &vectorIds, const std::vector<std::size_t> &podvectorIds,
            const std::unordered_set<long> *targetCells = nullptr);

    void _computeMapper(VolumeKernel * mesh);
    void _updateMapper(const std::vector<adaption::Info> & info);

    void diff(PiercedStorage<double> &fields, const pod::PODMode &mode,
            const std::vector<std::size_t> &scalarIds, const std::vector<std::size_t> &podscalarIds,
            const std::vector<std::array<std::size_t, 3>> &vectorIds, const std::vector<std::size_t> &podvectorIds,
            const std::unordered_set<long> *targetCells = nullptr);
    void sum(PiercedStorage<double> &fields, const pod::PODMode &mode,
            const std::vector<std::size_t> &scalarIds, const std::vector<std::size_t> &podscalarIds,
            const std::vector<std::array<std::size_t, 3>> &vectorIds, const std::vector<std::size_t> &podvectorIds,
            const std::unordered_set<long> *targetCells = nullptr);

};

}

#endif
