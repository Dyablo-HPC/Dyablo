#include "InitialConditions_base.h"
#include "AnalyticalFormula.h"

#include "foreach_cell/ForeachCell.h"
#include "particles/ForeachParticle.h"

#include "states/State_forward.h"

namespace dyablo{

class InitialConditions_particle_grid : public InitialConditions{ 
    ForeachCell& foreach_cell;
    ForeachParticle foreach_particle;
    const uint32_t nx, ny, nz;
    const real_t xmin, xmax;
    const real_t ymin, ymax;
    const real_t zmin, zmax;
    const real_t total_mass;
    std::string particle_array_name;

public:
  InitialConditions_particle_grid(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
  : foreach_cell(foreach_cell),
    foreach_particle( foreach_cell.get_amr_mesh(), configMap ),
    nx(configMap.getValue<uint32_t>("particle_grid", "nx", 4)),
    ny(configMap.getValue<uint32_t>("particle_grid", "ny", 4)),
    nz(configMap.getValue<uint32_t>("particle_grid", "nz", 4)),
    xmin( configMap.getValue<real_t>("mesh", "xmin", 0.0) ), xmax( configMap.getValue<real_t>("mesh", "xmax", 1.0) ),
    ymin( configMap.getValue<real_t>("mesh", "ymin", 0.0) ), ymax( configMap.getValue<real_t>("mesh", "ymax", 1.0) ),
    zmin( configMap.getValue<real_t>("mesh", "zmin", 0.0) ), zmax( configMap.getValue<real_t>("mesh", "zmax", 1.0) ),
    total_mass(configMap.getValue<real_t>("particle_grid", "total_mass", 1.0)),
    particle_array_name(configMap.getValue<std::string>("particle_grid", "particle_array_name", "particles"))
  {}

  void init( UserData& U )
  {
    int mpi_size = GlobalMpiSession::get_comm_world().MPI_Comm_size();
    int mpi_rank = GlobalMpiSession::get_comm_world().MPI_Comm_rank();

    uint32_t nx = this->nx;
    uint32_t ny = this->ny;
    uint32_t nz = this->nz;
    uint64_t nbpart_global = nx*ny*nz;
    uint64_t ipart_global_begin = nbpart_global*mpi_rank/mpi_size;
    uint64_t ipart_global_end = nbpart_global*(mpi_rank+1)/mpi_size;
    uint32_t nbpart_local = ipart_global_end-ipart_global_begin;

    U.new_ParticleArray(particle_array_name, nbpart_local);
    UserData::ParticleArray_t P = U.getParticleArray( particle_array_name );

    real_t dx = (this->xmax-this->xmin)/this->nx;
    real_t dy = (this->ymax-this->ymin)/this->ny;
    real_t dz = (this->zmax-this->zmin)/this->nz;
    real_t xmin = this->xmin;
    real_t ymin = this->ymin;
    real_t zmin = this->zmin;

    foreach_particle.foreach_particle("InitialConditions_particle_grid::init_pos", P,
    KOKKOS_LAMBDA (ParticleData::ParticleIndex iPart) 
    {
      uint64_t ipart_global = ipart_global_begin + iPart;

      // TODO : maybe add particles in morton order?
      uint32_t i = ipart_global%nx;
      uint32_t j = (ipart_global/nx)%ny;
      uint32_t k = (ipart_global/nx)/ny;

      P.pos(iPart, IX) = xmin + i*dx + 0.5*dx;
      P.pos(iPart, IY) = ymin + j*dy + 0.5*dy;
      P.pos(iPart, IZ) = zmin + k*dz + 0.5*dz;
    });

    U.distributeParticles(particle_array_name);

    bool mass = this->total_mass / nbpart_global;

    U.new_ParticleAttribute(particle_array_name, "vx");
    U.new_ParticleAttribute(particle_array_name, "vy");
    U.new_ParticleAttribute(particle_array_name, "vz");
    U.new_ParticleAttribute(particle_array_name, "mass");
    
    enum VarIndex_particle_grid { IVX, IVY, IVZ, IMASS, IRHO };
    auto Pout = U.getParticleAccessor( particle_array_name,
                                      { {"vx", IVX}, 
                                        {"vy", IVY},
                                        {"vz", IVZ},
                                        {"mass", IMASS} });
    auto Uin = U.getAccessor( { {"rho", IRHO},
                                {"rho_vx", IVX}, 
                                {"rho_vy", IVY},
                                {"rho_vz", IVZ} });


    ForeachCell::CellMetaData cells = foreach_cell.getCellMetaData();

    foreach_particle.foreach_particle("InitialConditions_particle_grid::init_attributes", P,
    KOKKOS_LAMBDA (ParticleData::ParticleIndex iPart) 
    {
      ForeachCell::CellMetaData::pos_t pos = {P.pos(iPart, IX), P.pos(iPart, IY), P.pos(iPart, IZ)};
      ForeachCell::CellIndex iCell = cells.getCellFromPos( pos );

      real_t rho = Uin.at(iCell, IRHO);
      Pout.at( iPart, IVX ) = Uin.at(iCell, IVX) / rho;
      Pout.at( iPart, IVY ) = Uin.at(iCell, IVY) / rho;
      Pout.at( iPart, IVZ ) = Uin.at(iCell, IVZ) / rho;

      Pout.at( iPart, IMASS ) = mass;
    });
  }  
}; 

} // namespace dyablo


FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_particle_grid, 
                 "particle_grid");

