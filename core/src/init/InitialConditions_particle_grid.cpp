#include "InitialConditions_base.h"
#include "AnalyticalFormula.h"

#include "foreach_cell/ForeachCell.h"
#include "particles/ForeachParticle.h"

#include "states/State_forward.h"

namespace dyablo{

/**
 * Create particles along a cartesian grid, initial particle velocities are set using 
 * the rho_vx, rho_vy, rho_vz fields
 * Configured in .ini in the [particle_grid] section
 * - nx, ny, nz, the number of particles in each direction
 * - total_mass the cumulated mass of all the particles, each particle has the same mass
 * - dt_perturb : dt used to displace particles (using particle velocities)
 * 
**/
class InitialConditions_particle_grid : public InitialConditions{ 
    ForeachCell& foreach_cell;
    ForeachParticle foreach_particle;
    const uint32_t nx, ny, nz;
    const real_t xmin, xmax;
    const real_t ymin, ymax;
    const real_t zmin, zmax;
    const real_t total_mass;
    const real_t dt_perturb;
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
    dt_perturb(configMap.getValue<real_t>("particle_grid", "dt_perturb", 0)),
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

    real_t Lx = this->xmax-this->xmin;
    real_t Ly = this->ymax-this->ymin;
    real_t Lz = this->zmax-this->zmin;
    real_t dx = (Lx)/this->nx;
    real_t dy = (Ly)/this->ny;
    real_t dz = (Lz)/this->nz;
    real_t xmin = this->xmin;
    real_t ymin = this->ymin;
    real_t zmin = this->zmin;

    {
      UserData::ParticleArray_t P = U.getParticleArray( particle_array_name );
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
    }

    U.distributeParticles(particle_array_name);

    bool mass = this->total_mass / nbpart_global;
    real_t dt_perturb = this->dt_perturb;

    U.new_ParticleAttribute(particle_array_name, "vx");
    U.new_ParticleAttribute(particle_array_name, "vy");
    U.new_ParticleAttribute(particle_array_name, "vz");
    U.new_ParticleAttribute(particle_array_name, "mass");
    
    enum VarIndex_particle_grid { IVX, IVY, IVZ, IMASS, IRHO };
      

    {
      UserData::ParticleArray_t P = U.getParticleArray( particle_array_name );
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

        Pout.at( iPart, IMASS ) = mass;

        real_t rho = Uin.at(iCell, IRHO);
        real_t u = Uin.at(iCell, IVX) / rho;
        real_t v = Uin.at(iCell, IVY) / rho;
        real_t w = Uin.at(iCell, IVZ) / rho;

        Pout.at( iPart, IVX ) = u;
        Pout.at( iPart, IVY ) = v;
        Pout.at( iPart, IVZ ) = w;

        P.pos( iPart, IX ) = P.pos( iPart, IX ) + u * dt_perturb;
        P.pos( iPart, IY ) = P.pos( iPart, IY ) + v * dt_perturb;
        P.pos( iPart, IZ ) = P.pos( iPart, IZ ) + w * dt_perturb;

        // Compute periodic position
        P.pos(iPart, IX) = fmod( (P.pos(iPart, IX) - xmin) + (Lx) , Lx) + xmin;
        P.pos(iPart, IY) = fmod( (P.pos(iPart, IY) - ymin) + (Ly) , Ly) + ymin;
        P.pos(iPart, IZ) = fmod( (P.pos(iPart, IZ) - zmin) + (Lz) , Lz) + zmin;

      });
    }

    U.distributeParticles(particle_array_name);
  }  
}; 

} // namespace dyablo


FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_particle_grid, 
                 "particle_grid");

