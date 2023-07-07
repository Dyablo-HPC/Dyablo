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
    uint32_t nx, ny, nz;
    const real_t xmin, xmax;
    const real_t ymin, ymax;
    const real_t zmin, zmax;
    real_t total_mass;
    real_t dt_perturb;
    std::string particle_array_name;

public:
  InitialConditions_particle_grid(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
  : foreach_cell(foreach_cell),
    foreach_particle( foreach_cell.get_amr_mesh(), configMap ),
    xmin( configMap.getValue<real_t>("mesh", "xmin", 0.0) ), xmax( configMap.getValue<real_t>("mesh", "xmax", 1.0) ),
    ymin( configMap.getValue<real_t>("mesh", "ymin", 0.0) ), ymax( configMap.getValue<real_t>("mesh", "ymax", 1.0) ),
    zmin( configMap.getValue<real_t>("mesh", "zmin", 0.0) ), zmax( configMap.getValue<real_t>("mesh", "zmax", 1.0) ),
    
    particle_array_name(configMap.getValue<std::string>("particle_grid", "particle_array_name", "particles"))
  {
    AMRmesh& pmesh = foreach_cell.get_amr_mesh();
    uint32_t default_nx = foreach_cell.blockSize()[IX]*pmesh.get_coarse_grid_size()[IX];
    uint32_t default_ny = foreach_cell.blockSize()[IY]*pmesh.get_coarse_grid_size()[IY];
    uint32_t default_nz = foreach_cell.blockSize()[IZ]*pmesh.get_coarse_grid_size()[IZ];
    this->nx = configMap.getValue<uint32_t>("particle_grid", "nx", default_nx);
    this->ny = configMap.getValue<uint32_t>("particle_grid", "ny", default_ny);
    this->nz = configMap.getValue<uint32_t>("particle_grid", "nz", default_nz);
    
    real_t default_total_mass=1.0, default_dt_perturb=0.0;
    bool cosmo = configMap.hasValue("cosmology", "astart");
    if( cosmo )
    {
      real_t astart = configMap.getValue<real_t>( "cosmology", "astart" );
      real_t omegam = configMap.getValue<real_t>( "cosmology", "omegam" );
      real_t omegab = configMap.getValue<real_t>( "cosmology", "omegab" );
      real_t omegav = configMap.getValue<real_t>( "cosmology", "omegav" );
      
      real_t omegak = 1.0 - omegam - omegav;
      real_t eta    = sqrt(omegam / astart + omegav * astart * astart + omegak);
      real_t dladt = astart * eta;

      real_t fomega;
      if (omegam >= 1.0 && omegav <= 0.0)
        fomega = 1.0;
      else
      {
        real_t dplus;
        {
          auto ddplus = [&](real_t a)
          {
            if (a <= 0.0)
              return 0.0;

            real_t eta = sqrt(omegam / a + omegav * a * a + 1.0 - omegam - omegav);
            return 2.5 / (eta * eta * eta);
          };

          // UGLY trapezoid rule integration
          const real_t Np = 1000;
          const real_t da = astart / Np;
          real_t sum      = 0.0;
          for (int i = 0; i < Np; ++i)
          {
            sum += 0.5 * (ddplus(i * da) + ddplus((i + 1) * da));
          }
          sum *= da;

          dplus = eta / astart * sum;
        }
        fomega = (2.5 / dplus - 1.5 * omegam / astart - omegak) / (eta * eta);
      }

      default_total_mass = 1.0-omegab/omegam;
      default_dt_perturb = 1/ (2 * fomega * dladt / sqrt(omegam));
    }

    this->total_mass = configMap.getValue<real_t>("particle_grid", "total_mass", default_total_mass);
    this->dt_perturb = configMap.getValue<real_t>("particle_grid", "dt_perturb", default_dt_perturb);

  }

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

