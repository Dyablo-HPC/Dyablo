#include "InitialConditions_base.h"
#include "AnalyticalFormula.h"

#include "foreach_cell/ForeachCell.h"
#include "particles/ForeachParticle.h"

#include "states/State_forward.h"

namespace dyablo{

class InitialConditions_simple_particles : public InitialConditions{ 
    //ForeachCell& foreach_cell;
    ForeachParticle foreach_particle;
    real_t gamma0;
    int npart;
    Kokkos::View<double*> px,py,pz,vx,vy,vz,mass;
public:
  InitialConditions_simple_particles(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
  : //foreach_cell(foreach_cell),
    foreach_particle( foreach_cell.get_amr_mesh(), configMap ),
    gamma0(configMap.getValue<real_t>("hydro", "gamma0", 1.4)),
    npart(configMap.getValue<int>("simple_particles", "npart", 1)),
    px( "px", npart ), py( "py", npart ), pz( "pz", npart ), 
    vx( "vx", npart ), vy( "vy", npart ), vz( "vz", npart ), 
    mass( "mass", npart )
  {    
    auto parse_array = [&](const Kokkos::View<double*>& a, const std::string& var)
    {
      std::vector<double> values = configMap.getValue< std::vector<double> >("simple_particles", var, {});
      int nb_values = std::max((int)values.size(),npart); // Select at most npart values from .ini

      // Create unmanaged view to copy vector
      using UnmanagedHostView = Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
      UnmanagedHostView values_host( values.data(), nb_values ); 
      
      auto values_device = Kokkos::subview( a, std::make_pair(0,nb_values) );

      Kokkos::deep_copy(values_device, values_host);
    };

    parse_array(px, "px");
    parse_array(py, "py");
    parse_array(pz, "pz");
    parse_array(vx, "vx");
    parse_array(vy, "vy");
    parse_array(vz, "vz");
    parse_array(mass, "mass");

  }

  void init( UserData& U )
  {
    // Setting up particles
    int rank = GlobalMpiSession::get_comm_world().MPI_Comm_rank();

    if (rank == 0) {    
      U.new_ParticleArray("particles", npart);
      U.new_ParticleAttribute("particles", "vx");
      U.new_ParticleAttribute("particles", "vy");
      U.new_ParticleAttribute("particles", "vz");
      U.new_ParticleAttribute("particles", "mass");

      const ForeachParticle::ParticleArray& P = U.getParticleArray("particles"); 

      enum VarIndex_particle{
        IVX, IVY, IVZ, IM
      };

      const UserData::ParticleAccessor Pdata = U.getParticleAccessor("particles", 
                                              {{"vx", IVX}, 
                                               {"vy", IVY},
                                               {"vz", IVZ},
                                               {"mass", IM}});

      const Kokkos::View<double*>& px = this->px;
      const Kokkos::View<double*>& py = this->py;
      const Kokkos::View<double*>& pz = this->pz;
      const Kokkos::View<double*>& vx = this->vx;
      const Kokkos::View<double*>& vy = this->vy;
      const Kokkos::View<double*>& vz = this->vz;
      const Kokkos::View<double*>& mass = this->mass;

      foreach_particle.foreach_particle("InitialConditions_simple_particles", P,
        KOKKOS_LAMBDA (ParticleData::ParticleIndex iPart) {      
          P.pos(iPart, IX) = px(iPart);
          P.pos(iPart, IY) = py(iPart);
          P.pos(iPart, IZ) = pz(iPart);

          Pdata.at(iPart, IVX) = vx(iPart);
          Pdata.at(iPart, IVY) = vy(iPart);
          Pdata.at(iPart, IVZ) = vz(iPart);
          Pdata.at(iPart, IM)  = mass(iPart);
        });

    }

    U.distributeParticles("particles");
  }  
}; 

} // namespace dyablo


FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_simple_particles, 
                 "simple_particles");

