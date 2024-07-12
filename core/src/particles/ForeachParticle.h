#pragma once

#include "kokkos_shared.h"
#include "particles/ParticleArray.h"
#include "mpi/ViewCommunicator.h"

namespace dyablo
{

#define PARTICLE_LAMBDA KOKKOS_LAMBDA

class ForeachParticle
{
public:
  using ParticleArray = dyablo::ParticleArray;
  using ParticleIndex = ParticleArray::ParticleIndex;

  ForeachParticle(AMRmesh &pmesh, ConfigMap &configMap) 
    : pmesh(pmesh),
      xmin( configMap.getValue<real_t>("mesh", "xmin", 0) ),
      ymin( configMap.getValue<real_t>("mesh", "ymin", 0) ),
      zmin( configMap.getValue<real_t>("mesh", "zmin", 0) ),
      xmax( configMap.getValue<real_t>("mesh", "xmax", 1) ),
      ymax( configMap.getValue<real_t>("mesh", "ymax", 1) ),
      zmax( configMap.getValue<real_t>("mesh", "zmax", 1) )
  {}

  template <typename Function>
  void foreach_particle(const std::string &kernel_name, const ParticleArray &iter_space,
                        const Function &f) const
  {
    Kokkos::parallel_for(
        kernel_name, iter_space.getNumParticles(), 
        KOKKOS_LAMBDA(uint32_t iPart) 
    { 
      f(iPart); 
    });
  }

  template <typename Function, typename... Reducer_t>
  void reduce_particle(const std::string &kernel_name, const ParticleArray &iter_space,
                      const Function &f, const Reducer_t&... reducer ) const
  {
    Kokkos::parallel_reduce(
        kernel_name, iter_space.getNumParticles(), 
        KOKKOS_LAMBDA(uint32_t iPart, typename Reducer_t::value_type&... update) 
    { 
      f(iPart, update...); 
    }, reducer...);
  }

  template <typename Function, typename... Value_t>
  void reduce_particle(const std::string& kernel_name, const ParticleArray& iter_space, 
                       const Function& f, Value_t&... reducer) const
  {
    reduce_particle(kernel_name, iter_space, f, Kokkos::Sum<Value_t>(reducer)...);
  }

  ViewCommunicator get_distribute_communicator(const ParticleArray& particles_in)
  {
    const LightOctree& lmesh = pmesh.getLightOctree();
    uint32_t nbParticles = particles_in.getNumParticles();

    real_t xmin = this->xmin;
    real_t ymin = this->ymin;
    real_t zmin = this->zmin;
    real_t Lx = xmax - xmin;
    real_t Ly = ymax - ymin;
    real_t Lz = zmax - zmin;

    // Compute new particle domain and count particles to sent to each process
    Kokkos::View<int*> particle_domain( "particle_domain", nbParticles);
    foreach_particle("ForeachParticle::distribute::compute_exchange_list",
                     particles_in,
                     PARTICLE_LAMBDA( const ParticleIndex& iPart )
    {
      real_t x = particles_in.pos(iPart, IX);
      real_t y = particles_in.pos(iPart, IY);
      real_t z = particles_in.pos(iPart, IZ);
      real_t x01 = (x-xmin)/Lx;
      real_t y01 = (y-ymin)/Ly;
      real_t z01 = (z-zmin)/Lz;
      assert( x01 >= 0 && x01 < 1.0 );
      assert( y01 >= 0 && y01 < 1.0 );
      assert( z01 >= 0 && x01 < 1.0 );

      int domain = lmesh.getDomainFromPos({x01,y01,z01});

      particle_domain(iPart) = domain;
    });

    return ViewCommunicator( particle_domain, pmesh.getMpiComm() );
  }

  void distribute( ParticleData& particles_in )
  {
    ViewCommunicator part_comm = get_distribute_communicator(particles_in);
    uint32_t nbParticles_new = part_comm.getNumGhosts();
    // TODO fetch old name
    ParticleData particles_out( ParticleArray("xxx", nbParticles_new), particles_in.field_manager() );
    part_comm.exchange_ghosts<0>( particles_in.particle_position, particles_out.particle_position );
    part_comm.exchange_ghosts<0>( particles_in.particle_data, particles_out.particle_data );

    particles_in = particles_out;
  }

private:
  AMRmesh &pmesh;
  real_t xmin, ymin, zmin; /// min corner of physical domain
  real_t xmax, ymax, zmax; /// min corner of physical domain 
};

} // namespace dyablo