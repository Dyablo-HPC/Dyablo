#pragma once

#include <csignal>

#include "utils/config/ConfigMap.h"
#include "utils/monitoring/Timers.h"
#include "shared/amr/AMRmesh.h"
#include "muscl_block/foreach_cell/AMRBlockForeachCell_group.h"

#include "muscl_block/compute_dt/Compute_dt.h"
#include "muscl_block/refine_condition/RefineCondition.h"
#include "muscl_block/init/InitialConditions.h"
#include "muscl_block/io/IOManager.h"
#include "muscl_block/gravity/GravitySolver.h"
#include "muscl_block/update/MusclBlockUpdate.h"
#include "muscl_block/MapUserData.h"

namespace dyablo {
namespace muscl_block { // TODO : remove from muscl_block

/**
 * Main class to run Dyablo simulation
 **/
class DyabloTimeLoop{
private:
  AMRmesh init_amr_mesh( ConfigMap& configMap );
public:
  /**
   * Create ans initialize a simulation
   **/
  DyabloTimeLoop( ConfigMap& configMap )
  : m_iter_end( configMap.getInteger("run","nstepmax",1000) ),    
    m_t_end( configMap.getFloat("run", "tEnd", 0.0) ),
    m_nlog( configMap.getFloat("run", "nlog", 10) ),
    m_output_frequency( configMap.getFloat("run", "output_frequency", -1) ),
    m_output_timeslice( configMap.getFloat("run", "output_timeslice", -1) ),
    m_amr_cycle_frequency( configMap.getInteger("amr", "cycle_frequency", 1) ),
    m_loadbalance_frequency( configMap.getInteger("amr", "load_balancing_frequency", 10) ),
    m_communicator( GlobalMpiSession::get_comm_world() )
  {
    // TODO : remove HydroParams
    HydroParams params;
    params.setup(configMap);  // TODO : first step : fill params only with used values


    FieldManager field_manager = FieldManager::setup(params, configMap);
    auto fm = field_manager.get_id2index();

    uint32_t bx = configMap.getInteger("amr", "bx", 0);
    uint32_t by = configMap.getInteger("amr", "by", 0);
    uint32_t bz = configMap.getInteger("amr", "bz", 1);

    {
      int ndim = params.dimType == TWO_D ? 2 : 3;
      //int ndim = configMap.getInteger("mesh", "ndim", 3);
      int codim = ndim;
      BoundaryConditionType bxmin  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_xmin", BC_ABSORBING));
      BoundaryConditionType bxmax  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_xmax", BC_ABSORBING));
      BoundaryConditionType bymin  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_ymin", BC_ABSORBING));
      BoundaryConditionType bymax  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_ymax", BC_ABSORBING));
      BoundaryConditionType bzmin  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_zmin", BC_ABSORBING));
      BoundaryConditionType bzmax  = static_cast<BoundaryConditionType>(configMap.getInteger("mesh","boundary_type_zmax", BC_ABSORBING));
      std::array<bool,3> periodic = {
        bxmin == BC_PERIODIC || bxmax == BC_PERIODIC,
        bymin == BC_PERIODIC || bymax == BC_PERIODIC,
        bzmin == BC_PERIODIC || bzmax == BC_PERIODIC
      };
      int amr_level_min = configMap.getInteger("amr","level_min", 5);
      int amr_level_max = configMap.getInteger("amr","level_max", 10);
      this->m_amr_mesh = std::make_shared<AMRmesh>( ndim, codim, periodic, amr_level_min, amr_level_max );
    }

    AMRmesh& amr_mesh = *m_amr_mesh;

    std::string godunov_updater_id = configMap.getString("hydro", "update", "MusclBlockUpdate_generic");
    this->godunov_updater = MusclBlockUpdateFactory::make_instance( godunov_updater_id,
      configMap,
      params,
      amr_mesh, 
      fm,
      bx, by, bz,
      timers
    );

    std::string iomanager_id = configMap.getString("output", "backend", "IOManager_hdf5");
    this->io_manager = IOManagerFactory::make_instance( iomanager_id,
      configMap,
      params,
      amr_mesh, 
      field_manager,
      bx, by, bz,
      timers
    );

    GravityType gravity_type = static_cast<GravityType>(configMap.getInteger("gravity", "gravity_type", GRAVITY_NONE));

    std::string gravity_solver_id = "none";
    if( gravity_type & GRAVITY_FIELD )
    {
      gravity_solver_id = configMap.getString("gravity", "solver", "GravitySolver_none");
      this->gravity_solver = GravitySolverFactory::make_instance( gravity_solver_id,
        configMap,
        params,
        m_amr_mesh, 
        fm,
        bx, by, bz,
        timers
      );
    } 

    this->compute_dt = std::make_unique<Compute_dt>(
      configMap,
      params,
      amr_mesh, 
      timers
    );

    this->refine_condition = std::make_unique<RefineCondition>(
      configMap,
      params,
      amr_mesh, 
      fm,
      bx, by, bz,
      timers
    );

    // Get initial conditions id
    std::string init_id = configMap.getString("hydro", "problem", "unknown");

    int rank = m_communicator.MPI_Comm_rank();
    if (rank==0) {
      std::cout << "##########################" << "\n";
      std::cout << "Godunov updater    : " << godunov_updater_id << std::endl;
      std::cout << "IO Manager         : " << iomanager_id << std::endl;
      std::cout << "Gravity solver     : " << gravity_solver_id << std::endl;
      std::cout << "Initial conditions : " << init_id << std::endl;
      std::cout << "##########################" << std::endl;
      
      float Udata_mem_size = DataArrayBlock::required_allocation_size( U.U.extent(0), U.U.extent(1), U.U.extent(2) ) * (2 / 1e6) ;

      // print parameters on screen
      params.print();
      std::cout << "##########################" << "\n";
      std::cout << "Memory requested (U + U2) : " << Udata_mem_size << " MBytes\n"; 
      std::cout << "##########################" << "\n";
    }

    // Initialize cells
    {
      // test if we are performing a re-start run (default : false)
      bool restartEnabled = configMap.getBool("run","restart_enabled", false);

      std::string init_name = restartEnabled ? "restart" : init_id;
      std::unique_ptr<InitialConditions> initial_conditions =
        InitialConditionsFactory::make_instance(init_id, 
          configMap,
          params,
          amr_mesh, 
          field_manager,
          1024,
          bx, by, bz,
          timers);

      initial_conditions->init(U);
    }

    // Allocate U2
    {
      const LightOctree& lmesh = amr_mesh.getLightOctree();
      ForeachCell foreach_cell{
        lmesh.getNdim(),
        lmesh, 
        bx, by, bz, 
        0, 0, 0, // Dummy foreach_cell to construct arrays
        0, 0, 0,
        0
      };

      U2 = foreach_cell.allocate_ghosted_array( "U2", amr_mesh, field_manager );
    }        
  }

  static int interrupted;
  static void interrupt_handler( int sig )
  {
    if( !interrupted )
    {
      std::cout << "Signal " << sig << " : ending simulation..." << std::endl;
      interrupted = true;
    }
    else
    {
      std::cout << "Signal " << sig << " : killing simulation!" << std::endl;
      raise(SIGABRT);
    }
  }

  /**
   * Run the simulation for multiple timesteps until a stop condition is met
   **/
  void run()
  {
    timers.get("Total").start();

    // Stop simulation when recieving SIGINT 
    // NOTE : mpirun catches SIGINT and forwards SIGTERM to child process
    // NOTE : scancel sends SIGTERM by default, you can use 'scancel -s INT'
    signal( SIGINT, interrupt_handler );
    bool finished = false;
    while( !finished )
    {
      step();
      m_iter++;
      int any_interrupted;
      m_communicator.MPI_Allreduce(&interrupted, &any_interrupted, 1, MpiComm::MPI_Op_t::OR);
      finished = ( m_t_end > 0    && m_t >= (m_t_end - 1e-14) ) // End if physical time exceeds end time
              || ( m_iter_end > 0 && m_iter >= m_iter_end    )  // Or if iter count exceeds maximum iter count
              || any_interrupted;
    }
    signal( SIGINT, SIG_DFL );

    // Always output after last iteration
    timers.get("outputs").start();
    if( m_output_frequency > 0 || m_output_timeslice > 0 )
      io_manager->save_snapshot(U.U, U.Ughost, m_iter, m_t); // TODO use CellArray instead of DataArrayBlock
    timers.get("outputs").stop();
    timers.get("Total").stop();

    int rank = m_communicator.MPI_Comm_rank();
    if ( rank == 0 ) 
    {
      printf("final time is %f\n", m_t);

      timers.print();

      // Todo : count cell updates
      //real_t t_tot   = timers.get("total").elapsed(Timers::Timer::Elapsed_mode_t::ELAPSED_CPU);
      // printf("Perf             : %5.3f number of Mcell-updates/s\n",
      //       1.0 * m_total_num_cell_updates * (bx*by*bz) / t_tot * 1e-6);

      // printf("Total number of cell-updates : %ld\n",
      //       m_total_num_cell_updates * (bx*by*bz));
    }

  }

  /**
   * Run one timestep of the simulation
   **/
  void step()
  {
    // Write output files
    {
      timers.get("outputs").start();
      // Always output first iteration
      bool first_iter = (m_iter == 0);
      // Output at fixed frequency set by 'm_output_frequency'
      bool frequency_trigger = ( m_output_frequency > 0 && m_iter % m_output_frequency == 0 );
      // Output at fixed physical time interval 'm_output_timeslice'
      bool timeslice_trigger = ( m_output_timeslice > 0 && (m_t - m_output_timeslice*m_output_timeslice_count) >= m_output_timeslice );
      if( timeslice_trigger )
        m_output_timeslice_count++;        

      if( first_iter || frequency_trigger || timeslice_trigger )
      {
        int rank = m_communicator.MPI_Comm_rank();
        if( rank == 0 )
        {
          std::cout << "Output results at time t=" << m_t << " step " << m_iter << std::endl;
        }
        io_manager->save_snapshot(U.U, U.Ughost, m_iter, m_t);
      }
      timers.get("outputs").stop();
    }

    // Compute new dt
    real_t dt = 0;    
    {
      timers.get("dt").start();
      double dt_local = compute_dt->compute_dt( U );

      m_communicator.MPI_Allreduce(&dt_local, &dt, 1, MpiComm::MPI_Op_t::MIN);
      // correct dt if end of simulation
      if (m_t_end > 0 && m_t + dt > m_t_end) {
        dt = m_t_end - m_t;
      }
      timers.get("dt").stop();
    }

    // Log iteration    
    { // Todo make a logger
      int rank = m_communicator.MPI_Comm_rank();
      if( rank == 0 && m_iter % m_nlog == 0 )
        printf("time step=%7d (dt=% 10.8f t=% 10.8f)\n",m_iter, dt, m_t);
    }
    
    GhostCommunicator ghost_comm(m_amr_mesh, m_communicator);

    // Update ghost cells
    // TODO use a timer INSIDE exchange_ghosts()?
    timers.get("MPI ghosts").start();
    U.exchange_ghosts( ghost_comm );
    timers.get("MPI ghosts").stop();
    
    // Update gravity
    if( gravity_solver )
      gravity_solver->update_gravity_field(U.U, U.Ughost, U.U);

    // Update hydro
    godunov_updater->update( U.U, U.Ughost, U2.U, dt ); //TODO : make U2 a temporary array?

    m_t += dt;

    // AMR cycle
    {
      if( m_amr_cycle_frequency > 0 && ( (m_iter % m_amr_cycle_frequency) == 0 ) )
      {
        timers.get("AMR").start();

        timers.get("MPI ghosts").start();
        U2.exchange_ghosts( ghost_comm );
        timers.get("MPI ghosts").stop();

        timers.get("AMR: Mark cells").start();
        refine_condition->mark_cells( U2 );
        timers.get("AMR: Mark cells").stop();

        // Backup old mesh
        LightOctree lmesh_old = m_amr_mesh->getLightOctree();

        timers.get("AMR: adapt").start();
        // 1. adapt mesh with mapper enabled
        m_amr_mesh->adapt(true);
        // Verify that adapt() doesn't need another iteration
        assert(m_amr_mesh->check21Balance());
        assert(!m_amr_mesh->checkToAdapt());        
        timers.get("AMR: adapt").stop();

        // Resize and fill U with copied/interpolated/extrapolated data
        timers.get("AMR: remap userdata").start();
        std::cout << "Reallocate U + U2 after remap : " << DataArrayBlock::required_allocation_size(U2.U.extent(0), U2.U.extent(1), U2.U.extent(2)) * (2/1e6) 
            << " -> " << DataArrayBlock::required_allocation_size(U2.U.extent(0), U2.U.extent(1), m_amr_mesh->getNumOctants()) * (2/1e6) << " MBytes" << std::endl;
        MapUserDataFunctor::apply( lmesh_old, m_amr_mesh->getLightOctree(), {U.bx,U.by,U.bz},
                      U2.U, U.Ughost, U.U );
        // now U contains the most up to date data after mesh adaptation
        // we can resize U2 for the next time-step
        Kokkos::realloc(U2.U, U.U.layout());
        timers.get("AMR: remap userdata").stop();

        timers.get("AMR").stop();
      }
      else
      {
        //TODO write U.deep_copy() or something
        Kokkos::deep_copy(U.U, U2.U);
      }
    }

    //Load Balancing
    {
      if( m_loadbalance_frequency > 0 && ( (m_iter % m_loadbalance_frequency) == 0 ) )
      {
        timers.get("AMR: load-balance").start();

        /* (Load)Balance the octree over the processes with communicating the data.
        * Preserve the family compact up to 3 levels over the max deepth reached
        * in the octree. */
        uint8_t levels = 3;

        m_amr_mesh->loadBalance_userdata(levels, U.U);

        // we probably need to resize arrays, ....
        // TODO clean
        Kokkos::realloc(U2.U,U.U.layout());
        Kokkos::realloc(U.Ughost, U.Ughost.extent(0), U.Ughost.extent(1), m_amr_mesh->getNumGhosts());
  
        timers.get("AMR: load-balance").stop();
      }
    }    
  }

private:
  // Simulation parameters
  int m_iter_end; //! Number of iterations before ending the simulation 
  real_t m_t_end; //! Physical time to end the simulation
  int m_nlog; //! Timestep log frequency  
  int m_output_frequency; //! Maximum number of iterations between outputs
  double m_output_timeslice; //! Physical time interval between outputs
  int m_amr_cycle_frequency; //! Number of iterations between amr cycles (<= 0 is never)
  int m_loadbalance_frequency; //! Number of iterations between load balancing (<= 0 is never)

  int m_iter = 0; //! Current Iteration number
  real_t m_t = 0; //! Current physical time
  int m_output_timeslice_count = 0; //! Number of timeslices already written

  MpiComm m_communicator;
  std::shared_ptr<AMRmesh> m_amr_mesh;

  using ForeachCell = AMRBlockForeachCell_group;
  using CellArray = AMRBlockForeachCell_group::CellArray_global_ghosted;

  CellArray U, U2;

  std::unique_ptr<Compute_dt> compute_dt;
  std::unique_ptr<RefineCondition> refine_condition;
  std::unique_ptr<MusclBlockUpdate> godunov_updater;
  std::unique_ptr<IOManager> io_manager;
  std::unique_ptr<GravitySolver> gravity_solver;

  Timers timers;
};

int DyabloTimeLoop::interrupted = false;

} // namespace muscl_block
} // namespace dyablo