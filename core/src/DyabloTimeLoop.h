#pragma once

#include <csignal>

#include "utils/config/ConfigMap.h"
#include "utils/monitoring/Timers.h"
#include "amr/AMRmesh.h"
#include "foreach_cell/ForeachCell.h"

#include "compute_dt/Compute_dt.h"
#include "refine_condition/RefineCondition.h"
#include "init/InitialConditions.h"
#include "io/IOManager.h"
#include "gravity/GravitySolver.h"
#include "hydro/HydroUpdate.h"
#include "amr/MapUserData.h"
#include "UserData.h"

namespace dyablo {


/**
 * Main class to run Dyablo simulation
 **/
class DyabloTimeLoop{
private:
  std::shared_ptr<AMRmesh> init_amr_mesh( ConfigMap& configMap )
  {
    AMRmesh::Parameters p = AMRmesh::parse_parameters(configMap);
    return std::make_shared<AMRmesh>( p.dim, p.dim, p.periodic, p.level_min, p.level_max, 
                                      p.coarse_grid_size );
  }
public:
  /**
   * Create ans initialize a simulation
   **/
  DyabloTimeLoop( ConfigMap& configMap )
  : m_iter_end( configMap.getValue<int>("run","nstepmax",1000) ),    
    m_t_end( configMap.getValue<real_t>("run", "tEnd", 0.0) ),
    m_nlog( configMap.getValue<int>("run", "nlog", 10) ),
    m_enable_output( configMap.getValue<bool>("run", "enable_output", true) ),
    m_output_frequency( configMap.getValue<int>("run", "output_frequency", -1) ),
    m_output_timeslice( configMap.getValue<real_t>("run", "output_timeslice", -1) ),  
    m_enable_checkpoint( configMap.getValue<bool>("run", "enable_checkpoint", true) ),
    m_checkpoint_frequency( configMap.getValue<int>("run", "checkpoint_frequency", -1) ),
    m_checkpoint_timeslice( configMap.getValue<real_t>("run", "checkpoint_timeslice", -1) ), 
    m_amr_cycle_frequency( configMap.getValue<int>("amr", "cycle_frequency", 1) ),
    m_loadbalance_frequency( configMap.getValue<int>("amr", "load_balancing_frequency", 10) ),
    m_iter_start( configMap.getValue<int>("run", "iter_start", 0) ),
    m_iter( m_iter_start ),
    m_t( configMap.getValue<real_t>("run", "tStart", 0.0) ), 
    m_output_timeslice_count( m_t / m_output_timeslice ),
    m_checkpoint_timeslice_count( m_t / m_checkpoint_timeslice ),
    m_communicator( GlobalMpiSession::get_comm_world() ),
    m_amr_mesh( init_amr_mesh( configMap ) ),
    m_foreach_cell( *m_amr_mesh, configMap ),
    U(configMap, m_foreach_cell),
    m_loadbalance_coherent_levels( configMap.getValue<int>("amr", "loadbalance_coherent_levels", 3) )
  {
    configMap.getValue<int>("mesh", "ndim", 3);
    

    std::string gravity_solver_id = configMap.getValue<std::string>("gravity", "solver", "none");
    if( gravity_solver_id == "none" )
    {
      m_gravity_type = GRAVITY_NONE;
      this->gravity_solver = nullptr;
    }
    else
    {
      this->gravity_solver = GravitySolverFactory::make_instance( gravity_solver_id,
        configMap,
        m_foreach_cell,
        timers
      );
      // [gravity]gravity_type should be set from the instanciaion of gravity_solver
      m_gravity_type = configMap.getValue<GravityType>("gravity", "gravity_type", GRAVITY_NONE);
    }

    std::string godunov_updater_id = configMap.getValue<std::string>("hydro", "update", "HydroUpdate_hancock");
    this->has_mhd = godunov_updater_id.find("MHD") != std::string::npos;
    this->godunov_updater = HydroUpdateFactory::make_instance( godunov_updater_id,
      configMap,
      this->m_foreach_cell,
      timers
    );

    std::string mapUserData_id = configMap.getValue<std::string>("amr", "remap", "MapUserData_mean");
    this->mapUserData = MapUserDataFactory::make_instance( mapUserData_id,
      configMap,
      this->m_foreach_cell,
      timers
    );

    std::string iomanager_id = configMap.getValue<std::string>("output", "backend", "IOManager_hdf5");
    this->io_manager = IOManagerFactory::make_instance( iomanager_id,
      configMap,
      m_foreach_cell,
      timers
    );

    std::string iomanager_checkpoint_id = configMap.getValue<std::string>("output", "checkpoint", "IOManager_checkpoint");
    this->io_manager_checkpoint = IOManagerFactory::make_instance( iomanager_checkpoint_id,
      configMap,
      m_foreach_cell,
      timers
    );

    

    std::string compute_dt_id = configMap.getValue<std::string>("dt", "dt_kernel", "Compute_dt_generic");
    this->compute_dt = Compute_dtFactory::make_instance( compute_dt_id,
      configMap,
      m_foreach_cell, 
      timers
    );

    std::string refine_condition_id = configMap.getValue<std::string>("amr", "markers_kernel", "RefineCondition_second_derivative_error");
    this->refine_condition = RefineConditionFactory::make_instance( refine_condition_id,
      configMap,
      m_foreach_cell,
      timers
    );

    // Get initial conditions id
    std::string init_id = configMap.getValue<std::string>("hydro", "problem", "unknown");

    int rank = m_communicator.MPI_Comm_rank();
    if (rank==0) {
      std::cout << "##########################" << "\n";
      std::cout << "Godunov updater    : " << godunov_updater_id << std::endl;
      std::cout << "IO Manager         : " << iomanager_id << std::endl;
      std::cout << "Gravity solver     : " << gravity_solver_id << std::endl;
      std::cout << "Initial conditions : " << init_id << std::endl;
      std::cout << "Refine condition   : " << refine_condition_id << std::endl;
      std::cout << "Compute dt         : " << compute_dt_id << std::endl;
      std::cout << "##########################" << std::endl;
    }

    // Initialize cells
    {
      // test if we are performing a re-start run (default : false)
      bool restartEnabled = configMap.getValue<bool>("run","restart_enabled", false);

      std::string init_name = restartEnabled ? "restart" : init_id;
      std::unique_ptr<InitialConditions> initial_conditions =
        InitialConditionsFactory::make_instance(init_id, 
          configMap,
          m_foreach_cell,
          timers);

      initial_conditions->init( U );
    }
 

    std::ofstream out_ini("last.ini" );
    configMap.output( out_ini );       
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
      m_communicator.MPI_Allreduce(&interrupted, &any_interrupted, 1, MpiComm::MPI_Op_t::LOR);
      finished = ( m_t_end > 0    && m_t >= (m_t_end - 1e-14) ) // End if physical time exceeds end time
              || ( m_iter_end > 0 && m_iter >= m_iter_end    )  // Or if iter count exceeds maximum iter count
              || any_interrupted;
    }
    signal( SIGINT, SIG_DFL );

    // Always output after last iteration
    timers.get("outputs").start();
    if( m_enable_output )
      io_manager->save_snapshot(U, m_iter, m_t);
    timers.get("outputs").stop();
    timers.get("checkpoint").start();
    if( m_enable_checkpoint )
      io_manager_checkpoint->save_snapshot(U, m_iter, m_t);
    timers.get("checkpoint").stop();

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
      bool first_iter = (m_iter == m_iter_start);
      // Output at fixed frequency set by 'm_output_frequency'
      bool frequency_trigger = ( m_output_frequency > 0 && m_iter % m_output_frequency == 0 );
      // Output at fixed physical time interval 'm_output_timeslice'
      bool timeslice_trigger = ( m_output_timeslice > 0 && (m_t - m_output_timeslice*m_output_timeslice_count) >= m_output_timeslice );
      if( timeslice_trigger )
        m_output_timeslice_count++;        

      if( m_enable_output && (first_iter || frequency_trigger || timeslice_trigger) )
      {
        int rank = m_communicator.MPI_Comm_rank();
        if( rank == 0 )
        {
          std::cout << "Output results at time t=" << m_t << " step " << m_iter << std::endl;
        }
        io_manager->save_snapshot(U, m_iter, m_t);
      }
      timers.get("outputs").stop();
    }

    // Write checkpoint
    {
      timers.get("checkpoint").start();
      // Output at fixed frequency set by 'm_output_frequency'
      bool frequency_trigger = ( m_checkpoint_frequency > 0 && m_iter % m_checkpoint_frequency == 0 );
      // Output at fixed physical time interval 'm_output_timeslice'
      bool timeslice_trigger = ( m_checkpoint_timeslice > 0 && (m_t - m_checkpoint_timeslice*m_checkpoint_timeslice_count) >= m_checkpoint_timeslice );
      if( timeslice_trigger )
        m_checkpoint_timeslice_count++;        

      if( m_enable_checkpoint && (frequency_trigger || timeslice_trigger) )
      {
        int rank = m_communicator.MPI_Comm_rank();
        if( rank == 0 )
        {
          std::cout << "Checkpoint at time t=" << m_t << " step " << m_iter << std::endl;
        }
        io_manager_checkpoint->save_snapshot(U, m_iter, m_t);
      }
      timers.get("checkpoint").stop();
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

    U.new_fields({"rho_next", "e_tot_next", "rho_vx_next", "rho_vy_next", "rho_vz_next"});
    
    // TODO automatic new fields according to kernel
    if( this->has_mhd )
      U.new_fields({"Bx_next", "By_next", "Bz_next"});
    
    if (m_gravity_type & GRAVITY_FIELD) {
      U.new_fields({"gx", "gy", "gz"});
      if( !U.has_field("gphi") )
        U.new_fields({"gphi"});
    }

    // Update gravity
    if( gravity_solver )
      gravity_solver->update_gravity_field(U);

    // Update hydro
    godunov_updater->update( U, dt );

    m_t += dt;

    U.move_field( "rho", "rho_next" ); 
    U.move_field( "e_tot", "e_tot_next" ); 
    U.move_field( "rho_vx", "rho_vx_next" ); 
    U.move_field( "rho_vy", "rho_vy_next" ); 
    U.move_field( "rho_vz", "rho_vz_next" );
    if( this->has_mhd )
    {
      U.move_field( "Bx", "Bx_next" ); 
      U.move_field( "By", "By_next" ); 
      U.move_field( "Bz", "Bz_next" );
    }
    
    if (m_gravity_type & GRAVITY_FIELD)
    {
      U.delete_field("gx");
      U.delete_field("gy");
      U.delete_field("gz");
    }    

    // AMR cycle
    {
      if( m_amr_cycle_frequency > 0 && ( (m_iter % m_amr_cycle_frequency) == 0 ) )
      {
        timers.get("AMR").start();

        timers.get("MPI ghosts").start();
        U.exchange_ghosts( ghost_comm );
        timers.get("MPI ghosts").stop();

        timers.get("AMR: Mark cells").start();
        refine_condition->mark_cells( U );
        timers.get("AMR: Mark cells").stop();

        // Backup old mesh
        mapUserData->save_old_mesh();

        timers.get("AMR: adapt").start();
        // 1. adapt mesh with mapper enabled
        m_amr_mesh->adapt(true);
        // Verify that adapt() doesn't need another iteration
        assert(m_amr_mesh->check21Balance());
        assert(!m_amr_mesh->checkToAdapt());        
        timers.get("AMR: adapt").stop();

        // Resize and fill U with copied/interpolated/extrapolated data
        timers.get("AMR: remap userdata").start();
        
        //TODO
        //std::cout << "Resize U after remap : " << DataArrayBlock::required_allocation_size(U2.U.extent(0), U2.U.extent(1), U2.U.extent(2)) * (2/1e6) 
        //    << " -> " << DataArrayBlock::required_allocation_size(U2.U.extent(0), U2.U.extent(1), m_amr_mesh->getNumOctants()) * (2/1e6) << " MBytes" << std::endl;

        mapUserData->remap( U );

        timers.get("AMR: remap userdata").stop();

        timers.get("AMR").stop();
      }
    }

    //Load Balancing
    {
      if( m_loadbalance_frequency > 0 && ( (m_iter % m_loadbalance_frequency) == 0 ) )
      {
        timers.get("AMR: load-balance").start();

        m_amr_mesh->loadBalance_userdata(m_loadbalance_coherent_levels, U);

        timers.get("AMR: load-balance").stop();
      }
    }    
  }

private:
  // Simulation parameters
  int m_iter_end; //! Number of iterations before ending the simulation 
  real_t m_t_end; //! Physical time to end the simulation
  int m_nlog; //! Timestep log frequency  
  bool m_enable_output; //! Enable vizualization output and output at least at beginning and end of simulation
  int m_output_frequency; //! Maximum number of iterations between outputs (does nothing if m_enable_output==false)
  double m_output_timeslice; //! Physical time interval between outputs (does nothing if m_enable_output==false)
  bool m_enable_checkpoint; //! Enable checkpoint output and output at least at beginning and end of simulation
  int m_checkpoint_frequency; //! Maximum number of iterations between checkpoints (does nothing if m_enable_output==false)
  double m_checkpoint_timeslice; //! Physical time interval between checkpoints (does nothing if m_enable_output==false)
  
  int m_amr_cycle_frequency; //! Number of iterations between amr cycles (<= 0 is never)
  int m_loadbalance_frequency; //! Number of iterations between load balancing (<= 0 is never)
  GravityType m_gravity_type;
  
  int m_iter_start; //! First iteration (for restart)
  int m_iter; //! Current Iteration number
  real_t m_t; //! Current physical time
  int m_output_timeslice_count; //! Number of timeslices already written
  int m_checkpoint_timeslice_count; //! Number of timeslices already written

  MpiComm m_communicator;
  std::shared_ptr<AMRmesh> m_amr_mesh;
  ForeachCell m_foreach_cell;
  UserData U;
  int m_loadbalance_coherent_levels;

  using CellArray = ForeachCell::CellArray_global_ghosted;

  std::unique_ptr<Compute_dt> compute_dt;
  std::unique_ptr<RefineCondition> refine_condition;
  std::unique_ptr<HydroUpdate> godunov_updater;
  bool has_mhd; // TODO : remove this
  std::unique_ptr<MapUserData> mapUserData;
  std::unique_ptr<IOManager> io_manager, io_manager_checkpoint;
  std::unique_ptr<GravitySolver> gravity_solver;

  Timers timers;
};

int DyabloTimeLoop::interrupted = false;


} // namespace dyablo
