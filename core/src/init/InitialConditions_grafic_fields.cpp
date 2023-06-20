#include "InitialConditions_base.h"

#include "utils/io/FortranBinaryReader.h"

#include "utils/units/Units.h"

namespace dyablo {

class InitialConditions_grafic_fields : public InitialConditions
{
public:
  InitialConditions_grafic_fields(
        ConfigMap& configMap, 
        ForeachCell& foreach_cell,  
        Timers& timers )
  : 
    graficDir( configMap.getValue<std::string>("grafic", "inputDir", "data/IC/") ),
    foreach_cell(foreach_cell),
    xmin( configMap.getValue<real_t>("mesh", "xmin", 0.0) ),
    ymin( configMap.getValue<real_t>("mesh", "ymin", 0.0) ),
    zmin( configMap.getValue<real_t>("mesh", "zmin", 0.0) ),
    omegab(configMap.getValue<real_t>("cosmology", "omegab", 1.0)),
    gamma0(configMap.getValue<real_t>("hydro", "gamma0", 1.4)),
    smallr(configMap.getValue<real_t>("hydro", "smallr", 1e-10)),
    smallc(configMap.getValue<real_t>("hydro", "smallc", 1e-10)),
    smallp(smallc * smallc / gamma0)
  {}

  void init( UserData& U )
  {
    ForeachCell& foreach_cell = this->foreach_cell;
    AMRmesh& pmesh = foreach_cell.get_amr_mesh();

    struct grafic_header
    {
      int32_t nx,ny,nz;
      float dx;
      float xo,yo,zo;
      float astart;
      float om,ov,H0;
    };

    // Read header from ic_deltab
    grafic_header header;
    {
      // Open grafic file ic_deltab
      std::string grafic_filename = this->graficDir + "/ic_deltab";
      std::ifstream grafic_file(grafic_filename , std::ios::in|std::ios::binary );
      DYABLO_ASSERT_HOST_RELEASE(grafic_file, "Could not open grafic file : `" + grafic_filename + "`");
      FortranBinaryReader::read_record( grafic_file, &header, 1 );

      DYABLO_ASSERT_HOST_RELEASE( header.nx == foreach_cell.blockSize()[IX]*pmesh.get_coarse_grid_size()[IX], 
        "grafic mesh size does not match AMR grid size (X)");
      DYABLO_ASSERT_HOST_RELEASE( header.ny == foreach_cell.blockSize()[IY]*pmesh.get_coarse_grid_size()[IY], 
        "grafic mesh size does not match AMR grid size (Y)");
      DYABLO_ASSERT_HOST_RELEASE( header.nz == foreach_cell.blockSize()[IZ]*pmesh.get_coarse_grid_size()[IZ], 
        "grafic mesh size does not match AMR grid size (Z)");
    }

    int32_t nx = header.nx;
    int32_t ny = header.ny;
    int32_t nz = header.nz;

    

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    auto fill_field = [&](const std::string& grafic_file_name, const std::string& dyablo_field_name)
    {
      Kokkos::View< float***, Kokkos::LayoutLeft > grafic_field_device ( 
          grafic_file_name,
          nx, ny, nz
      );

      {
        // Open grafic file
        std::string grafic_filename = this->graficDir + "/" + grafic_file_name;
        std::ifstream grafic_file(grafic_filename , std::ios::in|std::ios::binary );
        DYABLO_ASSERT_HOST_RELEASE(grafic_file, "Could not open grafic file : `" + grafic_filename + "`");
        // Read header from grafic file
        grafic_header header;
        FortranBinaryReader::read_record( grafic_file, &header, 1 );
        // Read array from grafic file
        auto grafic_field_host = Kokkos::create_mirror_view( grafic_field_device );
        for(uint32_t z=0; z<nz; z++)
          FortranBinaryReader::read_record( grafic_file, &grafic_field_host(0,0,z), nx*ny );
        Kokkos::deep_copy( grafic_field_device, grafic_field_host );
      }

      enum VarIndex { Ifield };
      UserData::FieldAccessor U_field = U.getAccessor({{dyablo_field_name, Ifield}});

      foreach_cell.foreach_cell( "InitialConditions_grafic_fields::init_field", U_field.getShape(),
        KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
      {
        auto c = cellmetadata.getCellCenter( iCell );
        auto s = cellmetadata.getCellSize( iCell );
        uint32_t ix = (c[IX] - xmin) / s[IX];
        uint32_t iy = (c[IY] - ymin) / s[IY];
        uint32_t iz = (c[IZ] - zmin) / s[IZ];

        U_field.at( iCell, Ifield ) = grafic_field_device(ix,iy,iz);
      });
    };

    // Sequentially read density and velocities from grafic file
    U.new_fields({"rho","e_tot","rho_vx","rho_vy","rho_vz"});
    fill_field( "ic_deltab", "rho" );
    fill_field( "ic_velbx", "rho_vx" );
    fill_field( "ic_velby", "rho_vy" );
    fill_field( "ic_velbz", "rho_vz" );

    enum VarIndex_hydro {ID, IE, IU, IV, IW};
    UserData::FieldAccessor Uinout = U.getAccessor({
        {"rho", ID},
        {"e_tot", IE},
        {"rho_vx", IU},
        {"rho_vy", IV},
        {"rho_vz", IW}
    });

    using namespace Units;

    // Parameters?
    constexpr real_t YHE = 0.24; // Helium Mass fraction
    constexpr real_t yHE = (YHE/(1.-YHE)/MHE_OVER_MH); // Helium number fraction
    
    real_t gamma0 = this->gamma0;
    real_t omegab = this->omegab;
    real_t omegam = header.om;
    real_t astart = header.astart;

    real_t H0 = header.H0 * (Kilo * meter) / second / (Mega * parsec); // Hubble constant (s-1)
    real_t rhoc = 3. * H0 * H0 / (8. * M_PI * NEWTON_G); // comoving critical density (kg/m3)
    real_t rstar = header.nx * header.dx * (Mega * parsec); // box size in m 
    real_t tstar = 2. / H0 / sqrt(omegam); // sec
    real_t vstar = rstar / tstar; //m/s
    real_t rhostar = rhoc * omegam;
    real_t pstar = rhostar * vstar * vstar;

    real_t cosmo_z = 1. / astart - 1.;
    // let's assign a temperature (Recfast Calibration for Planck(+18?)
    real_t temp = 317.5 * (cosmo_z * cosmo_z) / (151.0 * 151.0);

    foreach_cell.foreach_cell( "InitialConditions_grafic_fields::compute_conservative", Uinout.getShape(),
      KOKKOS_LAMBDA( const ForeachCell::CellIndex& iCell )
    {
      real_t cosmo_density = Uinout.at( iCell, ID );
      real_t cosmo_velx = Uinout.at( iCell, IU );
      real_t cosmo_vely = Uinout.at( iCell, IV );
      real_t cosmo_velz = Uinout.at( iCell, IW );

      real_t u = cosmo_velx * 1e3 * astart / vstar;
      real_t v = cosmo_vely * 1e3 * astart / vstar;
      real_t w = cosmo_velz * 1e3 * astart / vstar;
      real_t u2 = u * u + v * v + w * w;

      real_t rho = (cosmo_density + 1.0) * omegab / omegam;
      real_t rho_u = rho * u;
      real_t rho_v = rho * v;
      real_t rho_w = rho * w;

      // Physical baryon density in kg/m3
      real_t cosmo_rhob = (cosmo_density + 1.0) * omegab * rhoc / (astart * astart * astart);

      // Physical pressure
      real_t cosmo_pressure = (gamma0 - 1.0) * 1.5 * (cosmo_rhob * (1. - YHE) / PROTON_MASS * (1. + yHE)) * KBOLTZ * temp;
      real_t p = fmax( cosmo_pressure/pstar * (astart * astart * astart * astart * astart), smallp );
      real_t e_tot = rho*u2/2.0 + p/(gamma0-1.0);

      Uinout.at( iCell, ID ) = rho;
      Uinout.at( iCell, IE ) = e_tot;
      Uinout.at( iCell, IU ) = rho_u;
      Uinout.at( iCell, IV ) = rho_v;
      Uinout.at( iCell, IW ) = rho_w;
    });
  }

private:
  std::string graficDir;
  ForeachCell& foreach_cell;
  real_t xmin, ymin, zmin;
  real_t omegab;
  real_t gamma0;
  real_t smallr,smallc,smallp;
};

} //namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_grafic_fields, 
                 "grafic_fields");