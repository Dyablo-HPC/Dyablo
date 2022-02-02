#include "InitialConditions_analytical.h"

#include "AnalyticalFormula_tools.h"

namespace dyablo{

struct AnalyticalFormula_blast : public AnalyticalFormula_base{
     // blast problem parameters
    const int ndim;
    const real_t blast_radius;
    const real_t blast_center_x;
    const real_t blast_center_y;
    const real_t blast_center_z;
    const real_t blast_density_in;
    const real_t blast_density_out;
    const real_t blast_pressure_in;
    const real_t blast_pressure_out;
    const int blast_nx;
    const int blast_ny;
    const int blast_nz;
    const real_t error_max;
    const real_t xmin, xmax;
    const real_t ymin, ymax;
    const real_t zmin, zmax;    
    const real_t gamma0, smallr, smallc, smallp;
    
    AnalyticalFormula_blast( ConfigMap& configMap ) :
        ndim( configMap.getValue<int>("mesh", "ndim", 3) ),
        // Length are scaled by quadrant width (0.5,0.5,0.5 is center of quadrant when blast_n* != 1)
        blast_radius ( configMap.getValue<real_t>("blast","radius", 0.1) ),
        blast_center_x ( configMap.getValue<real_t>("blast","center_x", 0.5) ),
        blast_center_y ( configMap.getValue<real_t>("blast","center_y", 0.5) ),
        blast_center_z ( configMap.getValue<real_t>("blast","center_z", 0.5) ),
        blast_density_in ( configMap.getValue<real_t>("blast","density_in", 1.0) ),
        blast_density_out ( configMap.getValue<real_t>("blast","density_out", 1.2) ),
        blast_pressure_in ( configMap.getValue<real_t>("blast","pressure_in", 10.0) ),
        blast_pressure_out ( configMap.getValue<real_t>("blast","pressure_out", 0.1) ),
        // Number of quadrants in each direction
        blast_nx ( configMap.getValue<int>("blast", "blast_nx", 1) ),
        blast_ny ( configMap.getValue<int>("blast", "blast_ny", 1) ),
        blast_nz ( configMap.getValue<int>("blast", "blast_nz", 1) ),  
        error_max(configMap.getValue<real_t>("amr", "error_max", 0.8)),      
        xmin( configMap.getValue<real_t>("mesh", "xmin", 0.0) ), xmax( configMap.getValue<real_t>("mesh", "xmax", 1.0) ),
        ymin( configMap.getValue<real_t>("mesh", "ymin", 0.0) ), ymax( configMap.getValue<real_t>("mesh", "ymax", 1.0) ),
        zmin( configMap.getValue<real_t>("mesh", "zmin", 0.0) ), zmax( configMap.getValue<real_t>("mesh", "zmax", 1.0) ),
        gamma0 ( configMap.getValue<real_t>("hydro","gamma0", 1.4) ),
        smallr ( configMap.getValue<real_t>("hydro","smallr", 1e-10) ),
        smallc ( configMap.getValue<real_t>("hydro","smallc", 1e-10) ),
        smallp ( smallc*smallc / gamma0 )
    {}

    KOKKOS_INLINE_FUNCTION
    bool need_refine( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
    {
        real_t gamma0 = this->gamma0;
        real_t smallr = this->smallr;
        real_t smallp = this->smallp;
        real_t error_max = this->error_max;
        return AnalyticalFormula_tools::auto_refine( *this, gamma0, smallr, smallp, error_max,
                                                     x, y, z, dx, dy, dz );
    }

    // Geometrical version
    // KOKKOS_INLINE_FUNCTION
    // bool need_refine( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
    // {
    //     // Quadrant size
    //     real_t qsx = 1.0 / this->blast_nx;
    //     real_t qsy = 1.0 / this->blast_ny;
    //     real_t qsz = (this->ndim == 3) ? 1.0 / this->blast_nz : 1.0;
    //     real_t qs = FMIN(qsx, FMIN( qsy, qsz ) );
    //     real_t radius = this->blast_radius*qs;
    //     // Quadrant logical position
    //     int qix = (int)(x / qsx);
    //     int qiy = (int)(y / qsy);
    //     int qiz = (int)(z / qsz);
    //     // Quadrant physical center
    //     real_t qcx = (qix+0.5)*qsx;
    //     real_t qcy = (qiy+0.5)*qsy;
    //     real_t qcz = (qiz+0.5)*qsz;

    //     // Two refinement criteria are used : 
    //     //  1- If the cell size is larger than a quadrant we refine        
    //     bool should_refine = dx > qsx || dy > qsy || dz > qsz;

    //     //  2- If the distance to the interface is smaller than the size of
    //     //     half a diagonal we refine
    //     // Squared distance to quadrant center
    //     real_t r2 = (x-qcx)*(x-qcx) + (y-qcy)*(y-qcy);
    //     if( this->ndim == 3 ) r2 += (z-qcz)*(z-qcz);

    //     real_t half_cell_diag = (this->ndim == 3) ?
    //         sqrt(dx*dx+dy*dy+dz*dz)/2 :
    //         sqrt(dx*dx+dy*dy)/2 ;

    //     if( std::abs( std::sqrt(r2) - radius ) < half_cell_diag )
    //         should_refine = true;

    //     return should_refine;
    // } 

    KOKKOS_INLINE_FUNCTION
    HydroState3d value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
    {
        // Quadrant size
        real_t qsx = 1.0 / this->blast_nx;
        real_t qsy = 1.0 / this->blast_ny;
        real_t qsz = (this->ndim == 3 ? 1.0 / this->blast_nz : 1.0);
        real_t qs = FMIN(qsx, FMIN( qsy, qsz ) );
        real_t radius = this->blast_radius*qs;
        // Quadrant logical position
        int qix = (int)(x / qsx);
        int qiy = (int)(y / qsy);
        int qiz = (int)(z / qsz);
        // Quadrant physical center
        real_t qcx = (qix+0.5)*qsx;
        real_t qcy = (qiy+0.5)*qsy;
        real_t qcz = (qiz+0.5)*qsz;

        real_t r2 = (x-qcx)*(x-qcx) + (y-qcy)*(y-qcy);
        if( this->ndim == 3 ) r2 += (z-qcz)*(z-qcz);
        
        HydroState3d res {};

        if (r2 < radius*radius) {
            res[ID] = blast_density_in;
            res[IP] = blast_pressure_in/(gamma0-1.0);;
        } else {
            res[ID] = blast_density_out;
            res[IP] = blast_pressure_out/(gamma0-1.0);
        }

        return res;
    } 
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::muscl_block::InitialConditionsFactory, dyablo::muscl_block::InitialConditions_analytical<dyablo::AnalyticalFormula_blast>, "blast");