#include "InitialConditions_analytical.h"

#include "AnalyticalFormula_tools.h"

namespace dyablo{

struct AnalyticalFormula_MHD_blast : public AnalyticalFormula_base{
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
    const real_t blast_B;
    const real_t blast_B_angle;
    const int blast_nx;
    const int blast_ny;
    const int blast_nz;
    const real_t error_max;
    const real_t xmin, xmax;
    const real_t ymin, ymax;
    const real_t zmin, zmax;    
    const real_t gamma0, smallr, smallc, smallp;
    
    AnalyticalFormula_MHD_blast( ConfigMap& configMap ) :
        ndim( configMap.getValue<int>("mesh", "ndim", 3) ),
        // Length are scaled by quadrant width (0.5,0.5,0.5 is center of quadrant when blast_n* != 1)
        blast_radius ( configMap.getValue<real_t>("blast","radius", 0.1) ),
        blast_center_x ( configMap.getValue<real_t>("blast","center_x", 0.5) ),
        blast_center_y ( configMap.getValue<real_t>("blast","center_y", 0.5) ),
        blast_center_z ( configMap.getValue<real_t>("blast","center_z", 0.5) ),
        blast_density_in ( configMap.getValue<real_t>("blast","density_in", 1.0) ),
        blast_density_out ( configMap.getValue<real_t>("blast","density_out", 1.0) ),
        blast_pressure_in ( configMap.getValue<real_t>("blast","pressure_in", 10.0) ),
        blast_pressure_out ( configMap.getValue<real_t>("blast","pressure_out", 0.1) ),
        blast_B ( configMap.getValue<real_t>("blast", "blast_B", 3.54491) ),
        blast_B_angle ( configMap.getValue<real_t>("blast", "blast_alpha", 0.785398) ),
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
        const real_t gamma0 = this->gamma0;
        const real_t smallr = this->smallr;
        const real_t smallp = this->smallp;
        const real_t error_max = this->error_max;
        return AnalyticalFormula_tools::auto_refine( *this, gamma0, smallr, smallp, error_max,
                                                      x, y, z, dx, dy, dz );
    }

    KOKKOS_INLINE_FUNCTION
    ConsMHDState value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
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

        const real_t Bx = blast_B * cos(blast_B_angle);
        const real_t By = blast_B * sin(blast_B_angle);

        real_t r2 = (x-qcx)*(x-qcx) + (y-qcy)*(y-qcy);
        if( this->ndim == 3 ) r2 += (z-qcz)*(z-qcz);
        
        ConsMHDState res{};
        res.Bx = Bx;
        res.By = By;
        res.Bz = 0.0;

        real_t Emag = 0.5 * (Bx*Bx+By*By); 

        if (r2 < radius*radius) {
            res.rho = blast_density_in;
            res.e_tot = Emag + blast_pressure_in/(gamma0-1.0);;
        } else {
            res.rho = blast_density_out;
            res.e_tot = Emag + blast_pressure_out/(gamma0-1.0);
        }


        return res;
    } 
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::AnalyticalFormula_MHD_blast>, 
                 "MHD_blast");