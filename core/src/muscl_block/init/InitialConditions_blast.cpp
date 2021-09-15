#include "InitialConditions_analytical.h"

namespace dyablo{

struct AnalyticalFormula_blast{
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
    
    const real_t gamma0;
    
    AnalyticalFormula_blast( const ConfigMap& configMap, const HydroParams& params ) :
        ndim( (params.dimType == THREE_D) ? 3 : 2 ),
        blast_radius ( configMap.getFloat("blast","radius", 0.1) ),
        blast_center_x ( configMap.getFloat("blast","center_x", 0.5) ),
        blast_center_y ( configMap.getFloat("blast","center_y", 0.5) ),
        blast_center_z ( configMap.getFloat("blast","center_z", 0.5) ),
        blast_density_in ( configMap.getFloat("blast","density_in", 1.0) ),
        blast_density_out ( configMap.getFloat("blast","density_out", 1.2) ),
        blast_pressure_in ( configMap.getFloat("blast","pressure_in", 10.0) ),
        blast_pressure_out ( configMap.getFloat("blast","pressure_out", 0.1) ),
        blast_nx ( configMap.getInteger("blast", "blast_nx", 1) ),
        blast_ny ( configMap.getInteger("blast", "blast_ny", 1) ),
        blast_nz ( configMap.getInteger("blast", "blast_nz", 1) ),
        gamma0 ( params.settings.gamma0 )

    {}

    KOKKOS_INLINE_FUNCTION
    bool need_refine( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
    {
        // Quadrant size
        const real_t qx = 1.0 / this->blast_nx;
        const real_t qy = 1.0 / this->blast_ny;
        const real_t qz = (this->ndim == 3 ? 1.0 / this->blast_nz : 1.0);
        const real_t q = std::min(qx, std::min( qy, qz ) );

        const int qix = (int)(x / qx);
        const int qiy = (int)(y / qy);
        const int qiz = (int)(z / qz);

        // Rescaling position wrt the current blast quadrant
        x = (x - qix * qx) / q - (qx/q - 1) * 0.5;
        y = (y - qiy * qy) / q - (qy/q - 1) * 0.5;
        z = (z - qiz * qz) / q - (qz/q - 1) * 0.5;

        // Two refinement criteria are used : 
        //  1- If the cell size is larger than a quadrant we refine
        //  2- If the distance to the blast is smaller than the size of
        //     half a diagonal we refine
        
        bool should_refine = dx > qx || dy > qy || dz > qz;

        real_t d2 = std::pow(x - blast_center_x, 2) +
                    std::pow(y - blast_center_y, 2);
        if (this->ndim == 3)
            d2 += std::pow(z - blast_center_z, 2);

        // Cell diag is calculated to be in the units of a quadrant
        const real_t cx = dx / qx;
        const real_t cy = dx / qy;
        const real_t cz = dx / qz;

        real_t cellDiag = (this->ndim == 3)
                            ? sqrt(cx*cx+cy*cy+cz*cz) * 0.5
                            : sqrt(cx*cx+cy*cy) * 0.5;

        if (fabs(sqrt(d2) - this->blast_radius) < cellDiag)
            should_refine = true;

        return should_refine;
    } 

    KOKKOS_INLINE_FUNCTION
    HydroState3d value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz ) const
    {
        // Quadrant size
        const real_t qx = 1.0 / this->blast_nx;
        const real_t qy = 1.0 / this->blast_ny;
        const real_t qz = 1.0 / this->blast_nz;
        const real_t q = std::min(qx, std::min(qy, qz));
        
        const int qix = (int)(x / qx);
        const int qiy = (int)(y / qy);
        const int qiz = (int)(z / qz);

        real_t radius2 = blast_radius*blast_radius;

        // Rescaling position wrt the current blast quadrant
        x = (x - qix * qx) / q - (qx/q - 1) * 0.5;
        y = (y - qiy * qy) / q - (qy/q - 1) * 0.5;
        z = (z - qiz * qz) / q - (qz/q - 1) * 0.5;

        // initialize
        real_t d2 = 
        (x-this->blast_center_x)*(x-this->blast_center_x)+
        (y-this->blast_center_y)*(y-this->blast_center_y);  
        
        if (this->ndim==3)
            d2 += (z-blast_center_z)*(z-blast_center_z);
        
        HydroState3d res {};

        if (d2 < radius2) {
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