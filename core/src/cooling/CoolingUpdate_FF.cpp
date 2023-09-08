#include "CoolingUpdate_base.h"

namespace dyablo{

namespace {
  enum VarIndex_Cooling {IE};

  enum CoolingLayer {ONE_CELL, LAYER};
}

/**
 * @brief Flux-Flux cooling term. 
 * This plugin will cool the domain at the top (zmin in 3D, ymin in 2D)
 * and heat the domain at the bottom (zmax/ymax). The heating will be
 * exactly compensated by the cooling so that no total energy is injected
 * in the box.
 */
class CoolingUpdate_FF : public CoolingUpdate
{
private:
  ForeachCell& foreach_cell;
  Timers& timers;

  CoolingLayer cooling_mode;
  real_t flux;

  real_t dmin, dmax;
public:
  CoolingUpdate_FF(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers )
  :  foreach_cell(foreach_cell),
     timers(timers),
     flux(configMap.getValue<real_t>("cooling", "FF_fluxes", 0.0))
  { 
    std::string cooling_mode = configMap.getValue<std::string>("cooling", "FF_layer", "one_cell");
    if (cooling_mode == "one_cell")
      this->cooling_mode = ONE_CELL;
    else if (cooling_mode == "layer")
      this->cooling_mode = LAYER;
    else {
      std::ostringstream err_msg;
      err_msg << "Unknown value for cooling/FF_layer : " << cooling_mode 
              << "; Available are : layer, one_cell" << std::endl;

      throw std::runtime_error(err_msg.str());
    }

    int ndim = configMap.getValue<int>("mesh", "ndim", 3);
    if (ndim == 2) {
      dmin = configMap.getValue<real_t>("mesh", "ymin", 0.0);
      dmax = configMap.getValue<real_t>("mesh", "ymax", 1.0);
    }
    else {
      dmin = configMap.getValue<real_t>("mesh", "zmin", 0.0);
      dmax = configMap.getValue<real_t>("mesh", "zmax", 1.0);
    }
  }

  ~CoolingUpdate_FF() {}

  template<int ndim>
  void update_aux( UserData &U,
                   real_t dt) {
    ForeachCell& foreach_cell = this->foreach_cell;

    timers.get("Cooling FF").start();

    UserData::FieldAccessor Uin  = U.getAccessor( {{"e_tot", VarIndex_Cooling::IE}} );
    UserData::FieldAccessor Uout = U.getAccessor( {{"e_tot_next", VarIndex_Cooling::IE}} );

    const real_t dmin = this->dmin;
    const real_t dmax = this->dmax;

    CoolingLayer cooling_mode = this->cooling_mode;
    real_t flux = this->flux; 

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();

    foreach_cell.foreach_cell( "Cooling::update", Uout.getShape(), 
      KOKKOS_LAMBDA(const ForeachCell::CellIndex& iCell_Uout) {
  
      auto pos = cellmetadata.getCellCenter(iCell_Uout);
      auto size = cellmetadata.getCellSize(iCell_Uout);

      const real_t d = (ndim == 2 ? pos[IY] : pos[IZ]);
      const real_t dh = (ndim == 2 ? size[IY] : size[IZ]);

      // Positive means heating, negative cooling
      const real_t F = flux / dh;

      real_t heating = 0.0;

      // Only applying on the first/last cell row of the domain
      if (cooling_mode == ONE_CELL) {      
        if (d < dmin + dh)
          heating = -F;
        else if (d > dmax-dh)
          heating = F;
      }
      else {
        // TO(re)DO this ^^;
      }

      Uout.at(iCell_Uout, VarIndex_Cooling::IE) += dt * heating;
    });

    timers.get("Cooling FF").stop();
  }

  void update( UserData &U,
               ScalarSimulationData& scalar_data)
  {
    uint32_t ndim = foreach_cell.getDim();
    real_t dt = scalar_data.get<real_t>("dt");
    if (ndim == 2)
      update_aux<2>(U, dt);
    else
      update_aux<3>(U, dt);
  }
};


} // namespace dyablo

FACTORY_REGISTER( dyablo::CoolingUpdateFactory, 
                  dyablo::CoolingUpdate_FF, 
                  "CoolingUpdate_FF" );