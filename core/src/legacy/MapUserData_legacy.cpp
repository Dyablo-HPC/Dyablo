#include "amr/MapUserData_base.h"

#include "legacy/MapUserData.h"

namespace dyablo {

class MapUserData_legacy : public MapUserData{
public: 
  MapUserData_legacy(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
    : bx( configMap.getValue<uint32_t>("amr", "bx", 0) ),
      by( configMap.getValue<uint32_t>("amr", "by", 0) ),
      bz( configMap.getValue<uint32_t>("amr", "bz", 1) ),
      foreach_cell(foreach_cell)
  {}
  
  ~MapUserData_legacy(){}

  void save_old_mesh()
  {
    this->lmesh_old = this->foreach_cell.get_amr_mesh().getLightOctree();
  }
  
  void remap(   const ForeachCell::CellArray_global_ghosted& Uin,
                const ForeachCell::CellArray_global_ghosted& Uout )
  {
    const LightOctree& lmesh_new = foreach_cell.get_amr_mesh().getLightOctree();

    MapUserDataFunctor::apply( this->lmesh_old, lmesh_new, blockSize_t{bx,by,bz}, Uin.U, Uin.Ughost, Uout.U );
  }

private:
  uint32_t bx, by, bz;
  ForeachCell& foreach_cell;
  LightOctree lmesh_old;
};

} //namespace dyablo 

FACTORY_REGISTER( dyablo::MapUserDataFactory , dyablo::MapUserData_legacy, "MapUserData_legacy")
