#include "amr/MapUserData_base.h"

#include "amr/CellIndexRemapper.h"

namespace dyablo {

/**
 * Implementation of MapUserData using mean of smaller cells when coarseneing
 **/
class MapUserData_mean : public MapUserData{
public: 
  MapUserData_mean(
                ConfigMap& configMap,
                ForeachCell& foreach_cell,
                Timers& timers )
    : foreach_cell(foreach_cell)
  {}
  
  ~MapUserData_mean(){}

  void save_old_mesh() override
  {
    this->lmesh_old = this->foreach_cell.get_amr_mesh().getLightOctree();
  }

  void remap( UserData& U ) override
  {
    using CellIndex = ForeachCell::CellIndex;
    int ndim = foreach_cell.getDim();
    int nbfields = U.nbFields();

    auto original_fields = U.getEnabledFields();

    std::vector< UserData::FieldAccessor_FieldInfo > old_fields, new_fields;
    for( const std::string& name : original_fields )
    {
      std::string name_new = name + "_remapped";
      U.new_fields({name_new});
      
      VarIndex last_index = (VarIndex)old_fields.size();
      old_fields.push_back( {name, last_index} );
      new_fields.push_back( {name+"_remapped", last_index} );
    }

    const UserData::FieldAccessor Uin = U.getAccessor( old_fields );
    UserData::FieldAccessor Uout = U.getAccessor( new_fields );

    CellIndexRemapper remapper( this->lmesh_old, this->foreach_cell );

    foreach_cell.foreach_cell( "MapUserData_mean::remap", Uout.getShape(),
      KOKKOS_LAMBDA( const CellIndex& iCell_Uout )
    {
      CellIndex iCell_Uin = remapper.get_old_cell( iCell_Uout );

      if( iCell_Uin.level_diff() >= 0 )
      {
        for(int ivar=0; ivar<nbfields; ivar++)
          Uout.at_ivar( iCell_Uout, ivar ) = Uin.at_ivar( iCell_Uin, ivar );
      }
      else
      {
        for(int ivar=0; ivar<nbfields; ivar++)
          Uout.at_ivar( iCell_Uout, ivar ) = 0;

        int nsubcells = (ndim-1) * 2 * 2;
        for(int8_t dz=0; dz<(ndim-1); dz++)
          for(int8_t dy=0; dy<2; dy++)
            for(int8_t dx=0; dx<2; dx++)
            {
              CellIndex iCell_Uin_n = iCell_Uin.getNeighbor_ghost({dx,dy,dz}, Uin.getShape());
              for(int ivar=0; ivar<nbfields; ivar++)
                Uout.at_ivar( iCell_Uout, ivar ) += Uin.at_ivar( iCell_Uin_n, ivar ) / nsubcells;
            }
      }
    });

    for( const std::string& name : original_fields )
    {
      U.move_field( name, name+"_remapped" );
    }
  }

private:
  ForeachCell& foreach_cell;
  LightOctree lmesh_old;
};

} // namespace dyablo;

FACTORY_REGISTER( dyablo::MapUserDataFactory , dyablo::MapUserData_mean, "MapUserData_mean")
