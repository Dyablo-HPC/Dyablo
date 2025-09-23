#include "SourceUpdate_base.h"
#include "foreach_cell/ForeachCell_utils.h"

namespace dyablo{

/**
 * @brief Belousov-Zhabotinsky Reaction equations. 
 * 
 * This plugin solves the Belousov-Zhabotinsky Reaction equations. Since these equations are not
 * hyperbolic, we don't use the traditional hyperbolic solver, and instead solve everything in this
 * kernel.
 * 
 * The equations are the following :
 * \f{eqnarray*}{}
 *   \dfrac{\partial a}{\partial t} - D_a\Delta a &=& \dfrac{1}{\mu}(-qa - ab + fc) \\
 *   \dfrac{\partial b}{\partial t} - D_b\Delta b &=& \dfrac{1}{\epsilon}(aq - ab + b(1-b)) \\
 *   \dfrac{\partial c}{\partial t} - D_c\Delta c &=& b - c
 * \f}
 * 
 * with $a$, $b$, $c$ the variables of the problem and $D_a$, $D_b$, $D_c$, $q$, $f$, $\epsilon$ and $\mu$ fixed constants.
 */
class SourceUpdate_BZ : public SourceUpdate
{
private:
  ForeachCell& foreach_cell;
  Timers& timers;

  real_t Da, Db, Dc, mu, epsilon, q, f;
public:
  SourceUpdate_BZ(
        ConfigMap& configMap,
        ForeachCell& foreach_cell,
        Timers& timers )
  :  foreach_cell(foreach_cell),
     timers(timers),
     Da(configMap.getValue<real_t>("BZ", "Da", 2.5e-3)),
     Db(configMap.getValue<real_t>("BZ", "Db", 2.5e-3)),
     Dc(configMap.getValue<real_t>("BZ", "Dc", 1.5e-3)),
     mu(configMap.getValue<real_t>("BZ", "mu", 1.0e-5)),
     epsilon(configMap.getValue<real_t>("BZ", "epsilon", 1.0e-2)),
     q(configMap.getValue<real_t>("BZ", "q", 2.0e-3)),
     f(configMap.getValue<real_t>("BZ", "f", 1.6))
  { 
    int ndim = configMap.getValue<int>("mesh", "ndim", 2);
    DYABLO_ASSERT_HOST_RELEASE(ndim == 2, "BZ equations are not implemented for a 3D mesh");
  }

  ~SourceUpdate_BZ() {}

  void update_aux( UserData &U,
                   real_t dt) {

    using PatchArray = ForeachCell::CellArray_patch;
    using CellIndex  = ForeachCell::CellIndex;
    using offset_t   = typename CellIndex::offset_t;

    ForeachCell& foreach_cell = this->foreach_cell;

    timers.get("BZ Update").start();

    enum VarIndex {
      IA, IB, IC
    };

    std::vector<UserData::FieldAccessor_FieldInfo> fields {{"a", VarIndex::IA},
                                                            {"b", VarIndex::IB},
                                                            {"c", VarIndex::IC}};
    UserData::FieldAccessor Uin  = U.getAccessor( fields );
    std::vector<UserData::FieldAccessor_FieldInfo> fields_next;
    for (auto f : fields) {
      std::string fname = f.name + "_next";
      fields_next.push_back({fname, f.id});
    }
    UserData::FieldAccessor Uout = U.getAccessor( fields_next );

    ForeachCell::CellMetaData cellmetadata = foreach_cell.getCellMetaData();
    const real_t nb_ghosts = 2;

    PatchArray::Ref Upatch_ = foreach_cell.reserve_patch_tmp("Upatch", nb_ghosts, nb_ghosts, 0, 3);

    const real_t diff_coeff[3] {Da, Db, Dc};

    const real_t mu = this->mu;
    const real_t epsilon = this->epsilon;
    const real_t q = this->q;
    const real_t f = this->f;

    foreach_cell.foreach_patch( "BZ_Update", 
      PATCH_LAMBDA( const ForeachCell::Patch& patch )
    {
      PatchArray Upatch = patch.allocate_tmp(Upatch_);

      // Filling patches
      patch.foreach_cell( Upatch,
        CELL_LAMBDA( const CellIndex &iCell_Upatch )
      {
        auto iCell_Uin = Uin.getShape().convert_index_ghost(iCell_Upatch);
        int level_diff = iCell_Uin.level_diff();
        real_t a, b, c;
        a = 0.0;
        b = 0.0;
        c = 0.0;

        if (iCell_Uin.is_boundary()) { // Absorbing BCs
          CellIndex iCell_inside;
          offset_t offset;
          iCell_Uin.getBoundaryPosAndOffset(iCell_inside, offset);
          auto sign = [](int x){return (x>0)-(x<0);};
        
          CellIndex::offset_t symmetric_offset {
            (int16_t)(-offset[IX] + sign(offset[IX])), 
            (int16_t)(-offset[IY] + sign(offset[IY])), 
            (int16_t)(-offset[IZ] + sign(offset[IZ]))
          }; 
          CellIndex iCell_sym = iCell_inside.getNeighbor(symmetric_offset);

          a = Uin.at(iCell_sym, IA);
          b = Uin.at(iCell_sym, IB);
          c = Uin.at(iCell_sym, IC);
        }
        else if (level_diff < 0) {
          
          int subcell_count = 
          foreach_sibling(2, iCell_Uin, Uin.getShape(),
            [&](const CellIndex& iCell_neigh) {
              a += Uin.at(iCell_neigh, IA);
              b += Uin.at(iCell_neigh, IB);
              c += Uin.at(iCell_neigh, IC);
          });

          a /= subcell_count;
          b /= subcell_count;
          c /= subcell_count;
        }
        else {
          a = Uin.at(iCell_Uin, IA);
          b = Uin.at(iCell_Uin, IB);
          c = Uin.at(iCell_Uin, IC);          
        }

        Upatch.at(iCell_Upatch, IA) = a;
        Upatch.at(iCell_Upatch, IB) = b;
        Upatch.at(iCell_Upatch, IC) = c;
      });

      // Diffusion operation
      patch.foreach_cell( Uout.getShape(),
        CELL_LAMBDA( const CellIndex &iCell_Uout )
      {
        CellIndex iCell_Upatch = Upatch.getShape().convert_index(iCell_Uout);

        for (auto var: {IA, IB, IC}) {
          const real_t vij   = Upatch.at(iCell_Upatch, var);
          const real_t vimj  = Upatch.at(iCell_Upatch + offset_t{-1, 0, 0}, var);
          const real_t vimmj = Upatch.at(iCell_Upatch + offset_t{-2, 0, 0}, var);
          const real_t vipj  = Upatch.at(iCell_Upatch + offset_t{ 1, 0, 0}, var);
          const real_t vippj = Upatch.at(iCell_Upatch + offset_t{ 2, 0, 0}, var);
          const real_t vijm  = Upatch.at(iCell_Upatch + offset_t{ 0,-1, 0}, var);
          const real_t vijmm = Upatch.at(iCell_Upatch + offset_t{ 0,-2, 0}, var);
          const real_t vijp  = Upatch.at(iCell_Upatch + offset_t{ 0, 1, 0}, var);
          const real_t vijpp = Upatch.at(iCell_Upatch + offset_t{ 0, 2, 0}, var);
          auto dh = cellmetadata.getCellSize(iCell_Uout);

          const real_t Dvx = diff_coeff[var] * ((-vimmj - vippj)/12.0 + (vimj + vipj) * 4.0/3.0 - 5.0/2.0 * vij) / (dh[IX]*dh[IX]);
          const real_t Dvy = diff_coeff[var] * ((-vijmm - vijpp)/12.0 + (vijm + vijp) * 4.0/3.0 - 5.0/2.0 * vij) / (dh[IY]*dh[IY]);
          Uout.at(iCell_Uout, var) = Uin.at(iCell_Uout, var) + dt * (Dvx + Dvy);
        }
      });

      // Source term
      patch.foreach_cell( Uout.getShape(),
        CELL_LAMBDA( const CellIndex &iCell_Uout)
      {
        CellIndex iCell_Upatch = Upatch.getShape().convert_index(iCell_Uout);
        const real_t a = Upatch.at(iCell_Upatch, IA);
        const real_t b = Upatch.at(iCell_Upatch, IB);
        const real_t c = Upatch.at(iCell_Upatch, IC);

        const real_t sa = (-q*a - a*b + f*c) / mu;
        const real_t sb = ( q*a - a*b + b*(1.0-b)) / epsilon;
        const real_t sc = b - c;

        Uout.at(iCell_Uout, IA) += dt * sa;
        Uout.at(iCell_Uout, IB) += dt * sb;
        Uout.at(iCell_Uout, IC) += dt * sc;
      });
    });

    timers.get("BZ Update").stop();
  }

  void update( UserData &U,
               ScalarSimulationData& scalar_data)
  {
    real_t dt = scalar_data.get<real_t>("dt");
    update_aux(U, dt);
  }
};


} // namespace dyablo

FACTORY_REGISTER( dyablo::SourceUpdateFactory, 
                  dyablo::SourceUpdate_BZ, 
                  "SourceUpdate_BZ" );