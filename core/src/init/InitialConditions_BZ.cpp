#include "InitialConditions_analytical.h"
#include "InitialConditions_base.h"

namespace dyablo{

/**
 * @brief Implements the Belousov-Zhabotinsky Reaction benchmark
 * 
 * The setup evolves three variables, $a$, $b$, $c$ which are initialized as follows : 
 * 
 * \f{eqnarray*}{
 *  \theta &=& arctan\left(\dfrac{y-0.5}{x-0.5}\right) \\
 *  b      &=& 0.8 \text{ if } \theta < 0.5 \text{ else } q\left(\dfrac{f+1}{f-1}\right) \\
 *  c      &=& q\left(\dfrac{f+1}{f-1} + \dfrac{\theta}{8\pi f}\right) \\
 *  a      &=& \dfrac{f-c}{q+b}
 * \f}
 * 
 * with $f$ and $q$ two constants
 */
class InitialConditions_BZ
{
public:
  enum VarIndex_BZ {
    IA, IB, IC
  };

  struct State {
    real_t a, b, c;
  };

  std::vector<UserData::FieldAccessor::FieldInfo> getFieldsInfo() const
  {
    return {{"a", IA}, {"b", IB}, {"c", IC}};
  }

  KOKKOS_INLINE_FUNCTION
  void setState(const UserData::FieldAccessor &Uout,
                const ForeachCell::CellIndex &iCell, const State &u) const
  {
    Uout.at(iCell, IA) = u.a;
    Uout.at(iCell, IB) = u.b;
    Uout.at(iCell, IC) = u.c;
  }
protected:
    real_t q, f;

public:
    InitialConditions_BZ(ConfigMap& configMap)
    {
        q = configMap.getValue<real_t>("BZ", "q", 2.0e-3);
        f = configMap.getValue<real_t>("BZ", "f", 1.6);
    }

    KOKKOS_INLINE_FUNCTION
    State value( real_t x, real_t y, real_t z, real_t dx, real_t dy, real_t dz) const 
    {
        const real_t xi = x - 0.5;
        const real_t yi = y - 0.5;
        real_t theta = atan(yi/xi);
        if (xi < 0.0)
            theta += M_PI;
        else if (yi < 0.0)
            theta += 2.0*M_PI;
        const real_t qf = q*(f+1)/(f-1);
        const real_t b = (theta < 0.5 ? 0.8 : qf);
        const real_t c = qf + theta / (8.0*M_PI*f);
        const real_t a = (f-c)/(q+b);

        State out {a, b, c};
        return out;
    }
};

} // namespace dyablo

FACTORY_REGISTER(dyablo::InitialConditionsFactory, 
                 dyablo::InitialConditions_analytical<dyablo::InitialConditions_BZ>, 
                 "BZ");
