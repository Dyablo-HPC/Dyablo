#include "MarkOctantsHydroFunctor.h"

#include "Kokkos_Atomic.hpp"

namespace dyablo { 
namespace muscl_block {

namespace{

constexpr real_t eps = std::numeric_limits<real_t>::epsilon();

KOKKOS_INLINE_FUNCTION
real_t second_derivative_error(
          const DataArrayBlock& Qgroup,
          int  ivar, 
          uint32_t iCell,
          uint32_t iCell_delta,
          uint32_t iOct_local)
{
  constexpr real_t epsref = 0.01;

  uint32_t iCellm1 = iCell + iCell_delta;
  uint32_t iCellp1 = iCell - iCell_delta;

  const real_t q   = Qgroup(iCell  ,ivar,iOct_local);
  const real_t qm1 = Qgroup(iCellm1,ivar,iOct_local);
  const real_t qp1 = Qgroup(iCellp1,ivar,iOct_local);

  const real_t fr = qp1 - q;    
  const real_t fl = qm1 - q;
  
  const real_t fc = FABS(qp1) + FABS(qm1) + 2 * FABS(q);
  real_t res = FABS(fr + fl) / (FABS(fr) + FABS(fl) + epsref * fc + eps);
    
  return res;

} // second_derivative_error

} //namespace

MarkOctantsHydroFunctor::markers_t::markers_t(uint32_t capacity)
: iOcts("markers_t::iOcts", capacity),
  markers("markers_t::markers", capacity),
  count("markers_t::count")
{}

KOKKOS_INLINE_FUNCTION void MarkOctantsHydroFunctor::markers_t::push_back(uint32_t iOct, int marker) const
{
  uint32_t i = Kokkos::atomic_fetch_add(&count(),(uint32_t)1);

  this->iOcts(i) = iOct;
  this->markers(i) = marker;
}

template< typename T >
KOKKOS_INLINE_FUNCTION T min(const T& a, const T& b) 
{ return a<b?a:b; }

void MarkOctantsHydroFunctor::apply(  
                    LightOctree    lmesh,
		                ConfigMap      configMap,
                    HydroParams    params,
		                id2index_t     fm,
                    blockSize_t    blockSizes,
                    uint32_t       ghostWidth,
                    uint32_t       nbOcts,
                    uint32_t       nbOctsPerGroup,
                    DataArrayBlock Qgroup,
                    uint32_t       iGroup,
                    real_t         error_min,
                    real_t         error_max,
                    markers_t      markers )
{
  const bool three_d = (params.dimType == THREE_D);

  const int level_max = params.level_max;
  const int level_min = params.level_min;
  const uint32_t bx = blockSizes[IX];
  const uint32_t by = blockSizes[IY];
  const uint32_t bz = three_d?(blockSizes[IZ]):1;
  const uint32_t bx_g = bx + 2*ghostWidth;
  const uint32_t by_g = by + 2*ghostWidth;
  //const uint32_t bz_g = three_d?(bz + 2*ghostWidth):1;
  const uint32_t nbCellsPerBlock = bx*by*bz;

  constexpr int nrefvar = 2;
  Kokkos::Array<int, nrefvar> ref_var = {fm[ID], fm[IP]};  

  using team_policy_t = Kokkos::TeamPolicy<Kokkos::IndexType<int32_t>>;
  using thread_t = team_policy_t::member_type;

  uint32_t nbTeams = configMap.getInteger("amr","nbTeams",16);

  Kokkos::parallel_for( "dyablo::muscl_block::MarkOctantsHydroFunctor",
                        team_policy_t(nbTeams, Kokkos::AUTO()), 
                        KOKKOS_LAMBDA(const thread_t& member)
  {
    uint32_t nbOcts_local = min( nbOctsPerGroup, nbOcts-iGroup*nbOctsPerGroup );
    for( uint32_t iOct_local = member.league_rank();
         iOct_local < nbOcts_local;
         iOct_local += nbTeams)
    {
      real_t error = 0;

      Kokkos::parallel_reduce(
          Kokkos::TeamVectorRange(member, nbCellsPerBlock),
          [=](const int32_t iCellInner, real_t &local_error) 
      {
        //index = i + bx * j + bx * by * k
        const uint32_t k = iCellInner / (bx*by);
        const uint32_t j = (iCellInner - k*bx*by) / bx;
        const uint32_t i = iCellInner - k*bx*by - j*bx;

        uint32_t iCell = i+ghostWidth + bx_g * (j+ghostWidth) + bx_g*by_g*(k+ghostWidth);

        for (int ivar = 0; ivar < nrefvar; ++ivar)
        {
          real_t fx, fy, fmax;

          fx = second_derivative_error(Qgroup,ref_var[ivar],iCell,1,iOct_local);
          fy = second_derivative_error(Qgroup,ref_var[ivar],iCell,bx_g,iOct_local);
          fmax = fx > fy ? fx : fy;

          if( three_d )
          {
            real_t fz;
            fz = second_derivative_error(Qgroup,ref_var[ivar],iCell,bx_g*by_g,iOct_local);
            fmax = fmax > fz ? fmax : fz;
          }
          
          local_error = local_error > fmax ? local_error : fmax;
        }
      },
      Kokkos::Max<real_t>(error)); // end TeamVectorRange

      Kokkos::single( Kokkos::PerTeam(member), 
                      [=]()
      {
        uint32_t iOct = iOct_local + iGroup*nbOctsPerGroup;

        // get current cell level
        uint8_t level = lmesh.getLevel({iOct,false});

        // -1 means coarsen
        //  0 means don't modify
        // +1 means refine
        int criterion = -1;

        if (error > error_min)
          criterion = criterion < 0 ? 0 : criterion;

        if (error > error_max)
          criterion = criterion < 1 ? 1 : criterion;

        if (level < level_max and criterion == 1)
          markers.push_back(iOct,1);
        else if (level > level_min and criterion == -1)
          markers.push_back(iOct,-1);
      });
    }
  });
}

uint32_t MarkOctantsHydroFunctor::markers_t::size()
{
  auto host_count = Kokkos::create_mirror_view( count );
  Kokkos::deep_copy(host_count,count);
  return host_count();
}

Kokkos::View<uint32_t*>::HostMirror MarkOctantsHydroFunctor::markers_t::getiOcts_host()
{
  uint32_t size = this->size();
  auto subview = Kokkos::subview(this->iOcts, std::make_pair((uint32_t)0,size));
  auto res = Kokkos::create_mirror_view( subview );
  Kokkos::deep_copy(res, subview);
  return res;
}
Kokkos::View<int*>::HostMirror MarkOctantsHydroFunctor::markers_t::getMarkers_host()
{
  uint32_t size = this->size();
  auto subview = Kokkos::subview(this->markers, std::make_pair((uint32_t)0,size));
  auto res = Kokkos::create_mirror_view( subview );
  Kokkos::deep_copy(res, subview);
  return res;
}

void MarkOctantsHydroFunctor::set_markers_pablo(markers_t markers, std::shared_ptr<AMRmesh> pmesh)
{
  auto markers_iOcts = markers.getiOcts_host();
  auto markers_markers = markers.getMarkers_host();

  Kokkos::parallel_for( "MarkOctantsHydroFunctor::set_markers_pablo", 
                        Kokkos::RangePolicy<Kokkos::OpenMP>(0,markers.size()),
                        [=](uint32_t i)
  {
    uint32_t iOct = markers_iOcts(i);
    int8_t marker = markers_markers(i);

    pmesh->setMarker(iOct, marker);
  });
}

} // namespace muscl_block
} // namespace dyablo