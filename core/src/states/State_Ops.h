#pragma once

#include "real_type.h"
#include "kokkos_shared.h"

namespace dyablo {

template<typename T>
struct State_traits
{
    static constexpr bool is_state = false;
};

#define DECLARE_STATE_TYPE_AUX( constness, State, N ) \
template<> \
struct State_traits<constness State> \
{ \
    static constexpr bool is_state = true; \
    static constexpr int nvars = N; \
    template< int I >  \
    KOKKOS_INLINE_FUNCTION \
    static constness real_t& get(constness State& s) \
    { \
        static_assert(I!=I, "Missing variable in state `" #State "`, add it with DECLARE_STATE_GET()"); \
        static real_t res = 0; \
        return res; \
    } \
}; \

#define DECLARE_STATE_TYPE( State, N ) \
DECLARE_STATE_TYPE_AUX( , State, N ); \
DECLARE_STATE_TYPE_AUX( const , State, N ); \

#define DECLARE_STATE_GET_AUX( State, I, expr ) \
template<> KOKKOS_INLINE_FUNCTION real_t& State_traits<State>::get<I>( State& s ) { return expr; } \
template<> KOKKOS_INLINE_FUNCTION const real_t& State_traits<const State>::get<I>( const State& s ) { return expr; } \

#define DECLARE_STATE_GET( State, I, var ) \
DECLARE_STATE_GET_AUX( State, I, s.var )

template< int I, typename State_t >
KOKKOS_INLINE_FUNCTION
real_t& state_get( State_t& s )
{
    return State_traits<State_t>::template get<I>(s);
}

template< int I, typename State_t >
KOKKOS_INLINE_FUNCTION
const real_t& state_get( const State_t& s )
{
    return State_traits<const State_t>::template get<I>(s);
}

template< int I=0, typename F, typename... State_t >
KOKKOS_INLINE_FUNCTION
void state_foreach_var( const F& f, State_t&... states )
{
    f( state_get<I>(states)... );
    if constexpr( ( (I+1 < State_traits<State_t>::nvars) && ...) )
        state_foreach_var<I+1>(f, states...);
}

// Operators on states
// Operator +
template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator+(const State_t& lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [](real_t& res, real_t l, real_t r){res=l+r;}, res, lhs, rhs );
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator+(const State_t& lhs, real_t rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t l){res=l+rhs;}, res, lhs );    
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator+(real_t lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t r){res=lhs+r;}, res, rhs );    
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t& operator+=(State_t &lhs, const State_t& rhs) {
    state_foreach_var( [&](real_t& l, real_t r){l+=r;}, lhs, rhs );
    return lhs;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t& operator+=(State_t &lhs, real_t rhs) {
    state_foreach_var( [&](real_t& l){l+=rhs;}, lhs );
    return lhs;
}

// Operator -
template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator-(const State_t& lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [](real_t& res, real_t l, real_t r){res=l-r;}, res, lhs, rhs );
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator-(const State_t& lhs, real_t rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t l){res=l-rhs;}, res, lhs );    
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator-(real_t lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t r){res=lhs-r;}, res, rhs );    
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t& operator-=(State_t &lhs, const State_t& rhs) {
    state_foreach_var( [&](real_t& l, real_t r){l-=r;}, lhs, rhs );
    return lhs;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t& operator-=(State_t &lhs, real_t rhs) {
    state_foreach_var( [&](real_t& l){l-=rhs;}, lhs );
    return lhs;
}

// Operator *
template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator*(const State_t& lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [](real_t& res, real_t l, real_t r){res=l*r;}, res, lhs, rhs );
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator*(const State_t& lhs, real_t rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t l){res=l*rhs;}, res, lhs );    
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator*(real_t lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t r){res=lhs*r;}, res, rhs );    
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t& operator*=(State_t &lhs, const State_t& rhs) {
    state_foreach_var( [&](real_t& l, real_t r){l*=r;}, lhs, rhs );
    return lhs;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t& operator*=(State_t &lhs, real_t rhs) {
    state_foreach_var( [&](real_t& l){l*=rhs;}, lhs );
    return lhs;
}

// Operator /
template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator/(const State_t& lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [](real_t& res, real_t l, real_t r){res=l/r;}, res, lhs, rhs );
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator/(const State_t& lhs, real_t rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t l){res=l/rhs;}, res, lhs );    
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t operator/(real_t lhs, const State_t& rhs)
{
    State_t res;
    state_foreach_var( [&](real_t& res, real_t r){res=lhs/r;}, res, rhs );    
    return res;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t& operator/=(State_t &lhs, const State_t& rhs) {
    state_foreach_var( [&](real_t& l, real_t r){l/=r;}, lhs, rhs );
    return lhs;
}

template<   typename State_t,
            std::enable_if_t< State_traits<State_t>::is_state, bool> = false > 
KOKKOS_INLINE_FUNCTION
State_t& operator/=(State_t &lhs, real_t rhs) {
    state_foreach_var( [&](real_t& l){l/=rhs;}, lhs );
    return lhs;
}


}