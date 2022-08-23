/**
 * Define arithmetic operations on States. States are structures containing 
 * variables for a cell (or a particle) needed inside kernels. 
 * 
 * A state `State_t` contains N variables of type `real_t`.
 * To enable arithmetic operators +,-,*,/,+=,-=,*=,/= as well as `state_get<I>(state)` and  `state_foreach_var()` :
 * * DECLARE_STATE_TYPE( State_t, N ) must be called to declare State_t as a state
 * * An index for every member variable must be set with DECLARE_STATE_GET( State_t, <index>, <member> )
 *      Every index from 0 to N-1 must be set
 * See State_hydro.h for an example
 **/

#pragma once

#include "real_type.h"
#include "kokkos_shared.h"

namespace dyablo {

/// By default T is not a State
template<typename T>
struct State_traits
{
    static constexpr bool is_state = false;
};

/// Type-trait for States
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

/// DECLARE_STATE_TYPE() declares `State` as a State and State_traits<State>::is_state == true;
#define DECLARE_STATE_TYPE( State, N ) \
DECLARE_STATE_TYPE_AUX( , State, N ); \
DECLARE_STATE_TYPE_AUX( const , State, N ); \

#define DECLARE_STATE_GET_AUX( State, I, expr ) \
template<> KOKKOS_INLINE_FUNCTION real_t& State_traits<State>::get<I>( State& s ) { return expr; } \
template<> KOKKOS_INLINE_FUNCTION const real_t& State_traits<const State>::get<I>( const State& s ) { return expr; } \

/// Associates index I to field s.var (`State` Must be declared as a State with DECLARE_STATE_TYPE() before )
#define DECLARE_STATE_GET( State, I, var ) \
DECLARE_STATE_GET_AUX( State, I, s.var )

/// Get field with index I form s (index associated with DECLARE_STATE_GET())
template< int I, typename State_t >
KOKKOS_INLINE_FUNCTION
real_t& state_get( State_t& s )
{
    return State_traits<State_t>::template get<I>(s);
}

/// Get field with index I form s (index associated with DECLARE_STATE_GET()), const version
template< int I, typename State_t >
KOKKOS_INLINE_FUNCTION
const real_t& state_get( const State_t& s )
{
    return State_traits<const State_t>::template get<I>(s);
}


/**
 * Iterate over each member variable for a set of states
 * @tparam I start index (mainly here for metaprogramming purpose)
 * @param states... states to read or modify. They can be const or not.
 *                  Mixing state types is not advised
 * @param f function to apply to each field in states of type
 *          f : (real_t(&), real_t(&), ...) -> void
 *          one real_t for each const State& in `states...`
 *          one real& for each State& in `states`
 *          e.g. state_foreach_var( [](real_t&, real_t, real_t){...}, State&, const State&, const State& );
 **/
template< int I=0, typename F, typename... State_t >
KOKKOS_INLINE_FUNCTION
void state_foreach_var( const F& f, State_t&... states )
{
    f( state_get<I>(states)... );
    if constexpr( ( (I+1 < State_traits<State_t>::nvars) && ...) )
        state_foreach_var<I+1>(f, states...);
}


//################
// Arithmetic operators on states
//################

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