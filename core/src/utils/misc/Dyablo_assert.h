#pragma once

#include <iostream>

#include "Kokkos_Core.hpp"

/// -----------------------
/// Private Implementation
/// -----------------------

#define IMPL_DYABLO_STRINGIFY(x) #x
#define IMPL_DYABLO_TOSTRING(x) KOKKOS_IMPL_STRINGIFY(x)

#define IMPL_DYABLO_EXPECT_THROW(cond, message)                                \
  {                                                                            \
    if (!bool(cond))                                                           \
    {                                                                          \
      std::ostringstream ss;                                                   \
      ss << "Dyablo expect failed at " << __FILE__ << ":" << __LINE__  << std::endl; \
      ss << "Function : " << __func__ << std::endl;                            \
      ss << "Condition : `" #cond << "`" << std::endl;                         \
      ss << "Message : " << message;                                           \
      throw std::runtime_error( ss.str() );                                    \
    }                                                                          \
  }

#define IMPL_DYABLO_EXPECT_ABORT(cond, message)                                \
  {                                                                            \
    if (!bool(cond))                                                           \
    {                                                                          \
      ::Kokkos::abort(  "Dyablo assert at " __FILE__ ":" IMPL_DYABLO_TOSTRING(__LINE__) "\n" \
                        "Condition : `" #cond "`\n"                            \
                        "Message : " #message);                                \
    }                                                                          \
  }

#define DYABLO_ASSERT_ASSERT(cond, message) assert(((void)#message, cond));

// Portability recommendations : 
// https://isocpp.org/std/standing-documents/sd-6-sg10-feature-test-recommendations/#testing-for-the-presence-of-an-attribute-__has_cpp_attribute
#ifndef __has_cpp_attribute  
//# warning "__has_cpp_attribute undefined"
# define __has_cpp_attribute(assume) 0
#endif
#ifndef __has_builtin
//# warning "__has_builtin undefined"
# define __has_builtin(b) 0
#endif

#if __has_cpp_attribute(assume) // detect if compiler supports [[assume]] (C++23)
  //# warning "Using [[assume]]"
  #define IMPL_DYABLO_OPTIM_ASSUME( cond ) [[assume(cond)]]
#elif __has_builtin(__builtin_assume) // Detect if compiler supports __builtin_assume (clang)
  //# warning "Using __builtin_assume"
  #ifdef __CLANG___
    #pragma clang diagnostic ignored "-Wassume"
  #endif
  #define IMPL_DYABLO_OPTIM_ASSUME( cond ) __builtin_assume(cond)
#else
  //# warning "assume not available"
  #define IMPL_DYABLO_OPTIM_ASSUME( cond ) {}
#endif

/// -----------------------
/// End Private Implementation
/// -----------------------

/**
 * Defines to enable/disable corresponding asserts
 * The default behavior is defined here, but asserts can be forced on/off
 * with -DDYABLO_ENABLE_ASSERT_*=0/1
 **/
#ifndef DYABLO_ENABLE_ASSERT_HOST_RELEASE
  #define DYABLO_ENABLE_ASSERT_HOST_RELEASE 1
#endif
#ifndef DYABLO_ENABLE_ASSERT_KOKKOS_RELEASE
  #define DYABLO_ENABLE_ASSERT_KOKKOS_RELEASE 1
#endif
#ifndef NDEBUG // In debug mode
  #ifndef DYABLO_ENABLE_ASSERT_HOST_DEBUG
    #define DYABLO_ENABLE_ASSERT_HOST_DEBUG 1
  #endif
  #ifndef DYABLO_ENABLE_ASSERT_KOKKOS_DEBUG
    #define DYABLO_ENABLE_ASSERT_KOKKOS_DEBUG 1
  #endif
#endif

/**
 * Enable/disable using [[assume]] or 
 * equivalent instead of assert when asser is disabled
 **/
#ifndef DYABLO_ENABLE_OPTIM_ASSUME
  #define DYABLO_ENABLE_OPTIM_ASSUME 0
#endif

/***
 * Provide information to the compiler that cond is always true
 ***/
#if DYABLO_ENABLE_OPTIM_ASSUME
  #define DYABLO_OPTIM_ASSUME(cond) IMPL_DYABLO_OPTIM_ASSUME(cond)
#else
  #define DYABLO_OPTIM_ASSUME(conde) {}
#endif

/***
 * Check if condition is met and abort otherwise
 * Works on Host only
 * Enabled in Debug and Release (when DYABLO_ENABLE_ASSERT_HOST_RELEASE is defined)
 * `message` expression is evaluated, can contain <<
 ***/
#if DYABLO_ENABLE_ASSERT_HOST_RELEASE
  #define DYABLO_ASSERT_HOST_RELEASE(cond, message) IMPL_DYABLO_EXPECT_THROW(cond, message)
#else
  #define DYABLO_ASSERT_HOST_RELEASE(cond, message) DYABLO_OPTIM_ASSUME(cond)
#endif

/***
 * Check if condition is met and abort otherwise
 * Works on Host only
 * Enabled in Debug (when DYABLO_ENABLE_ASSERT_HOST_DEBUG is defined)
 * `message` expression is evaluated, can contain <<
 ***/
#if DYABLO_ENABLE_ASSERT_HOST_DEBUG
  #define DYABLO_ASSERT_HOST_DEBUG(cond, message) IMPL_DYABLO_EXPECT_THROW(cond, message)
#else
  #define DYABLO_ASSERT_HOST_DEBUG(cond, message) DYABLO_OPTIM_ASSUME(cond)
#endif

/***
 * Check if condition is met and abort otherwise
 * Works on Host or Device
 * Enabled in Debug and Release
 * `message` expression is NOT evaluated (printed as in code)
 * This may have a big impact on performance, use it only when absolutely necessary
 ***/ 
#if DYABLO_ENABLE_ASSERT_KOKKOS_RELEASE
  #define DYABLO_ASSERT_KOKKOS_RELEASE(cond, message) IMPL_DYABLO_EXPECT_ABORT(cond, message)
#else
  #define DYABLO_ASSERT_KOKKOS_RELEASE(cond, message) DYABLO_OPTIM_ASSUME(cond)
#endif

/***
 * Check if condition is met and abort otherwise
 * Works on Host or Device
 * Enabled in Debug only
 * `message` expression is NOT evaluated (printed as in code)
 ***/ 
#if DYABLO_ENABLE_ASSERT_KOKKOS_DEBUG
    #define DYABLO_ASSERT_KOKKOS_DEBUG(cond, message) IMPL_DYABLO_EXPECT_ABORT(cond, message)
#else
    #define DYABLO_ASSERT_KOKKOS_DEBUG(cond, message) DYABLO_OPTIM_ASSUME(cond)
#endif