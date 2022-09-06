#pragma once 

#include <Kokkos_Core.hpp>
#include "DyabloSession.hpp"

namespace dyablo{

// HostDeviceSingletons should have static linkage because __device__ globals may 
// have static linkage and be local to compilation unit
namespace{

/**
 * Create a singleton from type T that is accessible on Host and Device
 * 
 * T is initialized on Host when `set()` (host only) is called and then copied to Device
 * `get()` provides an instance of T and can be called from host or device.
 * 
 * T must be default constructible and have an assignment operator, both available on Host and Device
 * HostDeviceSingleton<T>::set() should be specialized by user and call set(T)
 * 
 * WARNING : This may be unstable, use with caution. Be sure to stay within a single compilation unit 
 * for each specialization (if you don't know what a compilaion unit is, maybe you should abstain).
 * IMPORTANT : set() must be called at least once before get() IN EVERY COMPILATION UNIT where get() is called.
 **/
template< typename T >
class HostDeviceSingleton
{
public:
    /**
     * Initialize the result of `get()` in both Host and Device memory space
     * (Host only)
     * @param v_init value to initialize singleton on Host and Device
     **/
    static void set( const T& v )
    {
        get_private() = v;
        get_initialized() = true;
        Kokkos::parallel_for("HostDeviceSingleton_set_device", 1,
            KOKKOS_LAMBDA(int)
        {
            get_private() = v;
            get_initialized() = true;
        });
    }

    /** 
     * User defined default initialization for singleton
     **/
    static void set();

    /**
     * Get Singleton value (host or device)
     * Singleton needs to be set either at DyabloSession initialization 
     * (see `registered_with_dyablosession`) or by manually 
     * calling ::set() or ::set(T)
     **/
    KOKKOS_INLINE_FUNCTION
    static const T& get()
    {
        assert( get_initialized() );
        return get_private();
    }

private: 
    KOKKOS_INLINE_FUNCTION
    static T& get_private()
    {
        static T res;
        return res;
    }

    KOKKOS_INLINE_FUNCTION
    static bool& get_initialized()
    {
        static bool initialized = false;
        return initialized;
    }
};

} // namespace

} // namespace dyablo