/**
 * \file CudaTimer.h
 * \brief A simple timer class for CUDA based on events.
 *
 * \author Pierre Kestener
 * \date 30 Oct 2010
 *
 */
#ifndef DYABLO_UTILS_MONITORING_CUDA_TIMER_H_
#define DYABLO_UTILS_MONITORING_CUDA_TIMER_H_

#include <cuda_runtime.h>

/**
 * \brief a simple timer for CUDA kernel using CUDA events.
 * \sa https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
 * CUDA kernels run asynchronously from CPU, 
 * so don't use CPU timing routines.
 */
class CudaTimer
{
protected:
  //! CUDA start and stop events
  cudaEvent_t startEv, stopEv;

  //! total accumulated duration
  double total_time;
  bool pending = false;

public:
  CudaTimer()
  {
    cudaEventCreate(&startEv);
    cudaEventCreate(&stopEv);
    total_time = 0.0;
  }

  CudaTimer(const CudaTimer&) = delete;

  ~CudaTimer()
  {
    cudaEventDestroy(startEv);
    cudaEventDestroy(stopEv);
  }

  //! start timer, push a start even in a cuda stream
  void start()
  {
    if(pending)
      sync();
    cudaEventRecord(startEv, 0);
  }

  //! reset accumulated duration
  void reset()
  {
    total_time = 0.0;
    pending = false;
  }

  //! stop timer and accumulate time in seconds
  void stop()
  {
    cudaEventRecord(stopEv, 0);
    pending=true;
  }

  //! return elapsed time in seconds (as record in total_time)
  double elapsed()
  {
    if(pending) sync();
    return total_time;
  }

private:
  void sync()
  {
    float gpuTime;
    cudaEventSynchronize(stopEv);

    // get elapsed time in milliseconds
    cudaEventElapsedTime(&gpuTime, startEv, stopEv);

    // accumulate duration in seconds
    total_time += (double)1e-3*gpuTime;
    pending=false;
  }

}; // class CudaTimer

#endif // DYABLO_UTILS_MONITORING_CUDA_TIMER_H_
