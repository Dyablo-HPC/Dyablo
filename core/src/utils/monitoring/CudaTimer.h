/**
 * \file CudaTimer.h
 * \brief A simple timer class for CUDA based on events.
 *
 * \author Pierre Kestener, Arnaud Durocher
 * \date Jan 2020
 *
 */
#ifndef DYABLO_UTILS_MONITORING_CUDA_TIMER_H_
#define DYABLO_UTILS_MONITORING_CUDA_TIMER_H_

#include <queue>
#include <cuda_runtime.h>

/**
 * \brief a simple timer for CUDA kernel using CUDA events.
 * \sa https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
 * 
 * This timer uses a CUDA event queue to avoid using cudaEventSychronize() to accumulate measured times
 * start()/stop() create new cudaEvent_t handles and adds it to an event queue
 * cudaEventQuery() is called periodically (during start() and elapsed()) to empty the queue and accumulate time
 * CPU synchronization (cudaEventSynchronize()) is performed only in elapsed() (and reset())
 */
class CudaTimer
{
protected:
  //! queue CUDA start and stop events
  std::queue<cudaEvent_t> startEv_queue, stopEv_queue;
  //! total accumulated duration
  double total_time;  

public:
  CudaTimer()
  {
    total_time = 0.0;
  }

  CudaTimer(const CudaTimer&) = delete;

  ~CudaTimer()
  {
    empty_queue();
  }

  /**
   * @brief start timer
   * push a start even in a cuda stream 
   * also check events termination (cudaEventQuery()) and accumulate times
   **/
  void start()
  {
    cudaEvent_t startEv;
    cudaEventCreate(&startEv);
    cudaEventRecord(startEv, 0);
    startEv_queue.push(startEv);

    accumulate_finished();
  }

  //! reset accumulated duration and empty the queue
  void reset()
  {
    empty_queue();
    total_time = 0.0;
  }

  //! stop timer, push a stop event in a cuda stream
  void stop()
  {
    cudaEvent_t stopEv;
    cudaEventCreate(&stopEv);
    cudaEventRecord(stopEv, 0);
    stopEv_queue.push(stopEv);
  }

  /**
   * return total elapsed time in seconds
   * Performs a CPU synchronization (cudaEventSynchronize()) and flushes event queue 
   **/
  double elapsed()
  {
    empty_queue();

    return total_time;
  }

private:
  /// Wait for events and flush queue
  void empty_queue()
  {
    // Wait for last cuda event
    if(!stopEv_queue.empty())
      cudaEventSynchronize(stopEv_queue.front());

    accumulate_finished();
  
    assert(startEv_queue.empty());
    assert(stopEv_queue.empty());
  }

  /// Flush finished events from the queue
  void accumulate_finished()
  {
    while( !stopEv_queue.empty() and (cudaSuccess == cudaEventQuery(stopEv_queue.front())) )
    {
      // get elapsed time in milliseconds
      float gpuTime;
      cudaEventElapsedTime(&gpuTime, startEv_queue.front(), stopEv_queue.front());

      cudaEventDestroy(startEv_queue.front());
      cudaEventDestroy(stopEv_queue.front());

      // accumulate duration in seconds
      total_time += (double)1e-3*gpuTime;
      startEv_queue.pop();
      stopEv_queue.pop();
    }
  }

}; // class CudaTimer

#endif // DYABLO_UTILS_MONITORING_CUDA_TIMER_H_
