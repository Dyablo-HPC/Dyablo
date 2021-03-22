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
  class CUDAEvent{
  public:    
    CUDAEvent(){
      cudaEventCreate(&event);
      cudaEventRecord(event, 0);
    }
    
    CUDAEvent(const CUDAEvent& ) = delete;
    
    ~CUDAEvent(){
      cudaEventDestroy(event);
    }
    bool check() const
    {
      return cudaEventQuery(event) == cudaSuccess;
    }
    void wait() const
    {
      cudaEventSynchronize(event);
    }
    /// Elapsed time in seconds
    static double elapsed(const CUDAEvent& begin, const CUDAEvent& end)
    {
      assert(begin.check());
      assert(end.check());

      float gpuTime;
      cudaEventElapsedTime(&gpuTime, begin.event, end.event);
      return 1e-3*gpuTime;
    }
  private:
    cudaEvent_t event;
  };

  //! queue CUDA start and stop events
  std::queue<CUDAEvent> startEv_queue, stopEv_queue;
  //! total accumulated duration
  double total_time;  

public:
  CudaTimer()
  {
    total_time = 0.0;
  }

  /**
   * @brief start timer
   * push a start even in a cuda stream 
   * also check events termination (cudaEventQuery()) and accumulate times
   **/
  void start()
  {
    startEv_queue.emplace();

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
    stopEv_queue.emplace();
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
      stopEv_queue.front().wait();

    accumulate_finished();
  
    assert(startEv_queue.empty());
    assert(stopEv_queue.empty());
  }

  /// Flush finished events from the queue
  void accumulate_finished()
  {
    while( !stopEv_queue.empty() and stopEv_queue.front().check() )
    {
      total_time += CUDAEvent::elapsed(startEv_queue.front(), stopEv_queue.front());

      startEv_queue.pop();
      stopEv_queue.pop();
    }
  }

}; // class CudaTimer

#endif // DYABLO_UTILS_MONITORING_CUDA_TIMER_H_
