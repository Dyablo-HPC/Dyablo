/**
 * \file Timer.h
 * \brief A simple timer class for HIP based on events. generated from (CudaTimers.h with hipify)
 *
 * \author Arnaud Durocher
 *
 */
#pragma once

#include <queue>
#include <hip/hip_runtime.h>

/**
 * \brief a simple timer for HIP kernel using HIP events. (converted form CUDA with hipify)
 * \sa https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
 * 
 * This timer uses a HIP event queue to avoid using hipEventSychronize() to accumulate measured times
 * start()/stop() create new hipEvent_t handles and adds it to an event queue
 * hipEventQuery() is called periodically (during start() and elapsed()) to empty the queue and accumulate time
 * CPU synchronization (hipEventSynchronize()) is performed only in elapsed() (and reset())
 */
class HipTimer
{
protected:
  class HIPEvent{
  public:    
    HIPEvent(){
      hipEventCreate(&event);
      hipEventRecord(event, 0);
    }
    
    HIPEvent(const HIPEvent& ) = delete;
    
    ~HIPEvent(){
      hipEventDestroy(event);
    }
    bool check() const
    {
      return hipEventQuery(event) == hipSuccess;
    }
    void wait() const
    {
      hipEventSynchronize(event);
    }
    /// Elapsed time in seconds
    static double elapsed(const HIPEvent& begin, const HIPEvent& end)
    {
      assert(begin.check());
      assert(end.check());

      float gpuTime;
      hipEventElapsedTime(&gpuTime, begin.event, end.event);
      return 1e-3*gpuTime;
    }
  private:
    hipEvent_t event;
  };

  //! queue HIP start and stop events
  std::queue<HIPEvent> startEv_queue, stopEv_queue;
  //! total accumulated duration
  double total_time;  

public:
  HipTimer()
  {
    total_time = 0.0;
  }

  /**
   * @brief start timer
   * push a start even in a hip stream 
   * also check events termination (hipEventQuery()) and accumulate times
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

  //! stop timer, push a stop event in a hip stream
  void stop()
  {
    stopEv_queue.emplace();
  }

  /**
   * return total elapsed time in seconds
   * Performs a CPU synchronization (hipEventSynchronize()) and flushes event queue 
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
    // Wait for last hip event
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
      total_time += HIPEvent::elapsed(startEv_queue.front(), stopEv_queue.front());

      startEv_queue.pop();
      stopEv_queue.pop();
    }
  }

}; // class HipTimer
