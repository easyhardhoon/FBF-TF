#pragma once
#include <ctime>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdarg>
#include <vector>
#include <utility>
#include <queue>
#include "condition_variable"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/c/common.h"
#include <functional>
#include "thread"
#include "future"

#include "tensorflow/lite/worker_core.h"
#include "tensorflow/lite/scheduler.h"

/*
Author : Minsung Kim
Contact : alstjd025@gmail.com


This is a WorkFrame source for worker in tensorflow lite.

Below is basic structure of WorkFrame

/////////////////////////////////////////////////////////
/                       [WorkFrame]                     /
/                                                       /
/  |-------------|     |-------------|                  /
/  |   [Worker]  |     |   [Worker]  |      . . .       /
/  |    CPU 1    |     |   CPU2 GPU1 |                  /
/  |-------------|     |-------------|                  /
/                                                       /
/                                                       /
/  |--------Scheduler---------------------------------| /
/  |                                                  | /
/  | JobQueue - [job1, job2, job3,, ]                 | /
/  |                                                  | /
/  | interpreter - [job::subgraph, job::subgraph,, ]  | /
/  |                                                  | /
/  |--------------------------------------------------| /
/                                                       /
/                                                       /
/  |--------Profiler----------------------------------| /
/  |   interpreterbuilder                             | /
/  |   -> build from model (model name)               | /
/  |   -> profile model (maybe invoke?)               | /
/  |   -> make jobs (layer, subgraph partitioning)    | /
/  |   -> give jobs to scheduler                      | /
/  |--------------------------------------------------| /
/////////////////////////////////////////////////////////


*/

namespace tflite{
class Worker; // forward declare of Worker class

class WorkFrame{
  public:
    WorkFrame();

    // A WorkerAPI needs at least one model and one worker.
    WorkFrame(const char* model_name, workerType wType);
    
    
    ~WorkFrame();

    // Create worker of a given worker type and number of cpu
    // The number of cpu means how many cpus a single worker will use.
    TfLiteStatus CreateWorker(workerType wType, int cpu_num);
  
    // Allocates a task to specific worker
    TfLiteStatus AllocateTask();


    // Creates a profiler and a scheduler
    // Call this function at only initial phase.
    TfLiteStatus CreateProfilerandScheduler(const char* model);

    // Gives a job to scheduler.
    // This task will profile a whole model with profiler
    // and partitions with layers and subgraphs.
    // A job is a subset of subgraphs.
    // TODO : consider quantized model input
    TfLiteStatus CreateAndGiveJob(const char* model);

    // Tests invoke with a single job and a single worker
    // Subject to remove
    TfLiteStatus TestInvoke();

  private:

    // vector container of worker_ids
    std::vector<int> worker_ids;

    // vector container of workers
    std::vector<Worker*> workers;
    
    // Scheduler for workers.
    // Scheduler owns the interpreter alone and workers can own its
    // weak ptr only.
    std::weak_ptr<Scheduler> planner_;

    Profiler profiler_;

};



} //namespace tflite
