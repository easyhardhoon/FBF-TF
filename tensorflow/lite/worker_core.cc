#include "tensorflow/lite/worker_core.h"
#include "tensorflow/lite/interpreter.h"

#define TFLITE_WORKER_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

#define TFLITE_WORKER_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

namespace tflite{
  Worker::Worker(){
    
  };

//legacy
  Worker::Worker(ResourceType wType, int w_id, Interpreter* interpreter){
    type = wType;
    worker_id = w_id;
    state = WorkerState::INIT_WORK;
    interpreter_ = interpreter; 
    working_thread = std::thread(&Worker::Work, this);
    working_thread.detach();
  };

//legacy
  void Worker::ChangeStateTo(WorkerState new_state){
    std::cout << " changed worker state1 " << "\n";
    std::unique_lock<std::mutex> lock(worker_lock);
    state = new_state;
    std::cout << " changed worker state 2" << "\n";
  }

  void Worker::DeleteJob(int job_id){
    std::unique_lock<std::mutex> lock(worker_lock);
    for(size_t i=0; jobs.size(); ++i){
      if(jobs[i]->job_id == job_id)
        jobs.erase(jobs.begin() + i);
    }
  }

//legacy
  void Worker::GiveJob(tflite::Job* new_job){
    std::unique_lock<std::mutex> lock(worker_lock);
    std::cout << "got job1" << "\n";
    if(!have_job)
      have_job = true;
    std::cout << "got job2" << "\n";
    jobs.push_back(new_job);
    std::cout << "got job3" << "\n";
  }

//legacy
  void Worker::WakeWorker(){
    worker_cv.notify_all();
  }

//legacy
void Worker::Work(){
  std::cout << "Worker [" << worker_id << "] started" << "\n";
  while(true){
    std::unique_lock<std::mutex> lock(worker_lock);
    worker_cv.wait(lock, [&] { return state == WorkerState::WORKING; });
    std::cout << "Worker [" << worker_id << "] woke up" << "\n";
    for(int i=0; i<jobs.size(); ++i){
      Subgraph* working_graph; 
      if(jobs[i]->resource_type == type){
        int graphs_to_invoke = jobs[i]->subgraphs.size();
        for(int j=0; j<graphs_to_invoke; ++j){
          std::cout << "working graph id : " << jobs[i]->subgraphs[j].first << "\n";
          working_graph = interpreter_->subgraph_id(jobs[i]->subgraphs[j].first);
          // check if intermediate tensor copy needed here
          //
          ////
          if(working_graph->Invoke() != kTfLiteOk){
            std::cout << "Invoke returned Error" << "\n";
          }
          std::cout << "Worker " << worker_id << " job "
                      << jobs[i]->job_id << " done" << "\n";
          interpreter_->LockJobs();
          jobs[i]->state == JobState::DONE;
          interpreter_->UnlockJobs();
          if(working_graph->GetNextSubgraph() == nullptr){
            PrintOutput(working_graph);
          }
        }
      }
    }
  }
}

void Worker::CopyIntermediateDataIfNeeded(Subgraph* subgraph){
  // use source_graph_id, dest_graph_id
  auto connect = [&](int source_subgraph, int dest_subgraph){ 
    Subgraph* source_graph = interpreter_->subgraph_id(source_subgraph);
    Subgraph* dest_graph = interpreter_->subgraph_id(dest_subgraph);
    int source_tensor_idx = source_graph->outputs()[0];
    int dest_tensor_idx = dest_graph->GetInputTensorIndex();
    TfLiteTensor* source_tensor = source_graph->tensor(source_tensor_idx);
    TfLiteTensor* dest_tensor = dest_graph->tensor(dest_tensor_idx);
    size_t source_byte_size = source_tensor->bytes;
    size_t dest_byte_size = dest_tensor->bytes;
    //source_graph->PrintTensor(*source_tensor, UnitType::GPU0);
    if(source_byte_size != dest_byte_size){
      std::cout << "Source tensor[" << source_tensor_idx << "] size "
                << static_cast<int>(source_byte_size)
                << " and Dest tensor["<< dest_tensor_idx <<"] size " 
                << static_cast<int>(dest_byte_size) << " missmatch!" << "\n";
      return kTfLiteError;
    }
    auto data_source = (float*)source_tensor->data.data;
    auto data_dest = (float*)dest_tensor->data.data;
    memcpy(data_dest, data_source, source_byte_size);
    //dest_graph->PrintTensor(*dest_tensor, UnitType::GPU0);
    if(dest_tensor->data.raw == nullptr){
      std::cout << "dest data nullptr!" << "\n";
    }
    // Save used(filled) output tensor for 
    // TensorAndIndex* used_output = new TensorAndIndex;
    // used_output->idx = source_tensor_idx;
    // used_output->tensor = source_tensor;
    // used_tensor_and_index.push_back(used_output);
    #ifdef latency_debug
      std::cout << "Tensor connection done" << "\n";
    #endif
    return kTfLiteOk;
  };
  Subgraph* prev_graph = subgraph->GetPrevSubgraph();
  if(prev_graph != nullptr){
    int source_graph_id = prev_graph->GetGraphid();
    int dest_graph_id = subgraph->GetGraphid();
    if(connect(source_graph_id, dest_graph_id) != kTfLiteOk){
      std::cout << "Tensor connection failed" << "\n";
      return;
    }
  }else{ // if nulltpr returned
    return;
  }
  return;
};

void Worker::PrintOutput(Subgraph* subgraph){
  int output_tensor_idx = subgraph->outputs()[0];
  TfLiteTensor* output_tensor = subgraph->tensor(output_tensor_idx);
  if(output_tensor != nullptr){
    PrintTensor(*output_tensor);
  }else{
    std::cout << "Worker : output tensor print ERROR" << "\n";
  }
  return;
}

// TODO: can only print fp32 tensor only.
void Worker::PrintTensor(TfLiteTensor& tensor){
  std::cout << "[Print Tensor]" << "\n";
  int tensor_data_dims_size = tensor.dims->size-1;
  int tensor_data_ch_size = tensor.dims->data[tensor_data_dims_size];
  int tensor_data_size = 1;
  int tensor_axis;
  for(int i=0; i< tensor.dims->size; i++){
    if(i == 1){
      tensor_axis = tensor.dims->data[i];
    }
    tensor_data_size *= tensor.dims->data[i]; 
  }
  std::cout << " Nunber of data : " << tensor_data_size << "\n";
  std::cout << " Tensor DATA " << "\n";
  if(tensor.type == TfLiteType::kTfLiteFloat32){
    std::cout << "[FLOAT32 TENSOR]" << "\n";
    auto data_st = (float*)tensor.data.data;
    for(int i=0; i<tensor_data_ch_size; i++){
      std::cout << "CH [" << i << "] \n";
      for(int j=0; j<tensor_data_size/tensor_data_ch_size; j++){
        float data = *(data_st+(i+j*tensor_data_ch_size));
        if (data == 0) {
          printf("%0.6f ", data);
        }
        else if (data != 0) {
          printf("%s%0.6f%s ", C_GREN, data, C_NRML);
        }
        if (j % tensor_axis == tensor_axis-1) {
          printf("\n");
        }
      }
      std::cout << "\n";
    }
  }
}

//legacy
  Worker::~Worker(){
    std::cout << "Worker destuctor called " << "\n";
  };



} //namespace tflite