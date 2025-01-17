/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/interpreter.h"

#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <utility>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/delegates/status.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"

#include "tensorflow/lite/kmdebug.h"


// TODO(b/139446230): Move to portable platform header.
#if defined(__ANDROID__)
#define TFLITE_IS_MOBILE_PLATFORM
#endif  // defined(__ANDROID__)

#if defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_IPHONE_SIMULATOR
#define TFLITE_IS_MOBILE_PLATFORM
#elif TARGET_OS_IPHONE
#define TFLITE_IS_MOBILE_PLATFORM
#endif
#endif  // defined(__APPLE__)

// TODO(b/132087118): move static_assert to c_api_internal when compiled with
// C++.
static_assert(sizeof(TfLiteFloat16) == sizeof(uint16_t),
              "Float 16 type must be 16 bits.");

namespace tflite {

namespace {

// Gets the current TfLiteQuantization from the legacy TfLiteQuantizationParams.
TfLiteQuantization GetQuantizationFromLegacy(
    const TfLiteQuantizationParams& legacy_quantization) {
  TfLiteQuantization quantization;
  quantization.type = kTfLiteAffineQuantization;
  auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(
      malloc(sizeof(TfLiteAffineQuantization)));
  affine_quantization->scale = TfLiteFloatArrayCreate(1);
  affine_quantization->zero_point = TfLiteIntArrayCreate(1);
  affine_quantization->scale->data[0] = legacy_quantization.scale;
  affine_quantization->zero_point->data[0] = legacy_quantization.zero_point;
  quantization.params = affine_quantization;

  return quantization;
}

// TODO(b/153131797): We have put 'delegate_status' to 0 in the following macro
// temporarily because delegate-specific error codes are either not retrievable
// at the moment, which we will add later.
#define TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(runtime_event, a) \
  do {                                                                      \
    TfLiteStatus status = (a);                                              \
    runtime_event.set_runtime_status(/*delegate_status=*/0,                 \
                                     static_cast<int64_t>(status));         \
    TF_LITE_ENSURE_STATUS(status);                                          \
  } while (0)

}  // namespace

Interpreter::Interpreter(ErrorReporter* error_reporter)
    : error_reporter_(error_reporter ? error_reporter
                                     : DefaultErrorReporter()) {
  // TODO(b/128420794): Include the TFLite runtime version in the log.
  // Prod logging is useful for mobile platforms where scraping console logs is
  // critical for debugging.
#if defined(TFLITE_IS_MOBILE_PLATFORM)
  TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO, "Initialized TensorFlow Lite runtime.");
#else
  TFLITE_LOG_ONCE(TFLITE_LOG_INFO, "Initialized TensorFlow Lite runtime.");
#endif

  // There's always at least 1 subgraph which is the primary subgraph.
  AddSubgraphs(1);
  context_ = primary_subgraph().context();

  // Reserve some space for the tensors to avoid excessive resizing.
  for (int i = 0; i < kTfLiteMaxExternalContexts; ++i) {
    external_contexts_[i] = nullptr;
  }

  // This operation is cheap because we allocate the CPU context resources (i.e.
  // threads) lazily.
  own_external_cpu_backend_context_.reset(new ExternalCpuBackendContext());
  external_contexts_[kTfLiteCpuBackendContext] =
      own_external_cpu_backend_context_.get();

  primary_subgraph().UseNNAPI(false);
}

Interpreter::~Interpreter() {
  // The owned external Cpu Backend Context will go out of scope with this
  // interpreter. If we have an external backend context that is not
  // owned, we need to clear the cache for other interpreters that may
  // use the context.
  if (external_contexts_[kTfLiteCpuBackendContext] &&
      (external_contexts_[kTfLiteCpuBackendContext] !=
       own_external_cpu_backend_context_.get())) {
    ExternalCpuBackendContext* external_context =
        static_cast<ExternalCpuBackendContext*>(
            external_contexts_[kTfLiteCpuBackendContext]);
    TfLiteInternalBackendContext* internal_context =
        external_context->internal_backend_context();
    if (internal_context) {
      // This call may have negative performance impacts on the next inference
      // for any interpreter using this context. The cache will be refreshed
      // by the next inference.
      internal_context->ClearCaches();
    }
  }
}

void Interpreter::SetExternalContext(TfLiteExternalContextType type,
                                     TfLiteExternalContext* ctx) {
  if (ctx == own_external_cpu_backend_context_.get()) {
    error_reporter_->Report(
        "WARNING: The passed external context is identical to the internally "
        "owned one.");
    return;
  }

  // We have an internally owned external context of kTfLiteCpuBackendContext.
  // If it's overwritten here, we will release the resource of the internally
  // owned external context.
  // Note: the 'max thread count' info associated with the overwritten context
  // will be lost here, and such info is now determined by the new context, thus
  // affecting how much parallelism a TFLite op would have.
  if (kTfLiteCpuBackendContext == type &&
      external_contexts_[kTfLiteCpuBackendContext] ==
          own_external_cpu_backend_context_.get()) {
    own_external_cpu_backend_context_.reset();
  }

  // This essentially changes the "external_contexts_[type]".
  primary_subgraph().SetExternalContext(type, ctx);
}

TfLiteStatus Interpreter::SetCustomAllocationForTensor(
    int tensor_index, const TfLiteCustomAllocation& allocation) {
  return primary_subgraph().SetCustomAllocationForTensor(tensor_index,
                                                         allocation);
}

TfLiteStatus Interpreter::SetInputs(std::vector<int> inputs) {
  return primary_subgraph().SetInputs(std::move(inputs));
}

TfLiteStatus Interpreter::SetOutputs(std::vector<int> outputs) {
  return primary_subgraph().SetOutputs(std::move(outputs));
}

TfLiteStatus Interpreter::SetVariables(std::vector<int> variables) {
  return primary_subgraph().SetVariables(std::move(variables));
}

/// Minsung 
// This function is a modified version of below one.
// Allocate all tensors in every subgraphs.
TfLiteStatus Interpreter::AllocateTensorsofAllSubgraphsAndFixShape(){
  if(subgraph(0)->AllocateTensors() != kTfLiteOk)
    return kTfLiteError;
  // First fix every shared tensor's size with base tensors in primary subgraph.
  for(int i=0; i<shared_tensor_and_graph.size(); ++i){
    std::cout << "shared tensor [" << shared_tensor_and_graph[i].first << "] \n";
    int base_tensor = shared_tensor_and_graph[i].first;
    TfLiteTensor* working_tensor;
    std::vector<int> match_dims;
    for(int j=0; j<shared_tensor_and_graph[i].second.size(); ++j){
      std::cout << shared_tensor_and_graph[i].second[j] << " ";
      int working_subgraph = shared_tensor_and_graph[i].second[j];
      if(j == 0){
        working_tensor = subgraph(working_subgraph)->tensor(base_tensor);
        match_dims = subgraph(working_subgraph)->GetTensorShape(base_tensor);
      }
      else
        subgraph(working_subgraph)->ResizeInputTensor(base_tensor, match_dims);
    }
    working_tensor = nullptr;
    match_dims.clear();
    std::cout << "\n";
  }
  // Then allocate every tensors
  for(int i=1; i<subgraphs_size(); ++i){
    if(subgraph(i)->AllocateTensors() != kTfLiteOk)
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus Interpreter::AllocateTensorsofAllSubgraphs(){
  for(int i=0; i < subgraphs_size(); ++i){
    if(subgraph(i)->AllocateTensors() != kTfLiteOk)
      return kTfLiteError;
  }
}

// Minsung MUST_CHECK
TfLiteStatus Interpreter::AllocateTensors() {
  std::cout << "AllocateTensors" << "\n";
  // Apply the default delegate that TFLite will enable at this point to allow
  // other user-level delegates to be applied first.

  if (!lazy_delegate_providers_.empty()) {
    TFLITE_LOG(TFLITE_LOG_INFO,
               "Applying %zu TensorFlow Lite delegate(s) lazily.",
               lazy_delegate_providers_.size());
    // At the momement, XNNPACK delegate is the only one that might be applied
    // by default, in which case, the execution will fall back to default
    // implementation if the XNNPACK delegate fails to be applied. Therefore, we
    // ignore the return status here and let it fall through the rest of the
    // code.
    for (size_t i = 0; i < lazy_delegate_providers_.size(); ++i) {
      auto status =
          ModifyGraphWithDelegate(std::move(lazy_delegate_providers_[i]));
      switch (status) {
        case kTfLiteOk:
          TFLITE_LOG(TFLITE_LOG_INFO,
                     "Successfully applied the default TensorFlow Lite "
                     "delegate indexed at %zu.",
                     i);
          break;
        case kTfLiteError:
          TF_LITE_REPORT_ERROR(error_reporter_,
                               "Failed to apply the default TensorFlow Lite "
                               "delegate indexed at %zu.",
                               i);
		  return kTfLiteError;
        case kTfLiteDelegateError:
          TF_LITE_REPORT_ERROR(
              error_reporter_,
              "Error in applying the default TensorFlow Lite delegate indexed "
              "at %zu, and all previously applied delegates are reverted.",
              i);
          break;
        case kTfLiteApplicationError:
          TF_LITE_REPORT_ERROR(error_reporter_,
                               "Ignoring failed application of the default "
                               "TensorFlow Lite delegate indexed at %zu.",
                               i);
          break;
        default:
          TF_LITE_REPORT_ERROR(error_reporter_,
                               "Unknown status (%d) after applying the default "
                               "TensorFlow Lite delegate indexed at %zu.",
                               status, i);
		  return kTfLiteError;
      }
    }
    lazy_delegate_providers_.clear();
  }
  return primary_subgraph().AllocateTensors();
}

void Interpreter::ReserveNodes(int count) {
  primary_subgraph().ReserveNodes(count);
}

void Interpreter::AddSubgraphs(int subgraphs_to_add,
                               int* first_new_subgraph_index) {
  const size_t base_index = subgraphs_.size();
  if (first_new_subgraph_index) *first_new_subgraph_index = base_index;

  subgraphs_.reserve(base_index + subgraphs_to_add);
  for (int i = 0; i < subgraphs_to_add; ++i) {
    Subgraph* subgraph = new Subgraph(error_reporter_, external_contexts_,
                                      &subgraphs_, &resources_);
    subgraphs_.emplace_back(subgraph);
  }
}

TfLiteStatus Interpreter::AddNodeWithParameters(
    const std::vector<int>& inputs, const std::vector<int>& outputs,
    const char* init_data, size_t init_data_size, void* builtin_data,
    const TfLiteRegistration* registration, int* node_index) {
  return primary_subgraph().AddNodeWithParameters(
      inputs, outputs, {}, init_data, init_data_size, builtin_data,
      registration, node_index);
}

TfLiteStatus Interpreter::ResizeInputTensor(int tensor_index,
                                            const std::vector<int>& dims) {
  return primary_subgraph().ResizeInputTensor(tensor_index, dims);
}

TfLiteStatus Interpreter::ResizeInputTensorStrict(
    int tensor_index, const std::vector<int>& dims) {
  return primary_subgraph().ResizeInputTensorStrict(tensor_index, dims);
}

TfLiteStatus Interpreter::ReleaseNonPersistentMemory() {
  // TODO(b/138790287): We could do this for all subgraphs whose tensors have
  // been allocated. However, AllocateTensors() relies on Control Flow ops to
  // allocate tensors on 'children' subgraphs. Revisit this if required.
  return primary_subgraph().ReleaseNonPersistentMemory();
}

//Minsung
//Sets partitioning ratios of subgraphs
//TODO : Set Filter Tensor for partitioning  
TfLiteStatus Interpreter::SetPartitioning(int partitioning, UnitType eType){
  int subgraph_size = subgraphs_size();
  std::cout << "Interpreter has " << subgraph_size << " subgraphs \n";
  if(subgraph_size <= 0){
    std::cout << "ERROR Interpreter has " << subgraph_size << " subgraphs \n";
    return kTfLiteError;
  }
  for(int i=0; i<subgraph_size; i++){
    subgraph(i)->subgraph_Type = eType;
    subgraph(i)->partitioning_plan = partitioning;
    subgraph(i)->use_distribute_strategy = true;
    subgraph(i)->clock_measure_data = CreateClockMeasure(4);
    // if(subgraph(i)->CheckConv2dNodes() != kTfLiteOk){
    //   std::cout << "Error in number of Conv2d" << "\n";
    //   return kTfLiteError;
    // }
    context_->use_distribute_strategy_context = true;
  }
  return kTfLiteOk;
}

//Minsung 
//Quantize all Conv2d Layer of Current Context
//Only works in CPU Context
//Must call after SetPartitioning & channelPartitioning
TfLiteStatus Interpreter::QuantizeSubgraph(){
  std::cout << "QuantizeSubgraph \n";
  int subgraph_size = subgraphs_size();
  for(int i=0; i<subgraph_size; i++){
    if(subgraph(i)->QuantizeCurrentSubgraph() != kTfLiteOk)
      return kTfLiteError;
  }
  std::cout << "QuantizeSubgraph Good \n";
  return kTfLiteOk;
}

// Minsung
// Set experimental flag for deviding a model to multiple subgraphs
void Interpreter::SetMultipleSubgraphs(bool flag){
  devide_by_conv = flag;
}

// Minsung
// Returns experimental flag which have been set above
bool Interpreter::GetMultipleSubgraphFlag(){
  return devide_by_conv;
}

TfLiteStatus Interpreter::Invoke(UnitType eType, std::mutex& mtx_lock,
                     std::mutex& mtx_lock_,
                     std::mutex& mtx_lock_debug,
                     std::condition_variable& Ucontroller,
                     std::queue<SharedContext*>* qSharedData) {
  ScopedRuntimeInstrumentationProfile scoped_runtime_event(installed_profiler_,
                                                           "invoke");
  if(eType == UnitType::CPU0){
  TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(
     scoped_runtime_event, primary_subgraph().Invoke(eType, mtx_lock, mtx_lock_,
                                         mtx_lock_debug, Ucontroller, qSharedData));
  }else if(eType == UnitType::GPU0){
    int subgraph_size = subgraphs_size();
    //std::cout << "Invoke subgrph size : " << subgraph_size << "\n";
    auto connect = [&](int source_subgraph, int dest_subgraph){
      if(source_subgraph < subgraphs_size() && dest_subgraph <= subgraphs_size()){
        Subgraph* source_graph = subgraph(source_subgraph);
        Subgraph* dest_graph = subgraph(dest_subgraph);
        int source_tensor_idx = source_graph->outputs()[0];
        int dest_tensor_idx = dest_graph->GetInputsInMultipleSubgraphs();
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
        TensorAndIndex* used_output = new TensorAndIndex;
        used_output->idx = source_tensor_idx;
        used_output->tensor = source_tensor;
        used_tensor_and_index.push_back(used_output);
        std::cout << "Tensor connection done" << "\n";
        return kTfLiteOk;
      }
    };
    auto connectAdd = [&](int dest_subgraph){
      Subgraph* dest_graph = subgraph(dest_subgraph);
      std::vector<int> inputs = dest_graph->GetMultipleInputTensorIdx();
      TfLiteTensor* source_tensor = nullptr;
      TfLiteTensor* dest_tensor = nullptr;
      // This work needs at least two tensors in both inptus and used_tensors.
      if(inputs.size() < 2 || used_tensor_and_index.size() < 2){
        std::cout << "ADD node input connection failed(size)" << 
                " input size : "<< inputs.size() << " stored tensor size : " << 
                  used_tensor_and_index.size() << "\n";
        return kTfLiteError;
      }
      for(size_t i=0; i<inputs.size(); ++i){
        for(size_t j=0; j<used_tensor_and_index.size(); ++j){
          if(used_tensor_and_index[j]->idx == inputs[i]){
            source_tensor = used_tensor_and_index[j]->tensor;
            dest_graph->SwitchTensor(*source_tensor, used_tensor_and_index[j]->idx);
            dest_tensor = dest_graph->tensor(inputs[i]);
            if(source_tensor == nullptr){
              std::cout << "Add node input connection failed(nullptr)" << "\n";
              return kTfLiteError;
            }
            size_t source_byte_size = source_tensor->bytes;
            size_t dest_byte_size = dest_tensor->bytes;
            if(source_byte_size != dest_byte_size){
              std::cout << "Source tensor[" << used_tensor_and_index[j]->idx << "] size "
                        << static_cast<int>(source_byte_size)
                        << " and Dest tensor["<< inputs[i] <<"] size " 
                        << static_cast<int>(dest_byte_size) << " missmatch!" << "\n";
              return kTfLiteError;
            }
            auto data_source = (float*)source_tensor->data.data;
            auto data_dest = (float*)dest_tensor->data.data;
            memcpy(data_dest, data_source, source_byte_size);
          }
        }
      }
      return kTfLiteOk;
    };
    struct timespec begin, end;
    for(int i=0; i<subgraph_size; i++){
      //std::cout << "Invoke Subgraph idx : " << i << "\n";
      clock_gettime(CLOCK_MONOTONIC, &begin);
      if(i > 0){
        if(strcmp(subgraph(i)->GetFirstOpName(), "ADD") == 0){
          if(connectAdd(i) == kTfLiteError){
            std::cout << "TENSOR CONNECTION FAILED" << "\n";
            return kTfLiteError;
          }
        }
        else{
          if(connect(i-1, i) == kTfLiteError){
            std::cout << "TENSOR CONNECTION FAILED" << "\n";
            return kTfLiteError;
          }
        }
      }
      clock_gettime(CLOCK_MONOTONIC, &end);
      double latency = (end.tv_sec - begin.tv_sec) + \
                        ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
      //printf("Data transfer latency : %.6fs \n", latency);
      //printf("Transfer start Timestamp %.6f \n", (begin.tv_sec + (begin.tv_nsec) / 1000000000.0));
      //printf("Transfer end Timestamp %.6f \n", (end.tv_sec + (end.tv_nsec) / 1000000000.0));
      clock_gettime(CLOCK_MONOTONIC, &begin);
      if(subgraph(i)->Invoke(eType, mtx_lock, mtx_lock_,
                            mtx_lock_debug, Ucontroller, qSharedData) != kTfLiteOk)
        return kTfLiteError;
      clock_gettime(CLOCK_MONOTONIC, &end);
      latency = (end.tv_sec - begin.tv_sec) + \
                        ((end.tv_nsec - begin.tv_nsec) / 1000000000.0);
      //printf("Invoke latency : %.6fs, Invoke start timestamp : %.6fs, end timestamp : %.6fs \n",
                   //     latency, (begin.tv_sec + (begin.tv_nsec) / 1000000000.0),
                     //             (end.tv_sec + (end.tv_nsec) / 1000000000.0));
    }
    //printf("final data ? %f \n", *(final_subgraph().tensor(163)->data.f + 954));
    if (!allow_buffer_handle_output_) {
      for (int tensor_index : outputs()) {
        TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(
            scoped_runtime_event,
            primary_subgraph().EnsureTensorDataIsReadable(tensor_index));
      }
    }
    // Minsung : This job may solve the output tensor problem..?
    // if (!allow_buffer_handle_output_) {
    //   std::cout << "Allow buffer?" << "\n";
    //   for (int tensor_index : final_output()) {
    //     TfLiteBufferHandle tflite_buffer_handle = kTfLiteNullBufferHandle;
    //     SetBufferHandle(tensor_index, ++tflite_buffer_handle, 
    //                           final_subgraph().delegates_applied_[0]);
    //     std::cout << "Tensor Index : " << tensor_index << "\n";
    //     TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(
    //         scoped_runtime_event,
    //         final_subgraph().EnsureTensorDataIsReadable(tensor_index));
    //   }
    // }
  }
  return kTfLiteOk;
}

//Minsung 
//Overloaded Invoke for other invoke calling parts
TfLiteStatus Interpreter::Invoke() {
  ScopedRuntimeInstrumentationProfile scoped_runtime_event(installed_profiler_,
                                                           "invoke");
  TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(
      scoped_runtime_event, primary_subgraph().Invoke(UnitType::NONE));

  if (!allow_buffer_handle_output_) {
    for (int tensor_index : outputs()) {
      TF_LITE_ENSURE_STATUS_WITH_SCOPED_INSTRUMENTATION(
          scoped_runtime_event,
          primary_subgraph().EnsureTensorDataIsReadable(tensor_index));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Interpreter::AddTensors(int tensors_to_add,
                                     int* first_new_tensor_index) {
  return primary_subgraph().AddTensors(tensors_to_add, first_new_tensor_index);
}

TfLiteStatus Interpreter::ResetVariableTensors() {
  return primary_subgraph().ResetVariableTensors();
}

TfLiteStatus Interpreter::SetTensorParametersReadOnly(
    int tensor_index, TfLiteType type, const char* name,
    const std::vector<int>& dims, TfLiteQuantization quantization,
    const char* buffer, size_t bytes, const Allocation* allocation) {
  return primary_subgraph().SetTensorParametersReadOnly(
      tensor_index, type, name, dims.size(), dims.data(), quantization, buffer,
      bytes, allocation);
}

TfLiteStatus Interpreter::SetTensorParametersReadWrite(
    int tensor_index, TfLiteType type, const char* name,
    const std::vector<int>& dims, TfLiteQuantization quantization,
    bool is_variable) {
  return primary_subgraph().SetTensorParametersReadWrite(
      tensor_index, type, name, dims.size(), dims.data(), quantization,
      is_variable);
}

TfLiteStatus Interpreter::SetTensorParametersReadOnly(
    int tensor_index, TfLiteType type, const char* name, const size_t rank,
    const int* dims, TfLiteQuantizationParams quantization, const char* buffer,
    size_t bytes, const Allocation* allocation) {
  TfLiteQuantization new_quantization = GetQuantizationFromLegacy(quantization);
  return primary_subgraph().SetTensorParametersReadOnly(
      tensor_index, type, name, rank, dims, new_quantization, buffer, bytes,
      allocation);
}

TfLiteStatus Interpreter::SetTensorParametersReadWrite(
    int tensor_index, TfLiteType type, const char* name, const size_t rank,
    const int* dims, TfLiteQuantizationParams quantization, bool is_variable,
    const size_t rank_dims_signature, const int* dims_signature) {
  TfLiteQuantization new_quantization = GetQuantizationFromLegacy(quantization);
  return primary_subgraph().SetTensorParametersReadWrite(
      tensor_index, type, name, rank, dims, new_quantization, is_variable,
      rank_dims_signature, dims_signature);
}

TfLiteStatus Interpreter::SetExecutionPlan(const std::vector<int>& new_plan) {
  return primary_subgraph().SetExecutionPlan(new_plan);
}

void Interpreter::UseNNAPI(bool enable) {
  TFLITE_LOG_PROD_ONCE(TFLITE_LOG_INFO,
                       "Interpreter::UseNNAPI() is deprecated. Use "
                       "tflite::NnApiDelegate() directly instead.");
  primary_subgraph().UseNNAPI(enable);
}

TfLiteStatus Interpreter::SetNumThreads(int num_threads) {
  if (num_threads < -1) {
    context_->ReportError(context_,
                          "num_threads should be >=0 or just -1 to let TFLite "
                          "runtime set the value.");
    return kTfLiteError;
  }

  for (auto& subgraph : subgraphs_) {
    subgraph->context()->recommended_num_threads = num_threads;
  }

  for (int i = 0; i < kTfLiteMaxExternalContexts; ++i) {
    auto* c = external_contexts_[i];
    if (c && c->Refresh) {
      c->Refresh(context_);
    }
  }
  return kTfLiteOk;
}

void Interpreter::SetAllowFp16PrecisionForFp32(bool allow) {
  for (auto& subgraph : subgraphs_) {
    subgraph->context()->allow_fp32_relax_to_fp16 = allow;
  }
}

// TODO(b/121264966): Subgraphs added after cancellation is set will not get the
// cancellation function added to their context.
void Interpreter::SetCancellationFunction(void* data,
                                          bool (*check_cancelled_func)(void*)) {
  for (auto& subgraph : subgraphs_) {
    subgraph->SetCancellationFunction(data, check_cancelled_func);
  }
}

bool Interpreter::IsCancelled() { return primary_subgraph().IsCancelled(); }

TfLiteStatus Interpreter::ModifyGraphWithDelegate(TfLiteDelegate* delegate) {
  TfLiteStatus status = kTfLiteOk;
  // std::cout << "subgraph size : " << subgraphs_.size() << " \n";
  // for(int i=0; i<subgraphs_.size(); ++i){
    // if(subgraphs_[i]->CheckConv2dNodes() == kTfLiteOk)
      // std::cout <<"sungraph " << i << " got conv2d" << "\n";
    // else

      // std::cout <<"sungraph " << i << " no conv2d" << "\n";
  // }
  for (auto& subgraph : subgraphs_) {
    // std::cout << "Modify Subgraph with GPU delegate \n";
    status = subgraph->ModifyGraphWithDelegate(delegate);
    if (status != kTfLiteOk) {
      break;
    }
  }
  // Delegate-specific errors can be recovered from by restoring Interpreter to
  // its original state.
  if (status == kTfLiteDelegateError) {
    TF_LITE_ENSURE_STATUS(RemoveAllDelegates());
  }
  return status;
}

TfLiteStatus Interpreter::RemoveAllDelegates() {
  for (auto& subgraph : subgraphs_) {
    TF_LITE_ENSURE_STATUS(subgraph->RemoveAllDelegates());
  }
  return kTfLiteOk;
}

bool Interpreter::HasDelegates() { return primary_subgraph().HasDelegates(); }

TfLiteStatus Interpreter::SetBufferHandle(int tensor_index,
                                          TfLiteBufferHandle buffer_handle,
                                          TfLiteDelegate* delegate) {
  TF_LITE_ENSURE(context_, tensor_index < tensors_size());
  // Minsung : maybe needs change
  std::vector<TfLiteTensor>& tensors = final_subgraph().tensors();
  TfLiteTensor* tensor = &tensors[tensor_index];

  TF_LITE_ENSURE(context_,
                 tensor->delegate == nullptr || tensor->delegate == delegate);
  tensor->delegate = delegate;
  if (tensor->buffer_handle != kTfLiteNullBufferHandle) {
    TF_LITE_ENSURE(context_, tensor->delegate->FreeBufferHandle != nullptr);
    tensor->delegate->FreeBufferHandle(context_, tensor->delegate,
                                       &tensor->buffer_handle);
  }
  tensor->buffer_handle = buffer_handle;

  return kTfLiteOk;
}

TfLiteStatus Interpreter::GetBufferHandle(int tensor_index,
                                          TfLiteBufferHandle* buffer_handle,
                                          TfLiteDelegate** delegate) {
  TF_LITE_ENSURE(context_, tensor_index < tensors_size());
  std::vector<TfLiteTensor>& tensors = primary_subgraph().tensors();
  TfLiteTensor* tensor = &tensors[tensor_index];

  *delegate = tensor->delegate;
  *buffer_handle = tensor->buffer_handle;

  return kTfLiteOk;
}

void Interpreter::SetProfiler(Profiler* profiler) {
  // Release resources occupied by owned_profiler_ which is replaced by
  // caller-owned profiler.
  owned_profiler_.reset(nullptr);
  installed_profiler_ = profiler;
  SetSubgraphProfiler();
}

void Interpreter::SetProfiler(std::unique_ptr<Profiler> profiler) {
  owned_profiler_ = std::move(profiler);
  installed_profiler_ = owned_profiler_.get();
  SetSubgraphProfiler();
}

void Interpreter::SetSubgraphProfiler() {
  for (int subgraph_index = 0; subgraph_index < subgraphs_.size();
       ++subgraph_index) {
    subgraphs_[subgraph_index]->SetProfiler(installed_profiler_,
                                            subgraph_index);
  }
}

Profiler* Interpreter::GetProfiler() {
  return primary_subgraph().GetProfiler();
}

TfLiteStatus Interpreter::PrepareTensorsSharing(UnitType eType){
  if(subgraph(0)->PrepareTensorsSharing(eType) == kTfLiteOk)
    return kTfLiteOk;
  return kTfLiteError;
}

void Interpreter::PrintOutputTensor(UnitType eType){
  std::cout << "Interpreter has " << subgraphs_size() << " subgraphs \n";
  final_subgraph().PrintOutputTensorOfSubgraph(eType);
}


}  // namespace tflite
