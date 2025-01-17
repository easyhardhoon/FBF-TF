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

#include "tensorflow/lite/core/subgraph.h"

#include <algorithm>
#include <cstdint>

#include "tensorflow/lite/arena_planner.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#include <cmath>
#include "thread"
// #include "tensorflow/lite/kmdebug.h"
// #include "tensorflow/lite/kmcontext.h"
#include <fstream> //HOON. for YOLO parsing
#include "tensorflow/lite/hoon.h"
//#define debug
// #define YOLO



namespace tflite {

namespace {

struct TfLiteQuantizationDeleter {
  void operator()(TfLiteQuantization* q) {
    if (q) TfLiteQuantizationFree(q);
  }
};

using ScopedTfLiteQuantization =
    std::unique_ptr<TfLiteQuantization, TfLiteQuantizationDeleter>;

struct TfLiteSparsityDeleter {
  void operator()(TfLiteSparsity* s) {
    if (s) TfLiteSparsityFree(s);
  }
};

using ScopedTfLiteSparsity =
    std::unique_ptr<TfLiteSparsity, TfLiteSparsityDeleter>;

TfLiteStatus ReportOpError(TfLiteContext* context, const TfLiteNode& node,
                           const TfLiteRegistration& registration,
                           int node_index, const char* message) {
  context->ReportError(
      context, "Node number %d (%s) %s.\n", node_index,
      registration.custom_name
          ? registration.custom_name
          : EnumNameBuiltinOperator(
                static_cast<BuiltinOperator>(registration.builtin_code)),
      message);
  return kTfLiteError;
}

// Stub method which returns kTfLiteError when the function is forbidden.
// We're registering this function to several different function to save
// compiled binary size. Please note the restrictions:
// * The type of first parameter have to be `TfLiteContext*`.
// * All parameters must be trivially destructible. (E.g. No C++ class)
TfLiteStatus ForbiddenContextFunction(TfLiteContext* context, ...) {
  context->ReportError(context,
                       "The function is forbidden if not calling in delegate.");
  return kTfLiteError;
}

// Set the ForbiddenContextFunction to a compatible function pointer.
template <typename FunctionType>
void SetForbiddenContextFunction(FunctionType* func) {
  *func = reinterpret_cast<FunctionType>(ForbiddenContextFunction);
}

// Returns true if at least one tensor in the given list is kTfLiteDynamic.
template <typename TensorIntArray>
bool HasDynamicTensorImpl(const TfLiteContext& context,
                          const TensorIntArray& int_array) {
  for (int i : int_array) {
    if (i == kTfLiteOptionalTensor) continue;
    const TfLiteTensor& tensor = context.tensors[i];
    if (tensor.allocation_type == kTfLiteDynamic) {
      return true;
    }
  }
  return false;
}

bool HasDynamicTensor(const TfLiteContext& context,
                      const TfLiteIntArray* int_array) {
  return HasDynamicTensorImpl(context, TfLiteIntArrayView{int_array});
}

// Gets the legacy TfLiteQuantizationParams from the current TfLiteQuantization.
TfLiteQuantizationParams GetLegacyQuantization(
    const TfLiteQuantization& quantization) {
  TfLiteQuantizationParams legacy_quantization;
  legacy_quantization.scale = 0;
  legacy_quantization.zero_point = 0;
  // If the quantization type isn't affine, return the empty
  // legacy_quantization.
  if (quantization.type != kTfLiteAffineQuantization) {
    return legacy_quantization;
  }

  auto* affine_quantization =
      static_cast<TfLiteAffineQuantization*>(quantization.params);
  if (!affine_quantization || !affine_quantization->scale ||
      !affine_quantization->zero_point ||
      affine_quantization->scale->size != 1 ||
      affine_quantization->zero_point->size != 1) {
    return legacy_quantization;
  }

  // We know its per-layer quantization now.
  legacy_quantization.scale = affine_quantization->scale->data[0];
  legacy_quantization.zero_point = affine_quantization->zero_point->data[0];
  return legacy_quantization;
}

static constexpr const char kUnknownCustomOpName[] = "UnknownCustomOp";
const char* GetTFLiteOpName(const TfLiteRegistration& op_reg) {
  if (op_reg.builtin_code == tflite::BuiltinOperator_CUSTOM) {
    const char* const custom_name = op_reg.custom_name;
    return custom_name ? custom_name : kUnknownCustomOpName;
  }
  if (op_reg.builtin_code == tflite::BuiltinOperator_DELEGATE &&
      op_reg.custom_name) {
    return op_reg.custom_name;
  }
  return tflite::EnumNamesBuiltinOperator()[op_reg.builtin_code];
}

TfLiteStatus ValidateCustomAllocationForTensor(
    TfLiteContext* context, const TfLiteTensor* tensor,
    const TfLiteCustomAllocation& allocation) {
  TF_LITE_ENSURE(context, allocation.data != nullptr);
  TF_LITE_ENSURE(context, allocation.bytes >= tensor->bytes);
  // Ensure provided memory is aligned to what TFLite requires.
  const intptr_t data_ptr_value = reinterpret_cast<intptr_t>(allocation.data);
  TF_LITE_ENSURE(context, data_ptr_value % kDefaultTensorAlignment == 0);
  return kTfLiteOk;
}

}  // namespace

// A trivial implementation of GraphInfo around the Interpreter.
// NOTE: this interpreter info represents the subset of the
// graph that is executed according to execution plan. Thus,
// the indices are execution plan indices rather than raw node
// indices.
class InterpreterInfo : public GraphInfo {
 public:
  explicit InterpreterInfo(Subgraph* subgraph) : subgraph_(subgraph) {}

  size_t num_tensors() const override { return subgraph_->tensors().size(); }
  TfLiteTensor* tensor(size_t index) override {
    return &subgraph_->tensors()[index];
  }
  size_t num_execution_nodes() const override {
    return subgraph_->execution_plan().size();
  }
  size_t num_total_nodes() const override { return subgraph_->nodes_size(); }
  const TfLiteNode& node(size_t index) const override {
    int node_index = subgraph_->execution_plan()[index];
    return subgraph_->nodes_and_registration()[node_index].first;
  }
  size_t node_index(size_t index) const override {
    return subgraph_->execution_plan()[index];
  }
  const std::vector<int>& inputs() const override {
    return subgraph_->inputs();
  }
  const std::vector<int>& outputs() const override {
    return subgraph_->outputs();
  }
  const std::vector<int>& variables() const override {
    return subgraph_->variables();
  }

 public:
  Subgraph* subgraph_;
};

Subgraph::Subgraph(ErrorReporter* error_reporter,
                   TfLiteExternalContext** external_contexts,
                   std::vector<std::unique_ptr<Subgraph>>* subgraphs,
                   resource::ResourceMap* resources)
    : external_contexts_(external_contexts),
      error_reporter_(error_reporter),
      next_execution_plan_index_to_prepare_(0),
      next_execution_plan_index_to_plan_allocation_(0),
      subgraphs_(subgraphs),
      resources_(resources) {
  // TODO(b/161272052): Consider a better TfLiteContext initialization pattern:
  context_.impl_ = static_cast<void*>(this);
  context_.ResizeTensor = ResizeTensor;
  context_.ReportError = ReportErrorC;
  context_.AddTensors = AddTensors;
  context_.tensors = nullptr;
  context_.tensors_size = 0;
  context_.allow_fp32_relax_to_fp16 = false;
  context_.recommended_num_threads = -1;
  context_.GetExternalContext = GetExternalContext;
  context_.SetExternalContext = SetExternalContext;
  context_.profiler = nullptr;
  context_.GetTensor = nullptr;
  context_.GetEvalTensor = nullptr;
  //Minsung
  context_.use_distribute_strategy_context = false;

  // Reserve some space for the tensors to avoid excessive resizing.
  tensors_.reserve(kTensorsReservedCapacity);
  nodes_and_registration().reserve(kTensorsReservedCapacity);
  // Invalid to call these these except from TfLiteDelegate
  SwitchToKernelContext();
}

Subgraph::~Subgraph() {
  for (int node_index = 0; node_index < nodes_and_registration_.size();
       ++node_index) {
    CleanupNode(node_index);
  }

  for (size_t i = 0; i < context_.tensors_size; i++) {
    TfLiteTensor* tensor = &context_.tensors[i];
    if (tensor->buffer_handle != kTfLiteNullBufferHandle &&
        tensor->delegate->FreeBufferHandle != nullptr) {
      tensor->delegate->FreeBufferHandle(&context_, tensor->delegate,
                                         &tensor->buffer_handle);
    }
    TfLiteTensorFree(tensor);
  }
}

void Subgraph::CleanupNode(int node_index) {
  TfLiteNode& node = nodes_and_registration_[node_index].first;
  const TfLiteRegistration& registration =
      nodes_and_registration_[node_index].second;
  TfLiteIntArrayFree(node.inputs);
  TfLiteIntArrayFree(node.outputs);
  TfLiteIntArrayFree(node.temporaries);
  TfLiteIntArrayFree(node.intermediates);
  if (node.builtin_data) free(node.builtin_data);
  OpFree(registration, node.user_data);
  node.builtin_data = nullptr;
}

TfLiteStatus Subgraph::ReplaceNodeSubsetsWithDelegateKernels(
    TfLiteContext* context, TfLiteRegistration registration,
    const TfLiteIntArray* nodes_to_replace, TfLiteDelegate* delegate) {
  return static_cast<Subgraph*>(context->impl_)
      ->ReplaceNodeSubsetsWithDelegateKernels(registration, nodes_to_replace,
                                              delegate);
}

namespace {

// Copy a std::vector<int> to an existing TfLiteIntArray.
// This is a low-level data manipulation function, and it's caller's
// responsibility to ensure TfLiteIntArray has enough size.
void CopyVectorToTfLiteIntArray(const std::vector<int>& vec,
                                TfLiteIntArray* arr) {
  arr->size = vec.size();
  memcpy(arr->data, vec.data(), sizeof(int) * arr->size);
}

// This function allocates a continuous memory space that contains a
// TfLiteDelegateParams followed by a several TfLiteIntArray.
// When calling `free` at TfLiteDelegateParams*, all the allocated space
// will be freed together.
//
// +-----------------------------------+
// | TfLiteDelegateParams              |
// | TfLiteDelegate* delegate;         |
// | TfLiteIntArray* nodes_to_replace; |--\
// | TfLiteIntArray* input_tensors;    |--+--\
// | TfLiteIntArray* output_tensors;   |--+--+--\
// +-----------------------------------+  |  |  |
// | TfLiteIntArray (variable size)    |<-/  |  |
// +-----------------------------------+     |  |
// | TfLiteIntArray (variable size)    |<----/  |
// +-----------------------------------+        |
// | TfLiteIntArray (variable size)    |<-------/
// +-----------------------------------+
TfLiteDelegateParams* CreateDelegateParams(TfLiteDelegate* delegate,
                                           const NodeSubset& node_subset) {
  // Step 1: Calculate the allocation size.
  int allocation_size = sizeof(TfLiteDelegateParams);

  int nodes_to_replace_size =
      TfLiteIntArrayGetSizeInBytes(node_subset.nodes.size());
  allocation_size += nodes_to_replace_size;

  int input_tensors_size =
      TfLiteIntArrayGetSizeInBytes(node_subset.input_tensors.size());
  allocation_size += input_tensors_size;

  int output_tensors_size =
      TfLiteIntArrayGetSizeInBytes(node_subset.output_tensors.size());
  allocation_size += output_tensors_size;

  // Step 2: Allocate the memory.
  // Use `char*` for conveniently step through the allocated space by bytes.
  char* allocation = static_cast<char*>(malloc(allocation_size));

  // Step 3: Fill all data structures structures.
  TfLiteDelegateParams* params =
      reinterpret_cast<TfLiteDelegateParams*>(allocation);
  params->delegate = delegate;
  allocation += sizeof(TfLiteDelegateParams);

  params->nodes_to_replace = reinterpret_cast<TfLiteIntArray*>(allocation);
  CopyVectorToTfLiteIntArray(node_subset.nodes, params->nodes_to_replace);
  allocation += nodes_to_replace_size;

  params->input_tensors = reinterpret_cast<TfLiteIntArray*>(allocation);
  CopyVectorToTfLiteIntArray(node_subset.input_tensors, params->input_tensors);
  allocation += input_tensors_size;

  params->output_tensors = reinterpret_cast<TfLiteIntArray*>(allocation);
  CopyVectorToTfLiteIntArray(node_subset.output_tensors,
                             params->output_tensors);
  allocation += output_tensors_size;

  return params;
}

// Assumes that params is not nullptr.
void PopulatePreviewDelegateParams(const NodeSubset& node_subset,
                                   TfLiteDelegateParams* params) {
  // Since these params are used for previewing partitioning, params->delegate
  // is not required.
  params->delegate = nullptr;

  params->nodes_to_replace = TfLiteIntArrayCreate(node_subset.nodes.size());
  CopyVectorToTfLiteIntArray(node_subset.nodes, params->nodes_to_replace);

  params->input_tensors =
      TfLiteIntArrayCreate(node_subset.input_tensors.size());
  CopyVectorToTfLiteIntArray(node_subset.input_tensors, params->input_tensors);

  params->output_tensors =
      TfLiteIntArrayCreate(node_subset.output_tensors.size());
  CopyVectorToTfLiteIntArray(node_subset.output_tensors,
                             params->output_tensors);
}

}  // namespace

TfLiteStatus Subgraph::ReplaceNodeSubsetsWithDelegateKernels(
    TfLiteRegistration registration, const TfLiteIntArray* nodes_to_replace,
    TfLiteDelegate* delegate) {
  #ifdef DEBUG
  	SFLAG();
  #endif
  // std::cout << "(2) : Start ReplaceNodeSubsetsWithDelegateKernels logic" << "\n";
  // Ignore empty node replacement sets.
  if (!nodes_to_replace->size) {
	return kTfLiteOk;
  }

  // Annotate the registration as DELEGATE op.
  registration.builtin_code = BuiltinOperator_DELEGATE;

  // Analyze the graph to find all independent node_subsets that are either
  // fully not-this-delegate or this-delegate computation.
  InterpreterInfo info(this);
  std::vector<NodeSubset> node_subsets;
  PartitionGraphIntoIndependentNodeSubsets(&info, nodes_to_replace,
                                           &node_subsets);

  TFLITE_LOG(
      tflite::TFLITE_LOG_INFO,
      "Replacing %d node(s) with delegate (%s) node, yielding %zu partitions.",
      nodes_to_replace->size,
      registration.custom_name ? registration.custom_name : "unknown",
      node_subsets.size());
    

  context_cpu = context_;
  execution_plan_cpu = execution_plan_;
  execution_plan_.clear();
  
  for (auto& node_subset : node_subsets) {
    // Subsets claimed by the delegate should have a "macro" op created, the
    // other node_subsets (kTfNonPartition) just have their nodes added back to
    // the execution plan.
    switch (node_subset.type) {
      case NodeSubset::kTfNonPartition:
        for (auto it = node_subset.nodes.begin(); it != node_subset.nodes.end();
             ++it) {
          execution_plan_.push_back(*it);
        }
        break;
      case NodeSubset::kTfPartition: {
        int node_index;

        TfLiteDelegateParams* params =
            CreateDelegateParams(delegate, node_subset);
        TF_LITE_ENSURE_STATUS(AddNodeWithParameters(
            node_subset.input_tensors, node_subset.output_tensors, {}, nullptr,
            0, params, &registration, &node_index));

        // Initialize the output tensors's delegate-related fields.
        for (int tensor_index : node_subset.output_tensors) {
          TfLiteTensor* tensor = &tensors_[tensor_index];
          TF_LITE_ENSURE(&context_, tensor->delegate == nullptr ||
                                        tensor->delegate == delegate);
          tensor->delegate = delegate;
        }

        // Associate the node with the delegate.
        TfLiteNode* node = &nodes_and_registration_[node_index].first;
        node->delegate = delegate;
        //TfLiteRegistration* reg = &nodes_and_registration_[node_index].second;
        //std::cout << reg->custom_name << std::endl;
      } break;
      case NodeSubset::kTfUnexplored: ;
		return kTfLiteError;
    // std::cout << "(2) : End ReplaceNodeSubsetsWithDelegateKernels logic" << "\n";
        break;
    }
  }
  // std::cout << "(2) : End ReplaceNodeSubsetsWithDelegateKernels logic" << "\n";
  return kTfLiteOk;
}

TfLiteExternalContext* Subgraph::GetExternalContext(
    TfLiteExternalContextType type) {
  if (static_cast<int>(type) >= 0 && type < kTfLiteMaxExternalContexts) {
    return external_contexts_[type];
  }
  return nullptr;
}

TfLiteExternalContext* Subgraph::GetExternalContext(
    struct TfLiteContext* context, TfLiteExternalContextType type) {
  return static_cast<Subgraph*>(context->impl_)->GetExternalContext(type);
}

void Subgraph::SetExternalContext(TfLiteExternalContextType type,
                                  TfLiteExternalContext* ctx) {
  if (static_cast<int>(type) >= 0 && type < kTfLiteMaxExternalContexts) {
    external_contexts_[type] = ctx;
  }
}

void Subgraph::SetExternalContext(struct TfLiteContext* context,
                                  TfLiteExternalContextType type,
                                  TfLiteExternalContext* ctx) {
  return static_cast<Subgraph*>(context->impl_)->SetExternalContext(type, ctx);
}

// Gets an TfLiteIntArray* representing the execution plan. The interpreter owns
// this memory and it is only guaranteed to exist during the invocation of the
// delegate prepare.
TfLiteStatus Subgraph::GetExecutionPlan(TfLiteIntArray** execution_plan) {
  // TODO(aselle): Do not make a copy here
  plan_cache_.reset(TfLiteIntArrayCreate(execution_plan_.size()));
  *execution_plan = plan_cache_.get();
  static_assert(sizeof(plan_cache_->data[0]) == sizeof(execution_plan_[0]),
                "TfLiteIntArray and execution_plan do not contain same type.");
  std::memcpy(plan_cache_->data, execution_plan_.data(),
              sizeof(plan_cache_->data[0]) * execution_plan_.size());
  return kTfLiteOk;
}

// WARNING: This is an experimental interface that is subject to change.
// Entry point for C node plugin API to get the execution plan
TfLiteStatus Subgraph::GetExecutionPlan(struct TfLiteContext* context,
                                        TfLiteIntArray** execution_plan) {
  return static_cast<Subgraph*>(context->impl_)
      ->GetExecutionPlan(execution_plan);
}

void Subgraph::FreeDelegatePartitioningData() {
  for (auto& params : partitioning_preview_cache_) {
    TfLiteIntArrayFree(params.nodes_to_replace);
    TfLiteIntArrayFree(params.input_tensors);
    TfLiteIntArrayFree(params.output_tensors);
  }
  partitioning_preview_cache_.clear();
}

TfLiteStatus Subgraph::PreviewDelegatePartitioning(
    const TfLiteIntArray* nodes_to_replace,
    TfLiteDelegateParams** partition_params_array, int* num_partitions) {
  // Ensure partitioning cache is empty.
  FreeDelegatePartitioningData();
  // Defaults.
  if (!partition_params_array || !num_partitions) return kTfLiteError;
  *partition_params_array = nullptr;
  *num_partitions = 0;
  if (!nodes_to_replace->size) {
    return kTfLiteOk;
  }

  // Partition the execution plan into node subsets.
  InterpreterInfo info(this);
  std::vector<NodeSubset> node_subsets;
  PartitionGraphIntoIndependentNodeSubsets(&info, nodes_to_replace,
                                           &node_subsets);

  // Create one TfLiteDelegateParams per node-subset which would be delegated.
  for (auto& node_subset : node_subsets) {
    if (node_subset.type != NodeSubset::kTfPartition) {
      continue;
    }
    partitioning_preview_cache_.emplace_back();
    PopulatePreviewDelegateParams(node_subset,
                                  &partitioning_preview_cache_.back());
    ++*num_partitions;
  }

  *partition_params_array = partitioning_preview_cache_.data();
  return kTfLiteOk;
}

TfLiteStatus Subgraph::PreviewDelegatePartitioning(
    struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
    TfLiteDelegateParams** partition_params_array, int* num_partitions) {
  return static_cast<Subgraph*>(context->impl_)
      ->PreviewDelegatePartitioning(nodes_to_replace, partition_params_array,
                                    num_partitions);
}

TfLiteStatus Subgraph::SetInputs(std::vector<int> inputs) {
  TF_LITE_ENSURE_OK(&context_,
                    CheckTensorIndices("inputs", inputs.data(), inputs.size()));
  inputs_ = std::move(inputs);
  return kTfLiteOk;
}

TfLiteStatus Subgraph::SetOutputs(std::vector<int> outputs) {
  TF_LITE_ENSURE_OK(
      &context_, CheckTensorIndices("outputs", outputs.data(), outputs.size()));
  outputs_ = std::move(outputs);
  return kTfLiteOk;
}

TfLiteStatus Subgraph::SetVariables(std::vector<int> variables) {
  TF_LITE_ENSURE_OK(&context_, CheckTensorIndices("variables", variables.data(),
                                                  variables.size()));
  variables_ = std::move(variables);
  return kTfLiteOk;
}

void Subgraph::SetCancellationFunction(void* data,
                                       bool (*check_cancelled_func)(void*)) {
  cancellation_data_ = data;
  check_cancelled_func_ = check_cancelled_func;
}

bool Subgraph::IsCancelled() {
  return (check_cancelled_func_ != nullptr) &&
         (*check_cancelled_func_)(cancellation_data_);
}

void Subgraph::ReserveNodes(int count) {
  nodes_and_registration_.reserve(count);
}

TfLiteStatus Subgraph::CheckTensorIndices(const char* label, const int* indices,
                                          int length) {
  // Making sure kTfLiteOptionalTensor is not re-defined to something other than
  // -1.
  static_assert(kTfLiteOptionalTensor == -1,
                "kTfLiteOptionalTensor should be defined -1");

  for (int i = 0; i < length; i++) {
    int index = indices[i];
    // Continue if index == kTfLiteOptionalTensor before additional comparisons
    // below, size_t(-1) is always >= context_tensors_size.
    if (index == kTfLiteOptionalTensor) {
      continue;
    }
    if (index < 0 || static_cast<size_t>(index) >= context_.tensors_size) {
      ReportError(
          "Invalid tensor index %d in %s. The subgraph has %d tensors\n", index,
          label, context_.tensors_size);
      consistent_ = false;
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

// We have two arrays and we need to check that elements from one array don't
// show up in the other. We could sort both arrays and then iterate with two
// pointers from start to finish always increasing the smaller one but since
// these arrays are usually short (<25 elements for inputs, usually <3 for
// outputs), this might be slower than the naive approach (if arrays have size n
// and m, with n >> m ~ O(1), first approach is O(nlogn) whereas the other is
// O(n)). Plus, sorting the input and output arrays might not be something we
// want as it destroys ordering of elements.
//
// If it turns out that this is an issue, we can switch to the other algorithm.
TfLiteStatus Subgraph::CheckInputAndOutputForOverlap(const int* input_indices,
                                                     int num_inputs,
                                                     const int* output_indices,
                                                     int num_outputs) {
  for (int i = 0; i < num_inputs; i++) {
    for (int j = 0; j < num_outputs; j++) {
      if (input_indices[i] == output_indices[j]) {
        ReportError("Tensor %d is both input %d and output %d\n",
                    input_indices[i], i, j);
        consistent_ = false;
        return kTfLiteError;
      }
    }
  }
  return kTfLiteOk;
}

namespace {
// Multiply two sizes and return true if overflow occurred;
// This is based off tensorflow/overflow.h but is simpler as we already
// have unsigned numbers. It is also generalized to work where sizeof(size_t)
// is not 8.
TfLiteStatus MultiplyAndCheckOverflow(size_t a, size_t b, size_t* product) {
  // Multiplying a * b where a and b are size_t cannot result in overflow in a
  // size_t accumulator if both numbers have no non-zero bits in their upper
  // half.
  constexpr size_t size_t_bits = 8 * sizeof(size_t);
  constexpr size_t overflow_upper_half_bit_position = size_t_bits / 2;
  *product = a * b;
  // If neither integers have non-zero bits past 32 bits can't overflow.
  // Otherwise check using slow devision.
  if (TFLITE_EXPECT_FALSE((a | b) >> overflow_upper_half_bit_position != 0)) {
    if (a != 0 && *product / a != b) return kTfLiteError;
  }
  return kTfLiteOk;
}
}  // namespace

TfLiteStatus Subgraph::BytesRequired(TfLiteType type, const int* dims,
                                     size_t dims_size, size_t* bytes) {
  TF_LITE_ENSURE(&context_, bytes != nullptr);
  size_t count = 1;
  for (int k = 0; k < dims_size; k++) {
    size_t old_count = count;
    TF_LITE_ENSURE_MSG(
        &context_,
        MultiplyAndCheckOverflow(old_count, dims[k], &count) == kTfLiteOk,
        "BytesRequired number of elements overflowed.\n");
  }
  size_t type_size = 0;
  TF_LITE_ENSURE_OK(&context_, GetSizeOfType(&context_, type, &type_size));
  TF_LITE_ENSURE_MSG(
      &context_, MultiplyAndCheckOverflow(type_size, count, bytes) == kTfLiteOk,
      "BytesRequired number of bytes overflowed.\n");
  return kTfLiteOk;
}

TfLiteStatus Subgraph::AllocateTensors() {
  TFLITE_SCOPED_TAGGED_DEFAULT_PROFILE(profiler_.get(), "AllocateTensors");
  if (!consistent_) {
    ReportError("AllocateTensors() called on inconsistent model.");
	return kTfLiteError;
  }

  // Restore delegation state if applicable.
  TF_LITE_ENSURE_STATUS(RedoAllDelegates());
  
  // Explicit (re)allocation is necessary if nodes have been changed or tensors
  // have been resized. For inputs marked as dynamic, we can't short-circuit the
  // allocation as the client may have done the resize manually.
  if (state_ != kStateUninvokable &&
      !HasDynamicTensorImpl(context_, inputs())) {
    if (memory_planner_ && !memory_planner_->HasNonPersistentMemory()) {
      // If the only change was the release of non-persistent memory via
      // ReleaseNonPersistentMemory(), just re-allocate it. For any other type
      // of memory-planning change (for eg, ResizeInputTensor), the state would
      // be kStateUninvokable.
      memory_planner_->AcquireNonPersistentMemory();
    }	
    return kTfLiteOk;
  }

  next_execution_plan_index_to_prepare_ = 0;
  next_execution_plan_index_to_plan_allocation_ = 0;
  next_original_execution_plan_index_to_prepare_ = 0;
  if (memory_planner_) {
    TF_LITE_ENSURE_STATUS(memory_planner_->ResetAllocations());
  }

  TF_LITE_ENSURE_STATUS(PrepareOpsAndTensors());

  state_ = kStateInvokable;

  // Reset the variable tensors to zero after (re)allocating the tensors.
  // Developers shouldn't rely on the side effect of this function to reset
  // variable tensors. They should call `ResetVariableTensors` directly
  // instead.
  ResetVariableTensors(); 
  // KMCONTEXT(); // to use tflite->interpreter->subgraph  main structure ..
  return kTfLiteOk;
}

// TODO(ycling): Support non-zero default values.
TfLiteStatus Subgraph::ResetVariableTensors() {
  for (auto& tensor : tensors_) {
    if (!tensor.is_variable) {
      continue;
    }

    if (tensor.allocation_type == kTfLiteArenaRwPersistent) {
      // If variable tensors allocation type is `kTfLiteArenaRwPersistent`, then
      // they must be allocated after the initial `PrepareOpsAndTensors()` is
      // called.
      TF_LITE_ENSURE(&context_, tensor.data.raw != nullptr);
      tflite::ResetVariableTensor(&tensor);
    } else {
      // If variable tensors allocation type is not `kTfLiteArenaRwPersistent`,
      // then it can only be `kTfLiteCustom` in which case, we do not reset it.
      TF_LITE_ENSURE_EQ(&context_, tensor.allocation_type, kTfLiteCustom);
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Subgraph::AddNodeWithParameters(
    const std::vector<int>& inputs, const std::vector<int>& outputs,
    const std::vector<int>& intermediates, const char* init_data,
    size_t init_data_size, void* builtin_data,
    const TfLiteRegistration* registration, int* node_index) {
  #ifdef DEBUG
    SFLAG();
  #endif
  
  // std::cout << "in: ";
  // for(int i = 0; i < inputs.size(); ++i) {
  // 	std::cout << inputs[i] << " ";
  // }std::cout << std::endl;

  // std::cout << "out: ";
  // for(int i = 0; i < outputs.size(); ++i) {
	// std::cout << outputs[i] << " ";
  // } std::cout << std::endl;

  //   std::cout << "intermediates: ";
  // for(int i = 0; i < intermediates.size(); ++i) {
	// std::cout << intermediates[i] << " ";
  // } std::cout << std::endl;

  std::unique_ptr<void, decltype(free)*> builtin_data_deleter(builtin_data,
                                                              free);
  if (state_ == kStateInvokableAndImmutable) {
    ReportError("AddNodeWithParameters is disallowed when graph is immutable.");
    return kTfLiteError;
  }
  state_ = kStateUninvokable;

  TF_LITE_ENSURE_OK(&context_, CheckTensorIndices("node inputs", inputs.data(),
                                                  inputs.size()));
  TF_LITE_ENSURE_OK(
      &context_,
      CheckTensorIndices("node outputs", outputs.data(), outputs.size()));
  // For builtin ops, inputs and outputs must not overlap. Custom ops must do
  // this check by themselves if they don't support overlapping tensors. This
  // distinction is to allow custom ops to just forward a tensor, reusing it as
  // both input and output.
  if (builtin_data != nullptr) {
    TF_LITE_ENSURE_OK(&context_, CheckInputAndOutputForOverlap(
                                     inputs.data(), inputs.size(),
                                     outputs.data(), outputs.size()));
  }

  int new_node_index = nodes_and_registration_.size();
  if (node_index) *node_index = new_node_index;
  nodes_and_registration_.resize(nodes_and_registration_.size() + 1);
  auto& node_and_reg = nodes_and_registration_.back();
  TfLiteNode& node = node_and_reg.first;
  if (node.inputs) TfLiteIntArrayFree(node.inputs);
  if (node.outputs) TfLiteIntArrayFree(node.outputs);
  if (node.intermediates) TfLiteIntArrayFree(node.intermediates);
  if (node.temporaries) TfLiteIntArrayFree(node.temporaries);

  // NOTE, here we are not using move semantics yet, since our internal
  // representation isn't std::vector, but in the future we would like to avoid
  // copies, so we want the interface to take r-value references now.
  node.inputs = ConvertVectorToTfLiteIntArray(inputs);
  node.outputs = ConvertVectorToTfLiteIntArray(outputs);
  node.intermediates = ConvertVectorToTfLiteIntArray(intermediates);
  node.temporaries = TfLiteIntArrayCreate(0);
  if (init_data) {
    node.user_data = OpInit(*registration, init_data, init_data_size);
  } else {
    node.user_data = OpInit(
        *registration, static_cast<const char*>(builtin_data_deleter.get()), 0);
  }

  node.builtin_data = builtin_data_deleter.release();
  // TODO(ycling): Filling `custom_initial_data` and `custom_initial_data_size`
  // properly for nodes generated by ReplaceNodeSubsetsWithDelegateKernels.

  if (registration->builtin_code == BuiltinOperator_CUSTOM) {
    // When it's a CUSTOM op, the `custom_options` field in the Flatbuffer
    // `Operator` table is passed in.
    node.custom_initial_data = init_data;
    node.custom_initial_data_size = init_data_size;
  } else {
    node.custom_initial_data = nullptr;
    node.custom_initial_data_size = 0;
  }

  node.delegate = nullptr;
  // Copying of registration is required to support unresolved custom ops.
  node_and_reg.second = *registration;
  execution_plan_.push_back(new_node_index);
  #ifdef DEBUG
  //std::cout << "addnode : " << node.outputs->data[0] << std::endl;
  #endif
  //PrintNodeInfo(new_node_index, node, *registration);
  return kTfLiteOk;
}

TfLiteStatus Subgraph::ResizeInputTensor(int tensor_index,
                                         const std::vector<int>& dims) {
  const bool delegates_applied = !pre_delegation_execution_plan_.empty();
  const bool graph_is_immutable = state_ == kStateInvokableAndImmutable;
  if (graph_is_immutable && !delegates_applied) {
    ReportError("ResizeInputTensor is disallowed when graph is immutable.");
    return kTfLiteError;
  }

  // TODO(aselle): All bounds checks can be implemented as one-sided bounds
  // checks by casting to unsigned for efficiency. Profile before doing this.
  TF_LITE_ENSURE(&context_,
                 tensor_index < context_.tensors_size && tensor_index >= 0);
  TfLiteTensor* tensor = &context_.tensors[tensor_index];

  // Short-circuit the state change if the dimensions don't change, avoiding
  // unnecessary (re)allocations.
  //
  // Note that it's required to check `tensor->data.raw != nullptr`. Otherwise
  // the subgraph won't allocate memory for a dynamic tensor when its size
  // is equal to the original tensor size.
  if (tensor->data.raw != nullptr &&
      EqualArrayAndTfLiteIntArray(tensor->dims, dims.size(), dims.data())) {
    return kTfLiteOk;
  }

  if (graph_is_immutable) {
    // Undo delegation if it resulted in the graph being immutable.
    TF_LITE_ENSURE_STATUS(UndoAllDelegates());
  }
  state_ = kStateUninvokable;
  return ResizeTensorImpl(tensor, ConvertVectorToTfLiteIntArray(dims));
}

TfLiteStatus Subgraph::ResizeInputTensorStrict(int tensor_index,
                                               const std::vector<int>& dims) {
  TF_LITE_ENSURE(&context_,
                 tensor_index < context_.tensors_size && tensor_index >= 0);
  TfLiteTensor* tensor = &context_.tensors[tensor_index];

  // Ensure that only unknown dimensions can be resized.
  TF_LITE_ENSURE_EQ(&context_, tensor->dims->size, dims.size());
  for (size_t idx = 0; idx < dims.size(); idx++) {
    // `dims_signature` is not defined when no unknown dimensions are present.
    int dim_signature;
    if (tensor->dims_signature && tensor->dims_signature->size) {
      dim_signature = tensor->dims_signature->data[idx];
    } else {
      dim_signature = tensor->dims->data[idx];
    }

    if (dim_signature != -1 && dim_signature != dims[idx]) {
      ReportError(
          "Attempting to resize dimension %d of tensor %d with value %d to %d. "
          "ResizeInputTensorStrict only allows mutating unknown dimensions "
          "identified by -1.",
          idx, tensor_index, dim_signature, dims[idx]);
      return kTfLiteError;
    }
  }

  return ResizeInputTensor(tensor_index, dims);
}

TfLiteStatus Subgraph::ReleaseNonPersistentMemory() {
  if (memory_planner_) {
    TF_LITE_ENSURE_STATUS(memory_planner_->ReleaseNonPersistentMemory());
  }
  return kTfLiteOk;
}

TfLiteStatus Subgraph::OpPrepare(const TfLiteRegistration& op_reg,
                                 TfLiteNode* node) {
  if (op_reg.prepare == nullptr) {
    // Check if it's an unresolved custom op.
    if (IsUnresolvedCustomOp(op_reg)) {
      if (IsFlexOp(op_reg.custom_name)) {
        ReportError(
            "Regular TensorFlow ops are not supported by this interpreter. "
            "Make sure you apply/link the Flex delegate before inference.");
      } else {
        ReportError("Encountered unresolved custom op: %s.",
                    op_reg.custom_name ? op_reg.custom_name : "UnknownOp");
      }
      return kTfLiteError;
    }
    // Resolved ops can have a null Prepare function.
    return kTfLiteOk;
  }
  if(strcmp(GetOpName(op_reg), "CONV_2D") == 0) {
    // std::cout << "it's conv" << "\n";
    // std::cout << node->outputs->data[0] << " \n";
    // TfLiteTensor* tensor_ = GetOutputTensor(*node);
    // std::cout << "output tensor : ";
    // std::cout << tensor_->dims->data[0] << "\n";
  }else {
    // std::cout << "not conv " << GetOpName(op_reg) << "\n";
  }
  return op_reg.prepare(&context_, node);
}

TfLiteStatus Subgraph::PrepareOpsStartingAt(
    int first_execution_plan_index, const std::vector<int>& execution_plan,
    int* last_execution_plan_index_prepared) {
#ifdef DEBUG 
  SFLAG();
#endif
  if (first_execution_plan_index == 0) {
    has_dynamic_tensors_ = false;
  }
  for (int execution_plan_index = first_execution_plan_index;
       execution_plan_index < execution_plan.size(); execution_plan_index++) {
    int node_index = execution_plan[execution_plan_index];
    TfLiteNode& node = nodes_and_registration_[node_index].first;
    const TfLiteRegistration& registration =
        nodes_and_registration_[node_index].second;
    EnsureTensorsVectorCapacity();
    if (OpPrepare(registration, &node) != kTfLiteOk) {
      return ReportOpError(&context_, node, registration, node_index,
                           "failed to prepare");
    }

    *last_execution_plan_index_prepared = execution_plan_index;

    // Discontinue if the node has dynamic outputs. Note that we don't
    // stop for dynamic temporary tensors since they won't affect the
    // sizes of other tensors in the graph.
    if (HasDynamicTensor(context_, node.outputs)) {
      has_dynamic_tensors_ = true;
		return kTfLiteOk;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Subgraph::PrepareOpsAndTensors() {
#ifdef DEBUG
  SFLAG();
#endif
  if (!memory_planner_) {
    // std::cout << "Reset Memory Planner" << "\n";
    memory_planner_.reset(new ArenaPlanner(
        &context_, std::unique_ptr<GraphInfo>(new InterpreterInfo(this)),
        /*preserve_inputs=*/true, /*preserve_intermediates*/ false,
        kDefaultTensorAlignment));
    memory_planner_->PlanAllocations();
  }

  // Prepare original execution plan if any applied delegate wants it.
  // If any of the delegates is immutable, this won't be triggered
  // post-delegation (since we undo/redo delegation). For all other cases, other
  // delegates that do shape propagation themselves would still be able to.
  bool prepare_original_plan = false;
  if (!pre_delegation_execution_plan_.empty()) {
    for (int i = 0; i < delegates_applied_.size(); ++i) {
      if ((delegates_applied_[i]->flags &
           kTfLiteDelegateFlagsRequirePropagatedShapes)) {
        prepare_original_plan = true;
        break;
      }
    }
  }
  if (prepare_original_plan) {
    int last_original_exec_plan_index_prepared = 0;
    TF_LITE_ENSURE_STATUS(PrepareOpsStartingAt(
        next_execution_plan_index_to_prepare_, pre_delegation_execution_plan_,
        &last_original_exec_plan_index_prepared));
    next_original_execution_plan_index_to_prepare_ =
        last_original_exec_plan_index_prepared + 1;
  }

  int last_exec_plan_index_prepared = 0;
  TF_LITE_ENSURE_STATUS(
      PrepareOpsStartingAt(next_execution_plan_index_to_prepare_,
                           execution_plan_, &last_exec_plan_index_prepared));
  next_execution_plan_index_to_prepare_ = last_exec_plan_index_prepared + 1;

  // std::cout << "next_execution_plan_index_to_plan_allocation_ : "\
                << next_execution_plan_index_to_plan_allocation_ << "\n";
  // std::cout << "last_exec_plan_index_prepared : " \
                << last_exec_plan_index_prepared << "\n";
  // Execute arena allocations.
  TF_LITE_ENSURE_STATUS(memory_planner_->ExecuteAllocations(
      next_execution_plan_index_to_plan_allocation_,
      last_exec_plan_index_prepared));

  // Ensure custom allocations are still valid for applicable tensors.
  // This causes some extra validations for cases with dynamic tensors, but the
  // overhead should be minimal since the number of custom-allocated tensors
  // will typically be low.
  for (int i = 0; i < custom_allocations_.size(); ++i) {
    auto idx_and_alloc = custom_allocations_[i];
    auto& tensor = tensors()[idx_and_alloc.first];
    const auto& alloc = idx_and_alloc.second;
    TF_LITE_ENSURE(context(), tensor.allocation_type == kTfLiteCustom);
    TF_LITE_ENSURE_STATUS(
        ValidateCustomAllocationForTensor(context(), &tensor, alloc));
  }

  next_execution_plan_index_to_plan_allocation_ =
      last_exec_plan_index_prepared + 1; 
  return kTfLiteOk;
}

//HOON : TODO 
TfLiteStatus Subgraph::Invoke(UnitType eType, std::mutex& mtx_lock, 
                            std::mutex& mtx_lock_,
                            std::mutex& mtx_lock_debug,
                            std::condition_variable& Ucontroller,
                            std::queue<SharedContext*>* qSharedData) {
                                       
  //if(qSharedData == nullptr){
  //  ReportError("Got NULLPTR for qSharedData!");
  //  return kTfLiteError;
  //}
  //Minsung
  //Code for DetailedTimeMeasure

  use_detailed_latency_measure = false;
  if(use_detailed_latency_measure && eType == UnitType::GPU0){
    PrepareDetailedLatencyMeasure(4);
  }else if(use_detailed_latency_measure && eType == UnitType::CPU0){
    PrepareDetailedLatencyMeasure(4);
  }
  if (!consistent_) {
    ReportError("Invoke called on model that is not consistent.");
    return kTfLiteError;
  }

  TfLiteStatus status = kTfLiteOk;
  if (state_ == kStateUninvokable) {
    ReportError("Invoke called on model that is not ready.");
	return kTfLiteError;
  } else if (memory_planner_ && !memory_planner_->HasNonPersistentMemory()) {
    ReportError("Non-persistent memory is not available.");
	return kTfLiteError;
  }

  // This is only needed for UseNNAPI(true);
  if (should_apply_nnapi_delegate_ && !applied_nnapi_delegate_) {
    TF_LITE_ENSURE_OK(&context_, ModifyGraphWithDelegate(NnApiDelegate()));
    // only need to modify the graph once upon the first invocation.
    applied_nnapi_delegate_ = true;
  }
  



// --------------------------------------------------------------------------------------------------------------
// HOON : tensor access methods

  // Invocations are always done in node order.
  // Note that calling Invoke repeatedly will cause the original memory plan to
  // be reused, unless either ResizeInputTensor() or AllocateTensors() has been
  // called.
  int final_execution_index = execution_plan_.size()-1;
  std::cout << "\033[0;32m=== Excution plan info ===\033[0m : \n";  
  for (int execution_plan_index = 0;
       execution_plan_index < execution_plan_.size(); execution_plan_index++) {
    //if(eType == UnitType::GPU0)
    //std::cout << "Number of Tensors in current Context : " << context_.tensors_size << "\n";

    if (execution_plan_index == next_execution_plan_index_to_prepare_) {
      TF_LITE_ENSURE_STATUS(PrepareOpsAndTensors());
      TF_LITE_ENSURE(&context_, next_execution_plan_index_to_prepare_ >=
                                    execution_plan_index);
    }
    int node_index = execution_plan_[execution_plan_index];
    std::cout << node_index << " "; // HOON
    TfLiteNode& node = nodes_and_registration_[node_index].first;
    const TfLiteRegistration& registration =
        nodes_and_registration_[node_index].second;

    const char* op_name = nullptr;
    
    if (profiler_) op_name = GetTFLiteOpName(registration);
    TFLITE_SCOPED_TAGGED_OPERATOR_PROFILE(profiler_.get(), op_name, node_index);

    for (int i = 0; i < node.outputs->size; ++i) {
      int tensor_index = node.outputs->data[i];
      // printf("HOON. each node (%d)'s output tensor's index num  :  %d\n",execution_plan_index,tensor_index);
      TfLiteTensor* output_tensor = &context_.tensors[tensor_index]; //HOON  : same tool . indirect
    }

// --------------------------------------------------------------------------------------------------------

    // TODO(ycling): This is an extra loop through inputs to check if the data
    // need to be copied from Delegate buffer to raw memory, which is often not
    // needed. We may want to cache this in prepare to know if this needs to be
    // done for a node or not.

    for (int i = 0; i < node.inputs->size; ++i) {
      int tensor_index = node.inputs->data[i];
      if (tensor_index == kTfLiteOptionalTensor) {
        continue;
      }      // HOON : get tensor's index
    
      //TfLiteTensor* tensor = &context_.tensors[tensor_index]; //HOON  : same tool . indirect
      TfLiteTensor* tensor = &tensors_[tensor_index]; //HOON : same tool. direct. (not via tfitecontext)
      // *tensor.delegate == tensor->delegate
      // in CPU tensor->delegate == nullptr
      // in GPU.. tensor->delegate != nullptr


      
      if (tensor->delegate && tensor->delegate != node.delegate &&
          tensor->data_is_stale) {
        TF_LITE_ENSURE_STATUS(EnsureTensorDataIsReadable(tensor_index));
      }
      if (tensor->data.raw == nullptr && tensor->bytes > 0) {
        if (registration.builtin_code == kTfLiteBuiltinReshape && i == 1) {
          // In general, having a tensor here with no buffer will be an error.
          // However, for the reshape operator, the second input tensor is only
          // used for the shape, not for the data. Thus, null buffer is ok.
          continue;
        } else {
          // In all other cases, we need to return an error as otherwise we will
          // trigger a null pointer dereference (likely).
          ReportError("Input tensor %d lacks data", tensor_index);
    		  return kTfLiteError;
        }
      }
    }
    if (check_cancelled_func_ != nullptr &&
        check_cancelled_func_(cancellation_data_)) {
      ReportError("Client requested cancel during Invoke()");
	  return kTfLiteError;
    }

    // if(node_index == 0){
    //   auto input_pointer = (float *)tensor(0)->data.data;
    //   for (int i=0; i<416; i++){
    //       for (int j=0; j<416; j++){   // j<yolo_size ERROR_Point
    //           std::cout << *(input_pointer + i * 416 + j * 3) << " ";
    //           std::cout << *(input_pointer + i * 416 + j * 3 + 1) << " ";
    //           std::cout << *(input_pointer + i * 416 + j * 3 + 2) << " ";
    //           std::cout << std::endl;
    //       }
    //   }
    //   // std::cout << "\n";
    //   // std::cout << "\n";
    //   // for (int i=0; i<416; i++){
    //   //     for (int j=0; j<416; j++){   // j<yolo_size ERROR_Point
    //   //         std::cout << *(input_pointer + i * 416 + j * 3 + 1) << " ";
    //   //     }
    //   // }
    //   // std::cout << "\n";
    //   // std::cout << "\n";
    //   // for (int i=0; i<416; i++){
    //   //     for (int j=0; j<416; j++){   // j<yolo_size ERROR_Point
    //   //         std::cout << *(input_pointer + i * 416 + j * 3 + 2) << " ";
    //   //     }
    //   // }
    //   // std::cout << "\n";
    //   // std::cout << "\n";
    //   // PrintTensor(*tensor(0), UnitType::CPU0);  // Do not use
    // }

    EnsureTensorsVectorCapacity();
    tensor_resized_since_op_invoke_ = false;
    //=============== INVOKE =============== 
    //=============== INVOKE =============== 
    //=============== INVOKE =============== 
    //=============== INVOKE ===============
    if(use_detailed_latency_measure){
      clock_gettime(CLOCK_MONOTONIC, &(clock_measure_data->time_ary[0]));
    }

    //std::cout << "==================================" << "\n";

    //std::cout << "==================================" << "\n";
    //PrintNodeInfo(node_index, node, registration);
    // PrintInputTensor(node, eType);
    if (OpInvoke(registration, &node) != kTfLiteOk) {	
      return ReportOpError(&context_, node, registration, node_index,
                           "failed to invoke");
    }
    
    //PrintOutputTensor(node, eType); //hoon

    #ifdef debug
    if(eType == UnitType::CPU0){
      std::unique_lock<std::mutex> lock(mtx_lock_debug);
      PrintNodeInfo(node_index, node, registration);
      PrintOutputTensor(node, eType);
    }
    #endif
    
    if(use_detailed_latency_measure){
      clock_gettime(CLOCK_MONOTONIC, &(clock_measure_data->time_ary[1]));
      clock_measure_data->ary[0] += \
        ((clock_measure_data->time_ary[1].tv_sec - clock_measure_data->time_ary[0].tv_sec) + \
        ((clock_measure_data->time_ary[1].tv_nsec - clock_measure_data->time_ary[0].tv_nsec) / 1000000000.0));
    }
    use_distribute_strategy = false; //HH
    if(use_distribute_strategy){
      
      if(use_detailed_latency_measure){
        clock_gettime(CLOCK_MONOTONIC, &(clock_measure_data->time_ary[2]));
      }
    
      if(strcmp(GetOpName(registration), "CONV_2D") == 0 && 
                eType == UnitType::CPU0){ //Call ContextHandler right after Conv 2d
        if(ContextHandler(eType, GetOutputTensor(node), qSharedData, mtx_lock, mtx_lock_,
                        Ucontroller, node_index)
          != kTfLiteOk) {return kTfLiteError;}
      }
      
      if(strcmp(GetOpName(registration), "CONCATENATION") == 0 &&
                eType == UnitType::CPU0){ //Call ContextHandler right after CONCATENATION
        if(CPUPopContextFromQueue(qSharedData, node_index, mtx_lock, mtx_lock_) != kTfLiteOk) 
          {return kTfLiteError;}
      }

      if(strcmp(GetOpName(registration), "CONCATENATION") == 0 && 
                eType == UnitType::GPU0){ //Call ContextHandler right after CONCATENATION
        if(ContextHandler(eType, GetOutputTensor(node), qSharedData, mtx_lock,
                        mtx_lock_, Ucontroller, node_index)
          != kTfLiteOk) {return kTfLiteError;}
      }
      /*
      if(strcmp(GetOpName(registration), "CONV_2D") == 0 && 
                eType == UnitType::CPU0){ //Call ContextHandler right after Conv 2d
        if(ContextHandler(eType, GetOutputTensor(node), qSharedData, mtx_lock, mtx_lock_,
                        Ucontroller, node_index)
          != kTfLiteOk) {return kTfLiteError;}
      }
      
      if(strcmp(GetOpName(registration), "CONV_2D") == 0 &&
                eType == UnitType::CPU0){ //Call ContextHandler right after CONCATENATION
        if(CPUPopContextFromQueue(qSharedData, node_index, mtx_lock, mtx_lock_) != kTfLiteOk) 
          {return kTfLiteError;}
      }

      if(strcmp(GetOpName(registration), "CONV_2D") == 0 && 
                eType == UnitType::GPU0){ //Call ContextHandler right after CONCATENATION
        if(ContextHandler(eType, GetOutputTensor(node), qSharedData, mtx_lock,
                        mtx_lock_, Ucontroller, node_index)
          != kTfLiteOk) {return kTfLiteError;}
      }
      */
    //Print
    
    #ifdef debug
    if(strcmp(GetOpName(registration), "CONCATENATION") == 0 && 
                eType == UnitType::GPU0){
      std::unique_lock<std::mutex> lock(mtx_lock_debug);
      PrintNodeInfo(node_index, node, registration);
      PrintOutputTensor(node, eType);
    }
    #endif
    

      if(use_detailed_latency_measure){
        clock_gettime(CLOCK_MONOTONIC, &(clock_measure_data->time_ary[3]));
        double temp;
          temp = \
        ((clock_measure_data->time_ary[3].tv_sec - clock_measure_data->time_ary[2].tv_sec) + \
        ((clock_measure_data->time_ary[3].tv_nsec - clock_measure_data->time_ary[2].tv_nsec) / 1000000000.0));
        clock_measure_data->ary[1] += temp;
        // if(eType == UnitType::CPU0)
          // printf("temp : %.6f \n", temp);
      }
    }
	  // Force execution prep for downstream ops if the latest op triggered the
    // resize of a dynamic tensor.
    if (tensor_resized_since_op_invoke_ &&
        HasDynamicTensor(context_, node.outputs)) {
      next_execution_plan_index_to_prepare_ = execution_plan_index + 1;

      // This happens when an intermediate dynamic tensor is resized.
      // We don't have vim to prepare all the ops, but we need to recompute
      // the allocation plan.
      if (next_execution_plan_index_to_plan_allocation_ >
          next_execution_plan_index_to_prepare_) {
        next_execution_plan_index_to_plan_allocation_ =
            next_execution_plan_index_to_prepare_;
        if (memory_planner_) {
          TF_LITE_ENSURE_STATUS(memory_planner_->ResetAllocationsAfter(
              next_execution_plan_index_to_plan_allocation_ - 1));
        }
        // std::cout << "ResetAllocationsAfter" << "\n";
      }
    }
    if(number_of_conv_temp <= 0 && eType == UnitType::GPU0 && 
                                                  use_distribute_strategy){
      number_of_conv_temp = number_of_conv;
    }
    if(number_of_conv_temp <= 0 && eType == UnitType::CPU0 && 
                                                  use_distribute_strategy){
      status = kTfLiteOk;
      number_of_conv_temp = number_of_conv;
      return status;
    }
  }
  if(use_detailed_latency_measure){
    if(eType == UnitType::GPU0){
    }
  }
  ////////////////////////////////////////////////////////////////////////////////////////////
  #ifdef YOLO
  YOLO_Parser yolo_parser;
  printf("\033[0;33mStart YOLO parsing\033[0m\n");
  std::vector<int> real_bbox_index_vector;
  real_bbox_index_vector.clear();
  YOLO_Parser::real_bbox_cls_index_vector.clear();
  YOLO_Parser::real_bbox_cls_vector.clear();
  YOLO_Parser::real_bbox_loc_vector.clear();
  YOLO_Parser::result_boxes.clear();
  TfLiteTensor* cls_tensor = tensor(212);
  TfLiteTensor* loc_tensor = tensor(233);
  yolo_parser.make_real_bbox_cls_vector(cls_tensor, real_bbox_index_vector,
                                         YOLO_Parser::real_bbox_cls_vector);
  YOLO_Parser::real_bbox_cls_index_vector = \
              yolo_parser.get_cls_index(YOLO_Parser::real_bbox_cls_vector); 
  yolo_parser.make_real_bbox_loc_vector(loc_tensor, real_bbox_index_vector, 
                                        YOLO_Parser::real_bbox_loc_vector);
  float iou_threshold = 0.5;
  yolo_parser.PerformNMSUsingResults(real_bbox_index_vector, YOLO_Parser::real_bbox_cls_vector, 
        YOLO_Parser::real_bbox_loc_vector, iou_threshold,YOLO_Parser::real_bbox_cls_index_vector);
  printf("\033[0;33mEND YOLO parsing\033[0m\n");
  #endif
  return status;
}


// std::vector<YOLO_Parser::BoundingBox> YOLO_Parser::result_boxes;
// std::vector<std::vector<float>> YOLO_Parser::real_bbox_cls_vector; 
// std::vector<int> YOLO_Parser::real_bbox_cls_index_vector;
// std::vector<std::vector<int>> YOLO_Parser::real_bbox_loc_vector;

// std::vector<int> YOLO_Parser::get_cls_index(std::vector<std::vector<float>>& real_bbox_cls_vector){
//   float max=0;
//   int max_index = -1;
//   int index = 0;
//   for (auto i : real_bbox_cls_vector) { 
//     index = 0;
// 		for (auto j : i) { 
//       if (j > max){
//         max = j;
//         max_index = index;
//       }
//       index+=1;
// 		}
//     real_bbox_cls_index_vector.push_back(max_index);
//     max = 0;
//     max_index = -1;
// 	}
//   return real_bbox_cls_index_vector;
// }

// void YOLO_Parser::make_real_bbox_cls_vector(TfLiteTensor* cls_tensor, 
//  std::vector<int>& real_bbox_index_vector, std::vector<std::vector<float>>& real_bbox_cls_vector){
//   TfLiteTensor* output_tensor = cls_tensor;  
//   const float* output_data = (float*)output_tensor->data.data;
//   const int num_raw_bboxes = output_tensor->dims->data[1]; 
//   std::vector<float> classifications;
//   float cls_thresh = 0.05; // Hyperparam
//   for (int i = 0; i < num_raw_bboxes; ++i) {
//     for (int j = 0; j < 80; ++j) {
//         classifications.push_back(output_data[i*80 + j]);  
//        }
//   }
//   std::vector<float> raw_vector;
//   for (int i = 0; i < num_raw_bboxes; ++i) {
//     bool is_survived = false;
//     for (int j = 0; j < 80; ++j) {
//       raw_vector.push_back(classifications[i * 80 + j]); 
//     }
//     // SOFTMAX(raw_vector); // Not use Softmax currently
//     for (int k = 0; k < 80; ++k) {
//       if (raw_vector[k] > cls_thresh){
//         is_survived = true;
//       }
//     }
//     if(is_survived){
//       real_bbox_index_vector.push_back(i); 
//       real_bbox_cls_vector.push_back(raw_vector);
//     }
//     raw_vector.clear();
//   }
//   classifications.clear();
//   printf("\033[0;32mBefore NMS : \033[0m");
//   std::cout << " Number of bounding boxes before NMS : " << real_bbox_index_vector.size() << std::endl;
// }

// void YOLO_Parser::make_real_bbox_loc_vector(TfLiteTensor* loc_tensor,std::vector<int>& real_bbox_index_vector,
//                                             std::vector<std::vector<int>>& real_bbox_loc_vector){
//   TfLiteTensor* output_tensor = loc_tensor;
//   auto input_pointer = (float *)output_tensor->data.data;
//   const float* output_data = (float*)output_tensor->data.data; 
//   const int num_raw_bboxes = output_tensor->dims->data[1]; 
//   const int num_columns = output_tensor->dims->data[2]; 
//   std::vector<float> boxes;
//   for (int i = 0; i < num_raw_bboxes; ++i) {
//        for (int j = 0; j < num_columns; ++j) {
//           boxes.push_back(output_data[i * 4 + j]);  
//        }
//   }
//   int image_size = 416; 
//   for (int i = 0; i < num_raw_bboxes; ++i) {
//       std::vector<int>tmp;
//       for(int j=0 ; j < real_bbox_index_vector.size(); j++){
//           if(i == real_bbox_index_vector[j]) {
//             float first = boxes[i * 4];      
//             float second = boxes[i * 4 + 1]; 
//             float third = boxes[i * 4 + 2]; 
//             float fourth = boxes[i* 4 + 3];   
//             int left = static_cast<int>(std::max(0.0f, std::min(static_cast<float> 
//             (image_size), first - third/2)));
//             int top = static_cast<int>(std::max(0.0f, std::min(static_cast<float> 
//             (image_size), second - fourth/2)));
//             int right = static_cast<int>(std::max(0.0f, std::min(static_cast<float> 
//             (image_size), first + third/2)));
//             int bottom = static_cast<int>(std::max(0.0f, std::min(static_cast<float> 
//             (image_size), second + fourth/2)));
//             tmp.push_back(left);
//             tmp.push_back(top);
//             tmp.push_back(right);
//             tmp.push_back(bottom);
//             real_bbox_loc_vector.push_back(tmp);
//             break;
//           }
//       }
//       tmp.clear();
//   }
// }
////////////////////////////////////////////////////////////////////////////////////////////



//Minsung
//Overloaded Invoke function for while.cc if.cc ... etc
TfLiteStatus Subgraph::Invoke(UnitType eType){
  std::mutex mtx_lock;
  std::mutex mtx_lock_;
  std::mutex mtx_lock_debug;
  std::condition_variable temp_cond;
  Invoke(eType, mtx_lock, mtx_lock_, mtx_lock_debug, temp_cond, nullptr);
}

TfLiteStatus Subgraph::ResizeTensor(TfLiteContext* context,
                                    TfLiteTensor* tensor,
                                    TfLiteIntArray* new_size) {
  // If the dimensions don't change, avoiding
  // unnecessary (re)allocations.
  //
  // Note that it's required to check `tensor->data.raw != nullptr`. Otherwise
  // the subgraph won't allocate memory for a dynamic tensor when its size
  // is equal to the original tensor size.
  if (tensor->data.raw != nullptr &&
      EqualArrayAndTfLiteIntArray(tensor->dims, new_size->size,
                                  new_size->data)) {
    // A number of clients assume |new_size| remains valid upon success, so
    // swap it in as the new (but logically identical) tensor dims.
    TfLiteIntArrayFree(tensor->dims);
    tensor->dims = new_size;
    return kTfLiteOk;
  }

  // Note here that context->impl_ is recovering the this pointer for an
  // instance of Interpreter to call into the member function ResizeTensorImpl
  // (this function is static).
  return static_cast<Subgraph*>(context->impl_)
      ->ResizeTensorImpl(tensor, new_size);
}

void Subgraph::ReportErrorImpl(const char* format, va_list args) {
  error_reporter_->Report(format, args);
}

void Subgraph::ReportErrorC(TfLiteContext* context, const char* format, ...) {
  va_list args;
  va_start(args, format);
  auto* f = static_cast<Subgraph*>(context->impl_);
  // Note here that context->impl_ is recovering the this pointer for an
  // instance of Subgraph to call into the member function ReportErrorImpl
  // (this function is static).
  f->ReportErrorImpl(format, args);
  va_end(args);
}

// Entry point for C node plugin API to report an error.
void Subgraph::ReportError(const char* format, ...) {
  va_list args;
  va_start(args, format);
  auto* f = static_cast<Subgraph*>(context_.impl_);
  // Note here that context->impl_ is recovering the this pointer for an
  // instance of Subgraph to call into the member function ReportErrorImpl
  // (this function is static).
  f->ReportErrorImpl(format, args);
  va_end(args);
}

TfLiteStatus Subgraph::AddTensors(int tensors_to_add,
                                  int* first_new_tensor_index) {
  #ifdef DEBUG
    SFLAG();
  #endif
  const size_t base_index = tensors_.size();
  if (first_new_tensor_index) *first_new_tensor_index = base_index;
  tensors_.resize(tensors_.size() + tensors_to_add);
  for (size_t i = base_index; i < tensors_.size(); i++) {
    memset(&tensors_[i], 0, sizeof(tensors_[i]));
    tensors_[i].buffer_handle = kTfLiteNullBufferHandle;
  }
  context_.tensors = tensors_.data();
  context_.tensors_size = tensors_.size();
  return kTfLiteOk;
}

TfLiteStatus Subgraph::AddTensors(TfLiteContext* context, int tensors_to_add,
                                  int* first_new_tensor_index) {
  // Note here that context->impl_ is recovering the this pointer for an
  // instance of Interpreter to call into the member function AddTensors
  // (this function is static).
  return static_cast<Subgraph*>(context->impl_)
      ->AddTensors(tensors_to_add, first_new_tensor_index);
}

TfLiteStatus Subgraph::GetNodeAndRegistration(
    int node_index, TfLiteNode** node, TfLiteRegistration** registration) {
  TF_LITE_ENSURE(&context_, node_index >= 0);
  auto nodes_size = nodes_and_registration_.size();
  TF_LITE_ENSURE(&context_, static_cast<size_t>(node_index) < nodes_size);
  TF_LITE_ENSURE(&context_, node != nullptr && registration != nullptr);
  auto& node_and_reg = nodes_and_registration_[node_index];
  *node = &node_and_reg.first;
  *registration = &node_and_reg.second;
  return kTfLiteOk;
}

TfLiteStatus Subgraph::GetNodeAndRegistration(
    struct TfLiteContext* context, int node_index, TfLiteNode** node,
    TfLiteRegistration** registration) {
  return static_cast<Subgraph*>(context->impl_)
      ->GetNodeAndRegistration(node_index, node, registration);
}

TfLiteStatus Subgraph::SetTensorParametersReadOnly(
    int tensor_index, TfLiteType type, const char* name, const size_t rank,
    const int* dims, TfLiteQuantization quantization, const char* buffer,
    size_t bytes, const Allocation* allocation, TfLiteSparsity* sparsity) {
  // Ensure quantization cleanup on failure.
  ScopedTfLiteQuantization scoped_quantization(&quantization);
  ScopedTfLiteSparsity scoped_sparsity(sparsity);
  if (state_ == kStateInvokableAndImmutable) {
    ReportError(
        "SetTensorParametersReadOnly is disallowed when graph is immutable.");
    return kTfLiteError;
  }

  TF_LITE_ENSURE(&context_,
                 tensor_index < context_.tensors_size && tensor_index >= 0);

  // For most tensors we know exactly how much memory is necessary so we can
  // ensure the buffer is large enough. However, we need to skip string tensors
  // and sparse tensors because their sizes change with the contents.
  // TODO(b/145615516): Extend BytesRequired to check sparse tensors.
  if (type != kTfLiteString && sparsity == nullptr) {
    size_t required_bytes;
    // std::cout << "SetTensorParametersReadOnly::BytesRequired" << "\n";
    TF_LITE_ENSURE_OK(&context_,
                      BytesRequired(type, dims, rank, &required_bytes));
    TF_LITE_ENSURE_EQ(&context_, required_bytes, bytes);
  }

  TfLiteTensor& tensor = context_.tensors[tensor_index];
  if (type == tensor.type &&
      EqualArrayAndTfLiteIntArray(tensor.dims, rank, dims)) {
    // Fast path which does not invalidate the invokable property.
    TfLiteTensorDataFree(&tensor);
    TfLiteQuantizationFree(&tensor.quantization);
    tensor.data.raw = const_cast<char*>(buffer);
    if (!tensor.dims) tensor.dims = ConvertArrayToTfLiteIntArray(rank, dims);
    tensor.params = GetLegacyQuantization(quantization);
    tensor.quantization = *scoped_quantization.release();
    tensor.sparsity = scoped_sparsity.release();
    tensor.allocation_type = kTfLiteMmapRo;
    tensor.allocation = allocation;
  } else {
    state_ = kStateUninvokable;
    TfLiteTensorReset(type, name, ConvertArrayToTfLiteIntArray(rank, dims),
                      GetLegacyQuantization(quantization),
                      const_cast<char*>(buffer), bytes, kTfLiteMmapRo,
                      allocation, false, &tensor);
    // TODO(suharshs): Update TfLiteTensorReset to include the new quantization
    // if there are other required callers.
    tensor.quantization = *scoped_quantization.release();
    tensor.sparsity = scoped_sparsity.release();
  }
  return kTfLiteOk;
}

// Set description of inputs/outputs/data/fptrs for node `node_index`.
// This variant assumes an external buffer has been allocated of size
// bytes. The lifetime of buffer must be ensured to be greater or equal
// to Interpreter.
TfLiteStatus Subgraph::SetTensorParametersReadWrite(
    int tensor_index, TfLiteType type, const char* name, const size_t rank,
    const int* dims, TfLiteQuantization quantization, bool is_variable,
    const size_t rank_dims_signature, const int* dims_signature) {
  // Ensure quantization cleanup on failure.
  ScopedTfLiteQuantization scoped_quantization(&quantization);
  if (state_ == kStateInvokableAndImmutable) {
    ReportError(
        "SetTensorParametersReadWrite is disallowed when graph is immutable.");
    return kTfLiteError;
  }
  TF_LITE_ENSURE(&context_,
                 tensor_index < context_.tensors_size && tensor_index >= 0);
  size_t required_bytes = 0;
  if (type != kTfLiteString) {
    // These types will be allocated in our arena so we need to record how
    // many bytes we will need based on the dimensions. String tensors are
    // allocated dynamically and we can't know ahead of time how much space
    // they will require.
    TF_LITE_ENSURE_OK(&context_,
                      BytesRequired(type, dims, rank, &required_bytes));
  }

  TfLiteAllocationType allocation_type = kTfLiteArenaRw;
  if (type == kTfLiteString) {
    if (is_variable) {
      // We don't have a real use case for string variable tensor.
      ReportError("String variable tensor isn't supported.");
      return kTfLiteError;
    }
    allocation_type = kTfLiteDynamic;
  } else if (is_variable) {
    allocation_type = kTfLiteArenaRwPersistent;
  }

  TfLiteTensor& tensor = context_.tensors[tensor_index];
  TfLiteTensorReset(type, name, ConvertArrayToTfLiteIntArray(rank, dims),
                    GetLegacyQuantization(quantization),
                    /*buffer=*/nullptr, required_bytes, allocation_type,
                    nullptr, is_variable, &tensor);
  // TODO(suharshs): Update TfLiteTensorReset to include the new quantization
  // if there are other required callers.
  tensor.quantization = *scoped_quantization.release();
  tensor.dims_signature =
      ConvertArrayToTfLiteIntArray(rank_dims_signature, dims_signature);
  return kTfLiteOk;
}

TfLiteStatus Subgraph::SetExecutionPlan(const std::vector<int>& new_plan) {
  for (int node_index : new_plan) {
    TF_LITE_ENSURE(&context_, node_index >= 0 &&
                                  node_index < nodes_and_registration_.size());
  }
  execution_plan_ = new_plan;
  return kTfLiteOk;
}

TfLiteStatus Subgraph::ResizeTensorImpl(TfLiteTensor* tensor,
                                        TfLiteIntArray* new_size) {
  // Note that in theory we could resize kTfLiteArenaRwPersistent tensors too.
  if (tensor->allocation_type == kTfLiteArenaRw ||
      tensor->allocation_type == kTfLiteDynamic ||
      tensor->allocation_type == kTfLiteArenaRwPersistent ||
      tensor->allocation_type == kTfLitePersistentRo ||
      tensor->allocation_type == kTfLiteCustom) {
    tensor_resized_since_op_invoke_ |=
        TfLiteIntArrayEqual(tensor->dims, new_size) == 0;
    if (tensor->type != kTfLiteString) {
      size_t bytesRequired;
      // std::cout << "BytesRequired" << "\n";
      TfLiteStatus status = BytesRequired(tensor->type, new_size->data,
                                          new_size->size, &bytesRequired);
      if (status != kTfLiteOk) {
        TfLiteIntArrayFree(new_size);
        return kTfLiteError;
      }

      // Realloc space for heap-allocated tensors.
      TfLiteTensorRealloc(bytesRequired, tensor);
      tensor->bytes = bytesRequired;
    }
    if (tensor->dims) TfLiteIntArrayFree(tensor->dims);
    tensor->dims = new_size;

    // Reset arena-allocated tensors; they will be allocated later.
    if (tensor->allocation_type == kTfLiteArenaRw ||
        tensor->allocation_type == kTfLiteArenaRwPersistent) {
      tensor->data.raw = nullptr;
    }
  } else {
    // kTfLiteMmapRo tensors are stored in the flatbuffer and are therefore
    // of fixed size.
    TfLiteIntArrayFree(new_size);
    ReportError("Attempting to resize a fixed-size tensor.");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

void Subgraph::UseNNAPI(bool enable) {
  // Note that there is no way to disable the delegate once it modified the
  // graph.
  if (applied_nnapi_delegate_ && !enable) {
    ReportError("Attempting to disable NNAPI delegate after it's applied.");
  } else {
    should_apply_nnapi_delegate_ = enable;
  }
}

void Subgraph::SwitchToDelegateContext() {
#ifdef DEBUG
  SFLAG();
#endif
  context_.GetNodeAndRegistration = GetNodeAndRegistration;
  context_.ReplaceNodeSubsetsWithDelegateKernels =
      ReplaceNodeSubsetsWithDelegateKernels;
  context_.GetExecutionPlan = GetExecutionPlan;
  context_.PreviewDelegatePartitioning = PreviewDelegatePartitioning;
}

void Subgraph::SwitchToKernelContext() {
  context_.GetNodeAndRegistration = [](struct TfLiteContext* context,
                                       int node_index, TfLiteNode** node,
                                       TfLiteRegistration** registration) {
    return ForbiddenContextFunction(context);
  };
  context_.ReplaceNodeSubsetsWithDelegateKernels =
      [](TfLiteContext* context, TfLiteRegistration registration,
         const TfLiteIntArray* nodes_to_replace, TfLiteDelegate* delegate) {
        return ForbiddenContextFunction(context);
      };
  context_.GetExecutionPlan = [](struct TfLiteContext* context,
                                 TfLiteIntArray**) {
    return ForbiddenContextFunction(context);
  };
  context_.PreviewDelegatePartitioning =
      [](struct TfLiteContext* context, const TfLiteIntArray* nodes_to_replace,
         TfLiteDelegateParams** partition_params_array,
         int* num_partitions) { return ForbiddenContextFunction(context); };
  // Free any memory that might have been allocated by
  // PreviewDelegatePartitioning.
  FreeDelegatePartitioningData();
}

TfLiteStatus Subgraph::UndoAllDelegates() {
  // Return early if there is nothing to reset to.
  if (pre_delegation_execution_plan_.empty()) return kTfLiteOk;

  // First free all delegate nodes.
  for (int execution_plan_index = 0;
       execution_plan_index < execution_plan_.size(); ++execution_plan_index) {
    int node_index = execution_plan_[execution_plan_index];
    TfLiteNode& node = nodes_and_registration_[node_index].first;
    if (node.delegate == nullptr) {
      continue;
    }
    CleanupNode(node_index);
  }

  // Reset execution plan.
  execution_plan_ = pre_delegation_execution_plan_;
  pre_delegation_execution_plan_.clear();

  // Handling FP16 delegation (if applies).
  //
  // First pass through execution plan to remember mapping of FP16
  // dequantizations in the graph.
  // This is required because delegates that support FP16 could remap supported
  // nodes' inputs to point to their fp16 versions (if delegate supports fp16
  // acceleration). This remapping is performed in FP16GraphPartitionHelper in
  // delegates/utils. We need to undo this remapping to ensure CPU kernels work.
  std::vector<int> fp16_to_fp32(tensors_size(), -1);
  for (int execution_plan_index = 0;
       execution_plan_index < execution_plan_.size(); ++execution_plan_index) {
    int node_index = execution_plan_[execution_plan_index];
    auto& node_and_reg = nodes_and_registration_[node_index];
    const TfLiteNode& node = node_and_reg.first;
    const TfLiteRegistration& reg = node_and_reg.second;
    if (reg.builtin_code == kTfLiteBuiltinDequantize &&
        node.inputs->size == 1 && node.outputs->size == 1) {
      const int input_idx = node.inputs->data[0];
      if (tensors_[input_idx].type == kTfLiteFloat16) {
        fp16_to_fp32[input_idx] = node.outputs->data[0];
      }
    }
  }
  // Second pass through the execution plan to remap applicable nodes' fp16
  // inputs to their original fp32 versions. Note that if a CPU kernel does
  // support fp16, the model will not contain a DEQUANTIZE for its constant
  // input.
  for (int execution_plan_index = 0;
       execution_plan_index < execution_plan_.size(); ++execution_plan_index) {
    int node_index = execution_plan_[execution_plan_index];
    auto& node_and_reg = nodes_and_registration_[node_index];
    const TfLiteNode& node = node_and_reg.first;
    const TfLiteRegistration& reg = node_and_reg.second;
    if (reg.builtin_code == kTfLiteBuiltinDequantize) continue;
    for (int i = 0; i < node.inputs->size; ++i) {
      const int original_input_idx = node.inputs->data[i];
      if (tensors_[original_input_idx].type == kTfLiteFloat16) {
        node.inputs->data[i] = fp16_to_fp32[original_input_idx];
      }
    }
  }

  // Delegate nodes are appended to nodes_and_registration_. Therefore,
  // cleanup nodes_and_registration_ to only contain nodes from
  // pre_delegation_execution_plan_.
  int max_retained_node_index = 0;
  for (int execution_plan_index = 0;
       execution_plan_index < execution_plan_.size(); ++execution_plan_index) {
    max_retained_node_index = std::max(max_retained_node_index,
                                       execution_plan_[execution_plan_index]);
  }
  nodes_and_registration_.resize(max_retained_node_index + 1);
  // After undoing delegates, the graph is uninvokable, but mutable.
  state_ = kStateUninvokable;

  delegates_undone_ = true;
  return kTfLiteOk;
}

TfLiteStatus Subgraph::RedoAllDelegates() {
  //std::cout << "tensorflow/lite/core/subgraph.cc/Subgraph::RedoAllDelegates()\n";
  if (!delegates_undone_) return kTfLiteOk;

  delegates_undone_ = false;
  std::vector<TfLiteDelegate*> delegates_to_apply;
  delegates_applied_.swap(delegates_to_apply);
  for (auto* delegate : delegates_to_apply) {
    TF_LITE_ENSURE_STATUS(ModifyGraphWithDelegate(delegate));
  }
  return kTfLiteOk;
}

TfLiteStatus Subgraph::RemoveAllDelegates() {
  TF_LITE_ENSURE_STATUS(UndoAllDelegates());
  delegates_applied_.clear();
  delegates_undone_ = false;
  TF_LITE_ENSURE_STATUS(EnsureMemoryAllocations());
  return kTfLiteOk;
}

bool Subgraph::HasDelegates() { return !delegates_applied_.empty(); }

void Subgraph::EnsureTensorsVectorCapacity() {
  const size_t required_capacity = tensors_.size() + kTensorsCapacityHeadroom;
  if (required_capacity > tensors_.capacity()) {
    // Whenever it's required to increase the vector capacity, make it at
    // least twice bigger. The behavior is consistent with the default
    // behavior of GCC STL's `std::vector::resize()`. This avoids frequently
    // allocating and copying the underlying buffer.
    size_t reserved_capacity =
        std::max(required_capacity, tensors_.capacity() * 2);
    tensors_.reserve(reserved_capacity);
    context_.tensors = tensors_.data();
  }
}

TfLiteStatus Subgraph::EnsureMemoryAllocations() {
  if (memory_planner_) {
    state_ = kStateUninvokable;
    TF_LITE_ENSURE_OK(&context_, memory_planner_->PlanAllocations());
  }
  TF_LITE_ENSURE_OK(&context_, AllocateTensors());
  TF_LITE_ENSURE_EQ(&context_, state_, kStateInvokable);
  return kTfLiteOk;
}

TfLiteStatus Subgraph::ModifyGraphWithDelegate(TfLiteDelegate* delegate) {
  TFLITE_SCOPED_TAGGED_DEFAULT_PROFILE(profiler_.get(),
                                       "ModifyGraphWithDelegate");

  // Restore delegation state if applicable.
  TF_LITE_ENSURE_STATUS(RedoAllDelegates());

  if (state_ == kStateInvokableAndImmutable) {
    ReportError(
        "ModifyGraphWithDelegate is disallowed when graph is immutable.");
	  return kTfLiteApplicationError;
  }
  // std::cout << "ModifyGraphWithDelegate logic (func) start " << "\n";
  if (!(delegate->flags & kTfLiteDelegateFlagsAllowDynamicTensors)) {
    int last_execution_plan_index_prepared;
    // Runtime Filter Modification for CPU&GPU Multithreading
    if(use_distribute_strategy){
      for (int node_index = 0;
        node_index < nodes_and_registration_.size(); node_index++) {
        TfLiteNode& node = nodes_and_registration_[node_index].first;
        const TfLiteRegistration& registration =
            nodes_and_registration_[node_index].second;
        int tensor_filter = 0;
        int tensor_bias = 0;
        if(!strcmp(GetOpName(registration), "CONV_2D")){
          tensor_filter = node.inputs->data[1];
          tensor_bias = node.inputs->data[2];
          conv_filter_before_modification =
                context_.tensors[tensor_filter].dims->data[0];
          int modified_value = 
                ceil(conv_filter_before_modification*((float)partitioning_plan/10));
          context_.tensors[tensor_filter].dims->data[0] = modified_value;
          context_.tensors[tensor_bias].dims->data[0] = modified_value;
          int modified_bytes = 1 * sizeof(float);
          for(int i=0; i<4; i++){
            modified_bytes *= context_.tensors[tensor_filter].dims->data[i];
          }
          context_.tensors[tensor_filter].bytes = modified_bytes;
          context_.tensors[tensor_bias].bytes = modified_value * sizeof(float);
        }
        else if(!strcmp(GetOpName(registration), "CONCATENATION")){
          if(conv_filter_before_modification <= 0){
            std::cout << "Error in filter Partitioning \n";
            return kTfLiteError;
          }
          tensor_filter = node.inputs->data[1];
          int modified_value =  conv_filter_before_modification - \
                ceil(conv_filter_before_modification*((float)partitioning_plan/10));
          TfLiteIntArray* ary = TfLiteIntArrayCreate(4);
          for(int i=0; i<4; i++){
            if(i==3){
              ary->data[i] = context_.tensors[tensor_filter].dims->data[i] + \
                              modified_value;
            }
            else{
              ary->data[i] = context_.tensors[tensor_filter].dims->data[i];
            }
          }
          SetTensorToDynamic(tensor(tensor_filter));
          ResizeTensorImpl(tensor(tensor_filter), ary);
        }
      }
    }
    state_ = kStateInvokable;

    // std::cout << "prepare_1" << "\n";
    std::cout << "Execution Plan Size : " << execution_plan_.size() << "\n";
    
    TF_LITE_ENSURE_OK(
        &context_, PrepareOpsStartingAt(0, execution_plan_,
                                        &last_execution_plan_index_prepared));
    if (has_dynamic_tensors_) {
      // Make sure that we are in a defined ready state before returning.
      // Plan and allocate tensors before returning.
      TF_LITE_ENSURE_OK(&context_, EnsureMemoryAllocations());
      ReportError(
          "Attempting to use a delegate that only supports static-sized "
          "tensors with a graph that has dynamic-sized tensors.");
		return kTfLiteApplicationError;
    }
  }
  const bool was_invokable_before_delegate = state_ == kStateInvokable;
  if (delegates_applied_.empty()) {
    // This is the first delegate being applied, so remember original execution
    // plan.
    // TODO(b/119623453): Restore execution plan to this state if delegate
    // application fails.
    pre_delegation_execution_plan_ = execution_plan_;
  }
  // TODO(aselle): Consider if it is worth storing pointers to delegates.
  // Setup additional context interface.
  // printf("HOON : Start to switch to delegate context\n");
  SwitchToDelegateContext();   // HOON : main code flow // just mapping func pointer for DELEGATE
  auto reset_delegation_if_not_ok = [this](TfLiteStatus status) {
    if (status != kTfLiteOk) {
      TF_LITE_ENSURE_STATUS(RemoveAllDelegates());
      ReportError(
          "Restored original execution plan after delegate application "
          "failure."); 
      return kTfLiteDelegateError;
    } 
    return kTfLiteOk;
  };
  // std::cout << "prepare_2" << "\n";
  // printf("HOON : delegate prepare start\n");
  // printf("<<------------------------------------------------------------------------------------>>\n");
  TfLiteStatus status = delegate->Prepare(&context_, delegate); // HOON  : DELEGATE prepare logic  
  // Remove additional context info.
  // printf("<<------------------------------------------------------------------------------------>>\n");
  // printf("HOON : delegate prepare end \n");
  // printf("HOON : Start to switch to kernel context\n");
  SwitchToKernelContext();
  TF_LITE_ENSURE_STATUS(reset_delegation_if_not_ok(status));
  if (!(delegate->flags & kTfLiteDelegateFlagsAllowDynamicTensors)) {
    // Reset the state to force tensor/op reallocation.
    state_ = kStateUninvokable;
    TF_LITE_ENSURE_STATUS(
        reset_delegation_if_not_ok(EnsureMemoryAllocations()));
    // After using a delegate which doesn't support dynamic tensors, make the
    // entire graph immutable.
    state_ = kStateInvokableAndImmutable;
  } else if (was_invokable_before_delegate) {
    // If the graph was invokable prior to delegate application, flush
    // allocation now to leave it in a consistent state.
    TF_LITE_ENSURE_STATUS(
        reset_delegation_if_not_ok(EnsureMemoryAllocations()));
  }
  delegates_applied_.push_back(delegate);
  return status;
}

TfLiteStatus Subgraph::SetCustomAllocationForTensor(
    int tensor_index, const TfLiteCustomAllocation& allocation) {
  TfLiteTensor* tensor = &context_.tensors[tensor_index];
  TF_LITE_ENSURE(context(),
                 (tensor->allocation_type == kTfLiteArenaRw ||
                  tensor->allocation_type == kTfLiteArenaRwPersistent ||
                  tensor->allocation_type == kTfLiteCustom));
  TF_LITE_ENSURE_STATUS(
      ValidateCustomAllocationForTensor(context(), tensor, allocation));

  // If tensor already has a custom alloc, just reassign.
  const auto alloc_it = std::find_if(
      custom_allocations_.begin(), custom_allocations_.end(),
      [tensor_index](
          const std::pair<int, TfLiteCustomAllocation>& existing_alloc) {
        return existing_alloc.first == tensor_index;
      });
  if (alloc_it == custom_allocations_.end()) {
    custom_allocations_.emplace_back(tensor_index, allocation);
  } else {
    alloc_it->second = allocation;
  }

  tensor->allocation_type = kTfLiteCustom;
  tensor->data.data = allocation.data;

  return kTfLiteOk;
}

void Subgraph::PrepareDetailedLatencyMeasure(int num_part){
  clock_measure_data = CreateClockMeasure(num_part);
  use_detailed_latency_measure = true;
  for(int i=0; i<clock_measure_data->size; i++){
    clock_measure_data->ary[i] = 0;
  }
}


void Subgraph::PrintNodeInfo(int node_index, TfLiteNode& node,
                     const TfLiteRegistration& registration){
  std::cout << "\n" << "[Print Node Info]" << "\n";
  std::cout << "OP Name : " << GetTFLiteOpName(registration) << "\n";
  std::cout << "Node Index : " << node_index << "\n";
  std::cout << "Tensor Data type : " << tensor(node.outputs->data[0])->type << "\n";
  std::cout << "Input Tensors : ";
  for(int i=0; i<node.inputs->size; ++i){
    std::cout << node.inputs->data[i] << " "; 
  }
  std::cout << "\n";
  std::cout << "OutputTensors : ";
  for(int i=0; i<node.outputs->size; ++i){
    std::cout << node.outputs->data[i] << " ";
  }
  std::cout << "\n";
  int tensor_index = node.outputs->data[node.outputs->size-1]; //output tensor index
  std::cout << "[" << tensor_index 
            << "] Tensor Size : " << tensor(tensor_index)->bytes << "\n";
  std::cout << "[" << tensor_index << "] Tensor Dimension : ";
  int tensor_data_size = 1;
  int tensor_data_dims_size = tensor(tensor_index)->dims->size-1;
  int tensor_data_ch_size = tensor(tensor_index)->dims->data[tensor_data_dims_size];
  for(int i=0; i< tensor(tensor_index)->dims->size; i++){
    std::cout << tensor(tensor_index)->dims->data[i] << " ";  //print dimension info
    tensor_data_size *= tensor(tensor_index)->dims->data[i]; 
  }
  std::cout << "\n";
}

void Subgraph::PrintInputTensor(TfLiteNode& node, UnitType eType){
  std::cout << "[Print Input Tensor] \n";
  //TfLiteTensor* temp = GetInputTensor(node);
  int tensor_index;
  tensor_index = node.inputs->data[0];  //HOON : Debugging for YOLO input image 
  // if(node.inputs->size == 3)
    // tensor_index = node.inputs->data[0];
  // else
    // tensor_index = node.inputs->data[1];
  TfLiteTensor* temp =  tensor(tensor_index);
  std::cout << "tensor_index is : " << tensor_index << std::endl;
  std::cout << "Possible Input Tensors : ";
  for(int i=0; i<node.inputs->size; ++i){
    std::cout << node.inputs->data[i] << " "; 
  }
  std::cout << "\n";
  int tensor_data_dims_size = temp->dims->size-1;
  int tensor_data_ch_size = temp->dims->data[tensor_data_dims_size];
  int tensor_data_size = 1;
  int tensor_axis;
  for(int i=0; i< temp->dims->size; i++){
    if(i == 1){
      tensor_axis = temp->dims->data[i];
    }
    tensor_data_size *= temp->dims->data[i]; 
  }
  std::cout << "[" << tensor_index << "] Nunber of Tensors : "\
                                           << tensor_data_size << "\n";
  std::cout << "[" << tensor_index << "] Tensor DATA " << "\n";
  std::cout << "[" << tensor_index << "] Tensor Dimension" << " ";
  for(int i=0; i< tensor(tensor_index)->dims->size; i++){
    std::cout << tensor(tensor_index)->dims->data[i] << " ";  //print dimension info
    tensor_data_size *= tensor(tensor_index)->dims->data[i]; 
  }
  std::cout << "\n";
  PrintTensor(*temp, eType);  
}


void Subgraph::PrintOutputTensor(TfLiteNode& node, UnitType eType){
  std::cout << "[Print OutPut Tensor] \n";
  TfLiteTensor* temp = GetOutputTensor(node);
  int tensor_index = GetOutputTensorIndex(node);
  int tensor_data_dims_size = temp->dims->size-1;
  int tensor_data_ch_size = temp->dims->data[tensor_data_dims_size];
  int tensor_data_size = 1;
  int tensor_axis;
  for(int i=0; i< temp->dims->size; i++){
    if(i == 1){
      tensor_axis = temp->dims->data[i];
    }
    tensor_data_size *= temp->dims->data[i]; 
  }
  std::cout << "\n";
  std::cout << "[" << tensor_index << "] Nunber of Tensors : "\
                                           << tensor_data_size << "\n";
  std::cout << "[" << tensor_index << "] Tensor DATA " << "\n";

  PrintTensor(*temp, eType);  
}

void Subgraph::PrintTensor(TfLiteTensor& tensor, UnitType eType){
    std::cout << "[Print Tensor]" << "\n";
  int tensor_channel_idx = tensor.dims->size-1;
  int tensor_data_ch_size = tensor.dims->data[tensor_channel_idx];
  int tensor_data_size = 1;
  int tensor_axis;
  for(int i=0; i< tensor.dims->size; i++){
    if(i == 2){
      tensor_axis = tensor.dims->data[i];
    }
    tensor_data_size *= tensor.dims->data[i]; 
  }
  std::cout << " Number of data : " << tensor_data_size << "\n";
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

//   std::cout << "[Print Tensor]" << "\n";
//   int tensor_data_dims_size = tensor.dims->size-1;
//   std::cout << "tensor_dims_size" << tensor.dims->size << std::endl;
//   int tensor_data_ch_size = tensor.dims->data[tensor_data_dims_size];
//   int tensor_data_size = 1;
//   int tensor_axis;
//   for(int i=0; i< tensor.dims->size; i++){
//     if(i == 1){
//       tensor_axis = tensor.dims->data[i];
//     }
//     tensor_data_size *= tensor.dims->data[i]; 
//   }
//   std::cout << " Nunber of data : " << tensor_data_size << "\n";
//   std::cout << " Tensor DATA " << "\n";
//   if(tensor.type == TfLiteType::kTfLiteFloat32){
//     std::cout << "[FLOAT32 TENSOR]" << "\n";
//     auto data_st = (float*)tensor.data.data;
//     for(int i=0; i<tensor_data_ch_size; i++){
//       std::cout << "CH [" << i << "] \n";
//       for(int j=0; j<tensor_data_size/tensor_data_ch_size; j++){
//         float data = *(data_st+(i+j*tensor_data_ch_size));
//         if (data == 0) {
//           printf("%0.6f ", data);
//         }
//         else if (data != 0) {
//           if(eType == UnitType::CPU0)
//             printf("%s%0.6f%s ", C_GREN, data, C_NRML);
//           else if(eType == UnitType::GPU0)
//             printf("%s%0.6f%s ", C_YLLW, data, C_NRML);
//         }
//         if (j % tensor_axis == tensor_axis-1) {
//           printf("\n");
//         }
//       }
//       std::cout << "\n";
//     }
//   }
//   else if(tensor.type == TfLiteType::kTfLiteInt8){
//     std::cout << "[INT8 TENSOR]" << "\n";
//     auto data_st = (int8_t*)tensor.data.data;
//     for(int i=0; i<tensor_data_ch_size; i++){
//       std::cout << "CH [" << i << "] \n";
//       for(int j=0; j<tensor_data_size/tensor_data_ch_size; j++){
//         int8_t data = *(data_st+(i+j*tensor_data_ch_size));
//         if (data == 0) {
//           printf("%d ", data);
//         }
//         else if (data != 0) {
//           if(eType == UnitType::CPU0)
//             printf("%s%d%s ", C_GREN, data, C_NRML);
//           else if(eType == UnitType::GPU0)
//             printf("%s%d%s ", C_YLLW, data, C_NRML);
//         }
//         if (j % tensor_axis == tensor_axis-1) {
//           printf("\n");
//         }
//       }
//     }
//     std::cout << "\n";
//   }
//   else if(tensor.type == TfLiteType::kTfLiteInt32){
//     std::cout << "[INT32 TENSOR]" << "\n";
//     auto data_st = (int*)tensor.data.data;
//     for(int i=0; i<tensor_data_ch_size; i++){
//       std::cout << "CH [" << i << "] \n";
//       for(int j=0; j<tensor_data_size/tensor_data_ch_size; j++){
//         int data = *(data_st+(i+j*tensor_data_ch_size));
//         if (data == 0) {
//           printf("%d ", data);
//         }
//         else if (data != 0) {
//           if(eType == UnitType::CPU0)
//             printf("%s%d%s ", C_GREN, data, C_NRML);
//           else if(eType == UnitType::GPU0)
//             printf("%s%d%s ", C_YLLW, data, C_NRML);
//         }
//         if (j % tensor_axis == tensor_axis-1) {
//           printf("\n");
//         }
//       }
//       std::cout << "\n";
//     }
//   }
// }

void Subgraph::PrintOutputTensorOfSubgraph(UnitType eType){
  int node_index = execution_plan_[execution_plan_.size()-1];
  TfLiteNode& node = nodes_and_registration_[node_index].first;
  PrintOutputTensor(node, eType);
}

TfLiteStatus Subgraph::PrepareTensorsSharing(UnitType eType){
  if(eType == UnitType::CPU0){ 
  }
  return kTfLiteOk;
}

//Minsung
//ContextHandler Controls Invoking Conv2d Layer.
//When executionplan Invokes Conv2d node,
//After Invoke, (Slave)ContextHandler will push one output tensor pointer to queue
//After Invoke, (Master)ContextHandler will pop output tensor pointer from queue
//(Master)ContextHandler will concat the tensor before invoking next node
TfLiteStatus Subgraph::ContextHandler(UnitType eType, TfLiteTensor* tensor,
                                    std::queue<SharedContext*>* qSharedData,
                                    std::mutex& mtx_lock, 
                                    std::mutex& mtx_lock_,
                                    std::condition_variable& Ucontroller,
                                    int execution_plan_index){
  if(eType == UnitType::CPU0){
    //std::cout << "Slave ContextHandler Called" << "\n";
    // HOON : typedef struct sharedcontext {typedef struct tflitetensor* tensor, Unittype etype}
    // HOON : slaveData->tensor refers to "CPU interpreter"'s CONV output tensor 
    // HOON : then push it to qsharedData
    SharedContext* slaveData = CreateSharedContext(eType, tensor);
    if(PushContextToQueue(slaveData, mtx_lock, mtx_lock_,
                           qSharedData, Ucontroller) != kTfLiteOk){
        return kTfLiteError;
    }
    number_of_conv_temp--;
    return kTfLiteOk;
  }
  else if(eType == UnitType::GPU0){
    //std::cout << "Master ContextHandler Called" << "\n";
    if(ConcatContext(tensor, execution_plan_index, Ucontroller, 
                      mtx_lock, mtx_lock_, qSharedData,
                      GPUPopContextFromQueue(qSharedData, mtx_lock, mtx_lock_))
       != kTfLiteOk){
        return kTfLiteError;
    }
    number_of_conv_temp--;
    return kTfLiteOk;
  }
  //TODO : After GPU recieving & concating are done, GPU sends tensor back to CPU
  //       And the CPU uses the tensor as input of next layer.
  //       Than, CPU don't have to re-write data from gpu
  //       (This way should be more effective than writing data back from GPU)   
}

// Minsung
int Subgraph::GetInputsInMultipleSubgraphs(){
  return inputs()[inputs().size()-1];
}

// Minsung
// Returns vector of all input tensor's idxs of current subgraph
// RETURNS ONLY FIRST EXCUTION PLAN's INPUT
std::vector<int>& Subgraph::GetMultipleInputTensorIdx(){
  std::vector<int>* input = new std::vector<int>;
  int node_index = execution_plan_[0];
  TfLiteNode& node = nodes_and_registration_[node_index].first;
  for(int i=0; i<node.inputs->size; ++i){
    input->push_back(node.inputs->data[i]);
  }
  return *input;
  // Needs Impl..
}

// Minsung
// Quantize Selected Tensor of called Interpreter
// This is an initial process
// FLoat32 -> Int8
//  <How-It-Works>
// Get Tensor.
// Allocate New int8 data array which has same dim with Original one. 
// Quantize Values from Original Tensor and allocate to New one.
// Swap Tensor data pointer to quantized one.
void Subgraph::QuantizeSymFloats(const float* values, const int size,
                                     int8_t* quantized_values, float* min_value,
                                     float* max_value, float* scaling_factor){
  auto minmax = std::minmax_element(values, values + size);
  *min_value = *minmax.first;
  *max_value = *minmax.second;
  QuantizeSymFloatsMain(values, size, quantized_values, *min_value,
                                  *max_value, scaling_factor);
}

void Subgraph::QuantizeSymFloatsMain(const float* values, const int size,
                                     int8_t* quantized_values, float min_value,
                                     float max_value, float* scaling_factor){
  const int32_t kScale = 127;
  const float range = std::max(std::abs(min_value), std::abs(max_value));
  if (range == 0) { //means given array is zero
    memset(quantized_values, 0, size * sizeof(int8_t));
    *scaling_factor = 1;
    return;
  }
  *scaling_factor = range / kScale;
  const float scaling_factor_inv = kScale / range;
  for (int i = 0; i < size; ++i) {
    const int32_t quantized_value =
        static_cast<int32_t>(TfLiteRound(values[i] * scaling_factor_inv));
    // Clamp: just in case some odd numeric offset.
    quantized_values[i] = static_cast<int8_t>(
        std::min(kScale, std::max(-kScale, quantized_value)));
  }
}

TfLiteStatus Subgraph::QuantizeSelectedTensor(TfLiteTensor* tensor){
  TfLiteTensor* working_tensor = tensor;
  working_tensor->allocation_type = kTfLiteDynamic;
  int tensor_data_dims_size = working_tensor->dims->size-1; 
  int tensor_data_ch_size = working_tensor->dims->data[tensor_data_dims_size];
  int tensor_data_size = 1;
  int tensor_axis;
  for(int i=0; i<working_tensor->dims->size; ++i){
    tensor_data_size *= working_tensor->dims->data[i]; 
  }
  //Initial process done.
  //Do quantization process
  //And save quantization info to TfLiteAffineQuantization in tensor.
  int8_t* quantized_values = (int8_t*)malloc(tensor_data_size);
  auto data_st_origin_float = (float*)working_tensor->data.data;
  float* scaling_factors = new float;
  int32_t* zero_points = new int32_t;
  QuantizeFloats(data_st_origin_float, 1, tensor_data_size, quantized_values,
                          scaling_factors, zero_points, false);
  working_tensor->type = TfLiteType::kTfLiteInt8;
  working_tensor->data.data = quantized_values;
  working_tensor->bytes = tensor_data_size;
  TfLiteQuantizationParams* quant_params = new TfLiteQuantizationParams;
  quant_params->scale = *scaling_factors;
  quant_params->zero_point = *zero_points;
  working_tensor->params.scale = *scaling_factors;
  working_tensor->params.zero_point = *zero_points;
  working_tensor->quantization.params = &quant_params;
  working_tensor->quantization.type = TfLiteQuantizationType::kTfLiteAffineQuantization;
  //PrintTensor(*working_tensor, UnitType::CPU0);
  return kTfLiteOk;
}

TfLiteStatus Subgraph::DequantizeSelectedTensor(TfLiteTensor* tensor){
  std::cout << "Dequnatize \n";
  TfLiteTensor* working_tensor = tensor;
  if(working_tensor->quantization.type != kTfLiteAffineQuantization &&\
      working_tensor->type != kTfLiteInt8){
    std::cout << "Dequantization Tensor Type Error \n";
    return kTfLiteError;
  }
  //if(working_tensor->allocation_type != kTfLiteDynamic || 
  //  working_tensor->quantization.type != kTfLiteAffineQuantization){
  //  return kTfLiteError;
  //}
  int tensor_data_dims_size = working_tensor->dims->size-1; 
  int tensor_data_ch_size = working_tensor->dims->data[tensor_data_dims_size];
  int tensor_data_size = 1;
  int tensor_axis;
  for(int i=0; i<working_tensor->dims->size; i++){
    tensor_data_size *= working_tensor->dims->data[i]; 
  }
  auto data_st_origin = (int8_t*)tensor->data.data;
  auto dequantized_values = (float*)malloc(tensor_data_size * sizeof(float));
  float scaling_factor = 
        ((TfLiteQuantizationParams *)(working_tensor->quantization.params))->scale;
  float zero_point = 
        ((TfLiteQuantizationParams *)(working_tensor->quantization.params))->zero_point;
  printf("scaling factor : %0.18f \n", scaling_factor);
  printf("zero point : %0.18f \n", zero_point);
  std::cout << "tensor data byte : " << working_tensor->bytes << "\n";
  std::cout << "tensor data size : " << tensor_data_size << "\n";
  for(int i=0; i<tensor_data_size; ++i){
    float temp = data_st_origin[i] * scaling_factor;
    printf("tensor data idx %d \n", i);
    //printf("tensor data : %d\n", data_st_origin[i]);
    dequantized_values[i] = temp;
  }
  working_tensor->type = TfLiteType::kTfLiteFloat32;
  working_tensor->data.data = dequantized_values;
  working_tensor->bytes = tensor_data_size * sizeof(float);
  working_tensor->params.scale = NULL;
  working_tensor->params.zero_point = NULL;
  working_tensor->quantization.params = NULL;
  working_tensor->quantization.type = TfLiteQuantizationType::kTfLiteNoQuantization;
  std::cout << "Dequnatize Done\n";
  return kTfLiteOk;
}

TfLiteStatus Subgraph::QuantizeCurrentSubgraph(){
  conv_node_index.push_back(0);
  conv_node_index.push_back(3);
  conv_node_index.push_back(6);
  QuantizeSelectedTensor(tensor(0)); 
  QuantizeSelectedTensor(tensor(15)); 
  for(int i = 0; i<conv_node_index.size(); ++i){
    TfLiteNode node = nodes_and_registration_[conv_node_index[i]].first;
    std::vector<TfLiteTensor*> weight_bias_tensors;
    weight_bias_tensors.push_back(tensor(node.inputs->data[0]));
    weight_bias_tensors.push_back(tensor(node.inputs->data[1]));
    weight_bias_tensors.push_back(tensor(node.inputs->data[2]));
    weight_bias_tensors.push_back(tensor(node.inputs->data[3]));
    for(int j=0; j<weight_bias_tensors.size(); j++){
      //Initial process for quantization.
      //Get size and dim info of Original Tensor.
      if(QuantizeSelectedTensor(weight_bias_tensors[j]) != kTfLiteOk)
        return kTfLiteError;
    }  
  }
  std::cout << "Quantization Complete \n";
  return kTfLiteOk;
}


//Concate CPU Tensor Context and GPU Tensor Context in Concat Layer
TfLiteStatus Subgraph::ConcatContext(TfLiteTensor* rc_tensor, 
                                int execution_plan_index,
                                std::condition_variable& Ucontroller,
                                std::mutex& mtx_lock,
                                std::mutex& mtx_lock_,
                                std::queue<SharedContext*>* qSharedData,
                                SharedContext* SlaveData){
  //rc : recieve
  //sd : send
  //st : start
  //cp : copy
  //ch : channel
  int concat_tensor_index = \
            nodes_and_registration_[execution_plan_index].first.inputs->data[1];
  int concat_tensor_filter = \
            tensor(concat_tensor_index)->dims->data[3];    
  TfLiteTensor* sd_tensor = SlaveData->tensor;
  int tensor_rc_data_ch_index = rc_tensor->dims->size-1;
  int tensor_rc_ch_size = \
            rc_tensor->dims->data[tensor_rc_data_ch_index] - concat_tensor_filter;
  int tensor_sd_data_ch_index = sd_tensor->dims->size-1;
  int tensor_sd_ch_size = sd_tensor->dims->data[tensor_sd_data_ch_index];
  int tensor_data_size = 1; 
  for(int i=0; i< rc_tensor->dims->size; i++){   //get number of tensor data to copy
                                              // ex)26x26 -> 676
    tensor_data_size *= rc_tensor->dims->data[i]; 
  }
  auto data_send = (float*)sd_tensor->data.data;    //data send pointer
  auto data_recieve = (float*)rc_tensor->data.data;  //data recieve pointer
  int ch_size = tensor_rc_ch_size + concat_tensor_filter; 
  //int ch_st = (ch_size * (0.1 * partitioning_plan));
  int ch_st = tensor_rc_ch_size;
  int tensor_data_per_ch = tensor_data_size / ch_size;
  for(int n=0; n<tensor_data_per_ch; n++){
    memcpy((data_recieve + (ch_st + n*ch_size)),
          (data_send + (n * tensor_sd_ch_size)), sizeof(float) * tensor_sd_ch_size);
  }
  if(!(number_of_conv_temp <= 1)){ //this needs to be modified
    SharedContext* newSharedContext = new SharedContext;
    newSharedContext->eType = UnitType::GPU0;
    newSharedContext->tensor = rc_tensor;
    std::unique_lock<std::mutex> lock(mtx_lock);
    qSharedData->push(newSharedContext);
  }
  Ucontroller.notify_one(); //HOON --> wake up algo solved by lock::condition_variable
  return kTfLiteOk;
} 


TfLiteStatus Subgraph::PushContextToQueue(SharedContext* slaveData,
                                  std::mutex& mtx_lock,
                                  std::mutex& mtx_lock_,
                                  std::queue<SharedContext*>* qSharedData,
                                  std::condition_variable& Ucontroller){
  if(slaveData != nullptr){
    std::unique_lock<std::mutex> lock(mtx_lock);
    qSharedData->push(slaveData);
    mtx_lock_.unlock();
    Ucontroller.wait(lock);
    return kTfLiteOk;
  }
  return kTfLiteError;
}

SharedContext* Subgraph::GPUPopContextFromQueue(std::queue<SharedContext*>* qSharedData,
                                            std::mutex& mtx_lock, std::mutex& mtx_lock_){
  SharedContext* temp;
  mtx_lock_.lock();
  std::unique_lock<std::mutex> lock(mtx_lock);
  if(qSharedData->empty()){
    std::cout << "QUEUE ERROR \n";
    //exit(1);
  }
  temp = qSharedData->front();
  qSharedData->pop();
  return temp;
}

TfLiteStatus Subgraph::CPUPopContextFromQueue(std::queue<SharedContext*>* qSharedData,
                                            int execution_plan_index,
                                            std::mutex& mtx_lock,
                                            std::mutex& mtx_lock_){
  int output_tensor_index = \
              nodes_and_registration_[execution_plan_index].first.outputs->data[0];
  std::unique_lock<std::mutex> lock(mtx_lock);
  if(qSharedData->empty()){
    std::cout << "Oh Yeah!! Welcome to error world!! lollo!!!!" << "\n";
    return kTfLiteError;
  }
  // HOON : shared tensor is first forked by "CPU CONV's output tensor"
  // HOON : so just update original tensor data 
  context_.tensors[output_tensor_index].data.data = \
                                    qSharedData->front()->tensor->data.data;
  qSharedData->pop();
  return kTfLiteOk;
}


SharedContext* Subgraph::CreateSharedContext(UnitType eType,
                                         TfLiteTensor* tensor){
  //PrintTensor(*tensor, UnitType::CPU0);
  //DequantizeSelectedTensor(tensor);
  //PrintTensor(*tensor, UnitType::CPU0);
  return new SharedContext{eType, tensor};
}

//Check number of Conv2d Layer & Node index
TfLiteStatus Subgraph::CheckConv2dNodes(){
  std::cout << "nodes size : " << nodes_and_registration_.size() << "\n";
  for (int node_index = 0;
    node_index < nodes_and_registration_.size(); node_index++) {
    const TfLiteRegistration& registration =
        nodes_and_registration_[node_index].second;
    if(strcmp(GetOpName(registration), "CONV_2D") == 0){
      number_of_conv++;
      conv_node_index.push_back(node_index);
    }
  }
  if(number_of_conv >= 1){
    number_of_conv_temp = number_of_conv;
    return kTfLiteOk;
  }
  else  
    return kTfLiteError;
}

const char* Subgraph::GetFirstOpName(){
  if(nodes_and_registration_.size() < 1){
    return "NO_OP";
  }
  const TfLiteRegistration& registration = 
      nodes_and_registration_[0].second;
  return GetOpName(registration);         
}

TfLiteStatus Subgraph::SwitchTensor(TfLiteTensor& tensor, int idx){
  context_.tensors[idx] = tensor;
}

std::vector<int> Subgraph::GetOutputShape(){
  int final_node = execution_plan_.size() - 1;
  TfLiteNode& node = nodes_and_registration_[final_node].first;
  int output_tensor = node.outputs->data[0];
  TfLiteTensor* tensor = &tensors_[output_tensor];
  std::vector<int> output_dims;
  for(int i=0; i<tensor->dims->size; ++i){
    output_dims.push_back(tensor->dims->data[i]);
  }
  return output_dims;
}

std::vector<int> Subgraph::GetTensorShape(int tensor_index){
  TfLiteTensor* tensor = &tensors_[tensor_index];
  std::vector<int> dims;
  for(int i=0; i<tensor->dims->size; ++i){
    dims.push_back(tensor->dims->data[i]);
  }
  return dims;  
}

int Subgraph::GetOutputTensorIndex(){
  int final_node = execution_plan_.size() - 1;
  TfLiteNode& node = nodes_and_registration_[final_node].first;
  return node.outputs->data[0];
}

}  // namespace tflite