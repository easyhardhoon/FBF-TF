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

// This file provides general C++ utility functions in TFLite.
// For example: Converting between `TfLiteIntArray`, `std::vector` and
// Flatbuffer vectors. These functions can't live in `context.h` since it's pure
// C.

#ifndef TENSORFLOW_LITE_UTIL_H_
#define TENSORFLOW_LITE_UTIL_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/c/common.h"

namespace tflite {

// Memory allocation parameter used by ArenaPlanner.
// Clients (such as delegates) might look at this to ensure interop between
// TFLite memory & hardware buffers.
// NOTE: This only holds for tensors allocated on the arena.
constexpr int kDefaultTensorAlignment = 64;

// The prefix of Flex op custom code.
// This will be matched agains the `custom_code` field in `OperatorCode`
// Flatbuffer Table.
// WARNING: This is an experimental API and subject to change.
constexpr char kFlexCustomCodePrefix[] = "Flex";

// Checks whether the prefix of the custom name indicates the operation is an
// Flex operation.
bool IsFlexOp(const char* custom_name);

// Converts a `std::vector` to a `TfLiteIntArray`. The caller takes ownership
// of the returned pointer.
TfLiteIntArray* ConvertVectorToTfLiteIntArray(const std::vector<int>& input);

// Converts an array (of the given size) to a `TfLiteIntArray`. The caller
// takes ownership of the returned pointer, and must make sure 'dims' has at
// least 'rank' elements.
TfLiteIntArray* ConvertArrayToTfLiteIntArray(const int rank, const int* dims);

// Checks whether a `TfLiteIntArray` and an int array have matching elements.
// The caller must guarantee that 'b' has at least 'b_size' elements.
bool EqualArrayAndTfLiteIntArray(const TfLiteIntArray* a, const int b_size,
                                 const int* b);

size_t CombineHashes(std::initializer_list<size_t> hashes);

struct TfLiteIntArrayDeleter {
  void operator()(TfLiteIntArray* a) {
    if (a) TfLiteIntArrayFree(a);
  }
};

// Helper for Building TfLiteIntArray that is wrapped in a unique_ptr,
// So that it is automatically freed when it goes out of the scope.
std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> BuildTfLiteIntArray(
    const std::vector<int>& data);

// Populates the size in bytes of a type into `bytes`. Returns kTfLiteOk for
// valid types, and kTfLiteError otherwise.
TfLiteStatus GetSizeOfType(TfLiteContext* context, const TfLiteType type,
                           size_t* bytes);

// Creates a stub TfLiteRegistration instance with the provided
// `custom_op_name`. The op will fail if invoked, and is useful as a
// placeholder to defer op resolution.
// Note that `custom_op_name` must remain valid for the returned op's lifetime..
TfLiteRegistration CreateUnresolvedCustomOp(const char* custom_op_name);

// Checks whether the provided op is an unresolved custom op.
bool IsUnresolvedCustomOp(const TfLiteRegistration& registration);

// Returns a descriptive name with the given op TfLiteRegistration.
std::string GetOpNameByRegistration(const TfLiteRegistration& registration);

////////////////////////////////////////////////////////////////////
// HOON : utility funcs for parsing Yolo output
// class YOLO_Parser{
//   public:
//     YOLO_Parser();
//     ~YOLO_Parser();
//     static std::vector<std::vector<float>> real_bbox_cls_vector; 
//     static std::vector<int> real_bbox_cls_index_vector;
//     static std::vector<std::vector<int>> real_bbox_loc_vector;
//     std::vector<int> get_cls_index(std::vector<std::vector<float>>& real_bbox_cls_vector);
//     void make_real_bbox_cls_vector(TfLiteTensor* cls_tensor, std::vector<int>& real_bbox_index_vector, std::vector<std::vector<float>>& real_bbox_cls_vector);
//     void make_real_bbox_loc_vector(TfLiteTensor* loc_tensor, std::vector<int>& real_bbox_index_vector,std::vector<std::vector<int>>& real_bbox_loc_vector);
//     void SOFTMAX(std::vector<float>& real_bbox_cls_vector);
//     struct BoundingBox {
//       float left, top, right, bottom;
//       float score;
//       int class_id;
//     };
//     static std::vector<YOLO_Parser::BoundingBox> result_boxes;
//     static bool CompareBoxesByScore(const BoundingBox& box1, const BoundingBox& box2);
//     float CalculateIoU(const BoundingBox& box1, const BoundingBox& box2);
//     void NonMaximumSuppression(std::vector<BoundingBox>& boxes, float iou_threshold);
//     void PerformNMSUsingResults(
//     const std::vector<int>& real_bbox_index_vector,
//     const std::vector<std::vector<float>>& real_bbox_cls_vector,
//     const std::vector<std::vector<int>>& real_bbox_loc_vector,
//     float iou_threshold, const std::vector<int> real_bbox_cls_index_vector);
// }; 
////////////////////////////////////////////////////////////////////



}  // namespace tflite

#endif  // TENSORFLOW_LITE_UTIL_H_
