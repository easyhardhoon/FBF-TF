/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/evaluation/stages/object_detection_stage.h"

#include <fstream>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace tflite {
namespace evaluation {

TfLiteStatus ObjectDetectionStage::Init(
    const DelegateProviders* delegate_providers) {
  // Ensure inference params are provided.
  if (!config_.specification().has_object_detection_params()) {
    LOG(ERROR) << "ObjectDetectionParams not provided";
    return kTfLiteError;
  }
  auto& params = config_.specification().object_detection_params();
  if (!params.has_inference_params()) {
    LOG(ERROR) << "inference_params not provided";
    return kTfLiteError;
  }
  if (all_labels_ == nullptr) {
    LOG(ERROR) << "Detection output labels not provided";
    return kTfLiteError;
  }

  // TfliteInferenceStage.
  EvaluationStageConfig tflite_inference_config;
  tflite_inference_config.set_name("tflite_inference");
  *tflite_inference_config.mutable_specification()
       ->mutable_tflite_inference_params() = params.inference_params();
  inference_stage_.reset(new TfliteInferenceStage(tflite_inference_config));

  // HOON : MAIN stage (model build, create interpreter & subgraph, modify subgraph)
  std::cout << "Go to tflite_inference_stage::init" << std::endl;
  TF_LITE_ENSURE_STATUS(inference_stage_->Init(delegate_providers));

  // Validate model inputs.
  const TfLiteModelInfo* model_info = inference_stage_->GetModelInfo();
  if (model_info->inputs.size() != 1 || model_info->outputs.size() != 4) {
    LOG(ERROR) << "Object detection model must have 1 input & 4 outputs";
    // return kTfLiteError;
  }
  TfLiteType input_type = model_info->inputs[0]->type;
  auto* input_shape = model_info->inputs[0]->dims;
  // Input should be of the shape {1, height, width, 3}
  if (input_shape->size != 4 || input_shape->data[0] != 1 ||
      input_shape->data[3] != 3) {
    LOG(ERROR) << "Invalid input shape for model";
    return kTfLiteError;
  }

  // ImagePreprocessingStage
  tflite::evaluation::ImagePreprocessingConfigBuilder builder(
      "image_preprocessing", input_type);
  builder.AddResizingStep(input_shape->data[2], input_shape->data[1], false);
  builder.AddDefaultNormalizationStep();
  preprocessing_stage_.reset(new ImagePreprocessingStage(builder.build()));
  TF_LITE_ENSURE_STATUS(preprocessing_stage_->Init());

  // ObjectDetectionAveragePrecisionStage
  EvaluationStageConfig eval_config;
  eval_config.set_name("average_precision");
  *eval_config.mutable_specification()
       ->mutable_object_detection_average_precision_params() =
      params.ap_params();
  eval_config.mutable_specification()
      ->mutable_object_detection_average_precision_params()
      ->set_num_classes(all_labels_->size());
  eval_stage_.reset(new ObjectDetectionAveragePrecisionStage(eval_config));
  TF_LITE_ENSURE_STATUS(eval_stage_->Init());

  return kTfLiteOk;
}

//----------------------------------------------------------------
TfLiteStatus ObjectDetectionStage::Run() {
  std::cout << "Count : " << counter << std::endl;
  counter +=1;
  if (image_path_.empty()) {
    LOG(ERROR) << "Input image not set";
    return kTfLiteError;
  }

  // Preprocessing.
  preprocessing_stage_->SetImagePath(&image_path_);
  TF_LITE_ENSURE_STATUS(preprocessing_stage_->Run());


  
  // Inference.
  std::vector<void*> data_ptrs = {};
  data_ptrs.push_back(preprocessing_stage_->GetPreprocessedImageData());
  inference_stage_->SetInputs(data_ptrs);
  std::cout << "\033[0;33m1.invoke\033[0m\n";
  TF_LITE_ENSURE_STATUS(inference_stage_->Run());

  std::cout  << "\033[0;33m2.output parsing\033[0m\n";
  // HOONING
  std::cout << "TODO point(change yolo's output tensor 2->4)\n";
  // Convert model output to ObjectsSet.
  predicted_objects_.Clear();
  const int class_offset =
      config_.specification().object_detection_params().class_offset();
  const std::vector<void*>* outputs = inference_stage_->GetOutputs();
  int num_detections = static_cast<int>(*static_cast<float*>(outputs->at(3)));
  // int num_detections = 3;
  float* detected_label_boxes = static_cast<float*>(outputs->at(0));
  float* detected_label_indices = static_cast<float*>(outputs->at(1));
  float* detected_label_probabilities = static_cast<float*>(outputs->at(2));
  // float* detected_label_probabilities = SOFTMAX(detected_label_indices);
  for (int i = 0; i < num_detections; ++i) {
    const int bounding_box_offset = i * 4;
    auto* object = predicted_objects_.add_objects();
    // Bounding box
    auto* bbox = object->mutable_bounding_box();
    bbox->set_normalized_top(detected_label_boxes[bounding_box_offset + 0]);
    bbox->set_normalized_left(detected_label_boxes[bounding_box_offset + 1]);
    bbox->set_normalized_bottom(detected_label_boxes[bounding_box_offset + 2]);
    bbox->set_normalized_right(detected_label_boxes[bounding_box_offset + 3]);
    // Class.
    object->set_class_id(static_cast<int>(detected_label_indices[i]) +
                         class_offset);
    // Score
    object->set_score(detected_label_probabilities[i]);
  }

  // AP Evaluation.
  eval_stage_->SetEvalInputs(predicted_objects_, *ground_truth_objects_);
  std::cout  << "\033[0;33m3.evaluate each data\033[0m\n";
  TF_LITE_ENSURE_STATUS(eval_stage_->Run());
  return kTfLiteOk;
}
//----------------------------------------------------------------


void ObjectDetectionStage::SOFTMAX(std::vector<std::vector<float>>& real_bbox_cls_vector){
  printf("\033[0;33mAfter SOFTMAX :\033[0m \n");
  const float threshold = 0.999999; 
  for (auto& row : real_bbox_cls_vector) {
      float maxElement = *std::max_element(row.begin(), row.end());
      float sum = 0.0;
      const float scalingFactor = 20.0;
      for (auto& i : row)
          sum += std::exp(scalingFactor * (i - maxElement));
      for (int i = 0; i < row.size(); ++i) {
          row[i] = std::exp(scalingFactor * (row[i] - maxElement)) / sum;
          if (row[i] > threshold)
              row[i] = threshold; 
      }
  }
  for (auto i : real_bbox_cls_vector) {
      for (auto j : i) {
          if (j > 0.15) {
              printf("\033[0;31m%.6f\033[0m ", j);
          }
          else {
              printf("%.6f ", j);
          }
      }
      std::cout << std::endl << std::endl;
  }
}


int ObjectDetectionStage::counter = 0;

TfLiteStatus ObjectDetectionStage::Run_hoon() {
  std::cout << "Count : " << counter << std::endl;
  counter +=1;
  if (image_path_.empty()) {
    LOG(ERROR) << "Input image not set";
    return kTfLiteError;
  }

  // Preprocessing.
  preprocessing_stage_->SetImagePath(&image_path_);
  TF_LITE_ENSURE_STATUS(preprocessing_stage_->Run());

  // Inference.
  std::vector<void*> data_ptrs = {};
  data_ptrs.push_back(preprocessing_stage_->GetPreprocessedImageData());
  inference_stage_->SetInputs(data_ptrs);
  TF_LITE_ENSURE_STATUS(inference_stage_->Run());


  // HOONING 
  // Convert model output to ObjectsSet. 
  predicted_objects_.Clear();
  const int class_offset =
      config_.specification().object_detection_params().class_offset();
  const std::vector<void*>* outputs = inference_stage_->GetOutputs();
  // TODO POINT : how to get num_detections
  int num_detections = static_cast<int>(*static_cast<float*>(outputs->at(1)));
  float* detected_label_boxes = static_cast<float*>(outputs->at(0));
  float* detected_label_scores = static_cast<float*>(outputs->at(1));
  for (int i = 0; i < num_detections; ++i) {
    const int bounding_box_offset = i * 4;
    auto* object = predicted_objects_.add_objects();
    // Bounding box
    auto* bbox = object->mutable_bounding_box();
    bbox->set_normalized_top(detected_label_boxes[bounding_box_offset + 0]);
    bbox->set_normalized_left(detected_label_boxes[bounding_box_offset + 1]);
    bbox->set_normalized_bottom(detected_label_boxes[bounding_box_offset + 2]);
    bbox->set_normalized_right(detected_label_boxes[bounding_box_offset + 3]);
    // Score
    object->set_score(detected_label_scores[i]);
    // Class ID can be set if available in the output tensors or from config.
  }

  // AP Evaluation.
  eval_stage_->SetEvalInputs(predicted_objects_, *ground_truth_objects_);
  TF_LITE_ENSURE_STATUS(eval_stage_->Run());

  return kTfLiteOk;
}

EvaluationStageMetrics ObjectDetectionStage::LatestMetrics() {
  EvaluationStageMetrics metrics;
  auto* detection_metrics =
      metrics.mutable_process_metrics()->mutable_object_detection_metrics();

  *detection_metrics->mutable_pre_processing_latency() =
      preprocessing_stage_->LatestMetrics().process_metrics().total_latency();
  EvaluationStageMetrics inference_metrics = inference_stage_->LatestMetrics();
  *detection_metrics->mutable_inference_latency() =
      inference_metrics.process_metrics().total_latency();
  *detection_metrics->mutable_inference_metrics() =
      inference_metrics.process_metrics().tflite_inference_metrics();
  *detection_metrics->mutable_average_precision_metrics() =
      eval_stage_->LatestMetrics()
          .process_metrics()
          .object_detection_average_precision_metrics();
  metrics.set_num_runs(inference_metrics.num_runs());
  return metrics;
}

TfLiteStatus PopulateGroundTruth(
    const std::string& grouth_truth_proto_file,
    absl::flat_hash_map<std::string, ObjectDetectionResult>*
        ground_truth_mapping) {
  if (ground_truth_mapping == nullptr) {
    return kTfLiteError;
  }
  ground_truth_mapping->clear();

  // Read the ground truth dump.
  std::ifstream t(grouth_truth_proto_file);
  std::string proto_str((std::istreambuf_iterator<char>(t)),
                        std::istreambuf_iterator<char>());
  ObjectDetectionGroundTruth ground_truth_proto;
  ground_truth_proto.ParseFromString(proto_str);

  for (const auto& image_ground_truth :
       ground_truth_proto.detection_results()) {
    (*ground_truth_mapping)[image_ground_truth.image_name()] =
        image_ground_truth;
  }

  return kTfLiteOk;
}

}  // namespace evaluation
}  // namespace tflite
