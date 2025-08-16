
#ifndef INTERFACE_H
#define INTERFACE_H

#include <pipewire/pipewire.h>
#include <spa/param/video/format-utils.h>

#pragma once
extern bool enable_profiling;

#ifdef __cplusplus
extern "C" {
#endif
  // void detectObjects(cv::Mat& img, std::vector<BoundingBox>& bBoxes, float confThreshold, float nmsThreshold,
  //                    const std::string& basePath, const std::string& classesFile, cv::dnn::Net& net, bool bVis);
  typedef void (*detection_callback_t)(struct pw_buffer* buffer,
                                      struct impl* impl,
                                      bool success);
  
  void detectObjects(struct pw_buffer *out_buffer,
                     float confThreshold,
                     float nmsThreshold,
                     const char* basePath,
                     const char* classesFile,
                     uint32_t frame_width,
                     uint32_t frame_height,
                     bool bVis);

  void detectObjects_async(struct pw_buffer *out_buffer,
                          float confThreshold,
                          float nmsThreshold,
                          const char* basePath,
                          const char* classesFile,
                          uint32_t frame_width,
                          uint32_t frame_height,
                          bool bVis,
                          detection_callback_t callback,
                          struct impl* user_data);

#ifdef __cplusplus
}

// C++ only gRPC function declarations
#include "dataStructures.h"
#include <string>

bool init_grpc_client(const std::string& server_address = "localhost:50051");

// Send detection via gRPC and return control message
// Returns: 0=continue, 1=switch, 2=stop, -1=error
int send_detection_grpc(const BoundingBox& bbox, const std::string& label,
                        float confidence, int class_id);

// Get latest server response (non-blocking, doesn't close stream)
// Returns: 0=continue, 1=switch, 2=stop, -1=error
int check_stream_response_grpc();

void cleanup_grpc_client();
std::string get_current_model_grpc();
void reset_grpc_client();
void shutdown_detection_with_grpc();

// Function to re-initialize neural network (to be called from objectDetection2D.cc)
bool reinitialize_nn_with_model_switch(const std::string& basePath);

// Force reinitialization of neural network (for model switching)
bool reinitialize_nn_force(const std::string& model_path);

#endif

#endif // INTERFACE_h
