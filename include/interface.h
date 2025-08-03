
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
#endif

#endif // INTERFACE_h
