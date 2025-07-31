// Based on: https://github.com/Geekgineer/YOLOs-CPP/blob/main/include/det/YOLO11.hpp
// Adapted for use in cam_infer_models project

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include "dataStructures.h"

#include "interface.h"

#include <thread>
#include <functional>

using namespace std;

// Structure to pass data to async thread
struct AsyncDetectionData {
    struct pw_buffer* buffer;
    float confThreshold;
    float nmsThreshold;
    std::string basePath;
    std::string classesFile;
    uint32_t frame_width;
    uint32_t frame_height;
    bool bVis;
    detection_callback_t callback;
    struct impl* user_data;
};


// Utility function for letterboxing
cv::Mat letterBox(const cv::Mat& image, const cv::Size& newShape, 
                  const cv::Scalar& color = cv::Scalar(114, 114, 114)) {
    // Calculate the scaling ratio to fit the image within the new shape
  float ratio = std::min(static_cast<float>(newShape.height) / static_cast<float>(image.rows),
                          static_cast<float>(newShape.width) / static_cast<float>(image.cols));

    // Calculate new dimensions after scaling
    int newUnpadW = static_cast<int>(std::round(static_cast<float>(image.cols) * ratio));
    int newUnpadH = static_cast<int>(std::round(static_cast<float>(image.rows) * ratio));

    // Calculate padding needed to reach the desired shape
    int dw = newShape.width - newUnpadW;
    int dh = newShape.height - newUnpadH;

    // Evenly distribute padding on both sides
    int padLeft = dw / 2;
    int padRight = dw - padLeft;
    int padTop = dh / 2;
    int padBottom = dh - padTop;

    cv::Mat resizedImage;
    // Resize the image if the new dimensions differ
    if (image.cols != newUnpadW || image.rows != newUnpadH) {
        cv::resize(image, resizedImage, cv::Size(newUnpadW, newUnpadH), 0, 0, cv::INTER_LINEAR);
    } else {
        resizedImage = image;
    }

    cv::Mat paddedImage;
    // Apply padding to reach the desired shape
    cv::copyMakeBorder(resizedImage, paddedImage, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, color);
    
    return paddedImage;
}

// Utility function to scale coordinates back to original image
cv::Rect scaleCoords(const cv::Size& imgShape, const cv::Rect& coords, 
                     const cv::Size& imgOriginalShape) {
    float gain = std::min(static_cast<float>(imgShape.height) / static_cast<float>(imgOriginalShape.height),
                         static_cast<float>(imgShape.width) / static_cast<float>(imgOriginalShape.width));

    int padX = static_cast<int>(std::round((static_cast<float>(imgShape.width) - static_cast<float>(imgOriginalShape.width) * gain) / 2.0f));
    int padY = static_cast<int>(std::round((static_cast<float>(imgShape.height) - static_cast<float>(imgOriginalShape.height) * gain) / 2.0f));

    cv::Rect result;
    result.x = static_cast<int>(std::round(static_cast<float>(coords.x - padX) / gain));
    result.y = static_cast<int>(std::round(static_cast<float>(coords.y - padY) / gain));
    result.width = static_cast<int>(std::round(static_cast<float>(coords.width) / gain));
    result.height = static_cast<int>(std::round(static_cast<float>(coords.height) / gain));

    // Clamp coordinates to image boundaries
    result.x = std::max(0, std::min(result.x, imgOriginalShape.width));
    result.y = std::max(0, std::min(result.y, imgOriginalShape.height));
    result.width = std::max(0, std::min(result.width, imgOriginalShape.width - result.x));
    result.height = std::max(0, std::min(result.height, imgOriginalShape.height - result.y));

    return result;
}
cv::Mat pwbuffer_to_cvmat(struct pw_buffer* buf, uint32_t frame_width, uint32_t frame_height) {
  if (!buf || buf->buffer->n_datas == 0) return {};
  struct spa_data* spa_data = &buf->buffer->datas[0];
  uint8_t* data = static_cast<uint8_t*>(spa_data->data);

  // Step 1: Wrap raw YUY2 buffer in cv::Mat
  cv::Mat yuy2_frame(frame_height, frame_width, CV_8UC2, data);

  // Step 2: Convert to RGB
  cv::Mat rgb_frame;
  cv::cvtColor(yuy2_frame, rgb_frame, cv::COLOR_YUV2BGR_YUY2);

  // Step 3: Display or use rgb_frame
  // cv::imshow("RGB Frame", rgb_frame);
  // cv::waitKey(1);
  return rgb_frame;
}

void cvmat_to_pwbuffer(cv::Mat visImg, struct pw_buffer* buf, uint32_t frame_width, uint32_t frame_height) {
  if (!buf || visImg.empty()) return;

  cv::Mat yuy2_frame;
  cv::resize(visImg, visImg, cv::Size(frame_width, frame_height));
  cv::cvtColor(visImg, yuy2_frame, cv::COLOR_BGR2YUV_YUY2);

  struct spa_data* spa_data = &buf->buffer->datas[0];
  auto* data = static_cast<uint8_t*>(spa_data->data);

  const size_t bytes_to_copy = frame_width * frame_height * 2;

  if (yuy2_frame.total() * yuy2_frame.elemSize() >= bytes_to_copy && spa_data->maxsize >= bytes_to_copy) {
    memcpy(data, yuy2_frame.data, bytes_to_copy);
  }
}


void detectObjects(struct pw_buffer *out_buffer,
                   float confThreshold,
                   float nmsThreshold,
                   const char* basePath,
                   const char* classesFile,
                   uint32_t frame_width,
                   uint32_t frame_height,
                   bool bVis)
{
    string yoloModelWeights = std::string(basePath) + "yolo11s.onnx";
    // Load class names from file
    vector<string> classes;
    ifstream ifs(std::string(classesFile).c_str());
    string line;
    while (getline(ifs, line))
      classes.push_back(line);

    cv::Mat img = pwbuffer_to_cvmat(out_buffer, frame_width, frame_height);
    constexpr int inputSize = 640;
    cv::Size originalSize = img.size();

    // Improved preprocessing with proper letterboxing
    cv::Mat letterboxed = letterBox(img, cv::Size(inputSize, inputSize));

    // Convert to blob with proper normalization
    cv::Mat blob;
    cv::dnn::blobFromImage(letterboxed, blob, 1.0/255.0, cv::Size(inputSize, inputSize),
                          cv::Scalar(0, 0, 0), true, false);


    cv::dnn::Net net = cv::dnn::readNetFromONNX(yoloModelWeights);
    //cv::dnn::Net net = cv::dnn::readNetFromDarknet(yoloModelConfiguration, yoloModelWeights);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    //net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    //net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    if (net.empty()) {
        cerr << "Error: Network not loaded properly" << endl;
        return;
    }

    // Get output layer names
    vector<cv::String> names = net.getUnconnectedOutLayersNames();

    // Forward pass
    net.setInput(blob);
    vector<cv::Mat> netOutput;
    net.forward(netOutput, names);

    std::cout << "Number of outputs: " << netOutput.size() << std::endl;
    for (size_t i = 0; i < netOutput.size(); ++i) {
        std::cout << "Output " << i << " dimensions: " << netOutput[i].dims << std::endl;
        std::cout << "Output " << i << " size: ";
        for (int j = 0; j < netOutput[i].dims; ++j) {
            std::cout << netOutput[i].size[j] << " ";
        }
        std::cout << std::endl;
    }

    if (netOutput.empty()) {
        cerr << "Error: No network output received" << endl;
        return;
    }

    cv::Mat output = netOutput[0];

    // YOLOv11 output format: [1, num_features, num_detections]
    // Features: [x, y, w, h, class_scores...]

    int dimensions = output.dims;
    if (dimensions != 3) {
        cerr << "Error: Expected 3D output tensor, got " << dimensions << "D" << endl;
        return;
    }

    int batch_size = output.size[0];  // Should be 1
    int num_features = output.size[1]; // x, y, w, h + num_classes
    int num_detections = output.size[2]; // Number of detections (e.g., 8400)

    int num_classes = num_features - 4; // Subtract x, y, w, h

    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Features: " << num_features << std::endl;
    std::cout << "Detections: " << num_detections << std::endl;
    std::cout << "Classes: " << num_classes << std::endl;

    if (num_classes <= 0) {
        cerr << "Error: Invalid number of classes: " << num_classes << endl;
        return;
    }

    // Get output dimensions
    vector<int> classIds;
    vector<float> confidences;
    vector<cv::Rect> boxes;

    auto data = reinterpret_cast<const float*>(output.data);

    // Process each detection
    for (int d = 0; d < num_detections; ++d) {
        // Extract bounding box coordinates (center format)
        float centerX = data[0 * num_detections + d];
        float centerY = data[1 * num_detections + d];
        float width = data[2 * num_detections + d];
        float height = data[3 * num_detections + d];

        // Find class with highest confidence
        int classId = -1;
        float maxScore = -FLT_MAX;
        for (int c = 0; c < num_classes; ++c) {
            float score = data[(4 + c) * num_detections + d];
            if (score > maxScore) {
                maxScore = score;
                classId = c;
            }
        }

        // Only keep detections above confidence threshold
        if (maxScore > confThreshold) {
            // Convert from center format to corner format
            float left = centerX - width / 2.0f;
            float top = centerY - height / 2.0f;

            // Create rectangle in letterboxed image coordinates
            cv::Rect box(static_cast<int>(left), static_cast<int>(top),
                        static_cast<int>(width), static_cast<int>(height));

            // Scale coordinates back to original image size
            cv::Rect scaledBox = scaleCoords(cv::Size(inputSize, inputSize), box, originalSize);

            boxes.push_back(scaledBox);
            classIds.push_back(classId);
            confidences.push_back(maxScore);
        }
    }

    // Perform non-maxima suppression
    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    std::vector<BoundingBox> bBoxes;
    // Create BoundingBox objects for results
    for(int idx : indices) {
        BoundingBox bBox;
        bBox.roi = boxes[idx];
        bBox.classID = classIds[idx];
        bBox.confidence = confidences[idx];
        bBox.boxID = static_cast<int>(bBoxes.size()); // zero-based unique identifier
        bBoxes.push_back(bBox);
    }

    cv::Mat visImg = img.clone();
    // Show results
    if(bVis) {
        for(const auto& box : bBoxes) {
            // Draw rectangle displaying the bounding box
            int top, left, width, height;
            top = box.roi.y;
            left = box.roi.x;
            width = box.roi.width;
            height = box.roi.height;
            cv::rectangle(visImg, cv::Point(left, top), cv::Point(left+width, top+height), cv::Scalar(0, 255, 0), 2);

            string label = cv::format("%.2f", box.confidence);
            if (box.classID < static_cast<int>(classes.size())) {
                label += classes[box.classID] + ":";
            }

            // Display label at the top of the bounding box
            int baseLine;
            cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);
            top = max(top, labelSize.height);
            rectangle(visImg, cv::Point(left, static_cast<int>(top - round(1.5*labelSize.height))),
                     cv::Point(static_cast<int>(left + round(1.5*labelSize.width)), top + baseLine),
                     cv::Scalar(255, 255, 255), cv::FILLED);
            cv::putText(visImg, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0,0,0), 1);
        }

        // string windowName = "Object classification";
        // cv::namedWindow(windowName, 1);
        // cv::imshow(windowName, visImg);
        // cv::waitKey(1);
    }
    cvmat_to_pwbuffer(visImg, out_buffer, frame_width, frame_height);
}

// Thread function that does the actual detection
void detectobjects_worker_thread(AsyncDetectionData* data)
{
    bool success = false;

    try {
        cv::Mat img = pwbuffer_to_cvmat(data->buffer, data->frame_width, data->frame_height);

        string yoloModelWeights = std::string(data->basePath) + "yolo11s.onnx";
        // Load class names from file
        vector<string> classes;
        ifstream ifs(std::string(data->classesFile).c_str());
        string line;
        while (getline(ifs, line))
        classes.push_back(line);

        
        constexpr int inputSize = 640;
        cv::Size originalSize = img.size();

        // Improved preprocessing with proper letterboxing
        cv::Mat letterboxed = letterBox(img, cv::Size(inputSize, inputSize));

        // Convert to blob with proper normalization
        cv::Mat blob;
        cv::dnn::blobFromImage(letterboxed, blob, 1.0/255.0, cv::Size(inputSize, inputSize),
                            cv::Scalar(0, 0, 0), true, false);


        cv::dnn::Net net = cv::dnn::readNetFromONNX(yoloModelWeights);
        //cv::dnn::Net net = cv::dnn::readNetFromDarknet(yoloModelConfiguration, yoloModelWeights);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        //net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        //net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

        if (net.empty()) {
            cerr << "Error: Network not loaded properly" << endl;
            return;
        }

        // Get output layer names
        vector<cv::String> names = net.getUnconnectedOutLayersNames();

        // Forward pass
        net.setInput(blob);
        vector<cv::Mat> netOutput;
        net.forward(netOutput, names);

        std::cout << "Number of outputs: " << netOutput.size() << std::endl;
        for (size_t i = 0; i < netOutput.size(); ++i) {
            std::cout << "Output " << i << " dimensions: " << netOutput[i].dims << std::endl;
            std::cout << "Output " << i << " size: ";
            for (int j = 0; j < netOutput[i].dims; ++j) {
                std::cout << netOutput[i].size[j] << " ";
            }
            std::cout << std::endl;
        }

        if (netOutput.empty()) {
            cerr << "Error: No network output received" << endl;
            return;
        }

        cv::Mat output = netOutput[0];

        // YOLOv11 output format: [1, num_features, num_detections]
        // Features: [x, y, w, h, class_scores...]

        int dimensions = output.dims;
        if (dimensions != 3) {
            cerr << "Error: Expected 3D output tensor, got " << dimensions << "D" << endl;
            return;
        }

        int batch_size = output.size[0];  // Should be 1
        int num_features = output.size[1]; // x, y, w, h + num_classes
        int num_detections = output.size[2]; // Number of detections (e.g., 8400)

        int num_classes = num_features - 4; // Subtract x, y, w, h

        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Features: " << num_features << std::endl;
        std::cout << "Detections: " << num_detections << std::endl;
        std::cout << "Classes: " << num_classes << std::endl;

        if (num_classes <= 0) {
            cerr << "Error: Invalid number of classes: " << num_classes << endl;
            return;
        }

        // Get output dimensions
        vector<int> classIds;
        vector<float> confidences;
        vector<cv::Rect> boxes;

        auto output_data = reinterpret_cast<const float*>(output.data);

        // Process each detection
        for (int d = 0; d < num_detections; ++d) {
            // Extract bounding box coordinates (center format)
            float centerX = output_data[0 * num_detections + d];
            float centerY = output_data[1 * num_detections + d];
            float width = output_data[2 * num_detections + d];
            float height = output_data[3 * num_detections + d];

            // Find class with highest confidence
            int classId = -1;
            float maxScore = -FLT_MAX;
            for (int c = 0; c < num_classes; ++c) {
                float score = output_data[(4 + c) * num_detections + d];
                if (score > maxScore) {
                    maxScore = score;
                    classId = c;
                }
            }

            // Only keep detections above confidence threshold
            if (maxScore > data->confThreshold) {
                // Convert from center format to corner format
                float left = centerX - width / 2.0f;
                float top = centerY - height / 2.0f;

                // Create rectangle in letterboxed image coordinates
                cv::Rect box(static_cast<int>(left), static_cast<int>(top),
                            static_cast<int>(width), static_cast<int>(height));

                // Scale coordinates back to original image size
                cv::Rect scaledBox = scaleCoords(cv::Size(inputSize, inputSize), box, originalSize);

                boxes.push_back(scaledBox);
                classIds.push_back(classId);
                confidences.push_back(maxScore);
            }
        }

        // Perform non-maxima suppression
        vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, data->confThreshold, data->nmsThreshold, indices);

        std::vector<BoundingBox> bBoxes;
        // Create BoundingBox objects for results
        for(int idx : indices) {
            BoundingBox bBox;
            bBox.roi = boxes[idx];
            bBox.classID = classIds[idx];
            bBox.confidence = confidences[idx];
            bBox.boxID = static_cast<int>(bBoxes.size()); // zero-based unique identifier
            bBoxes.push_back(bBox);
        }

        cv::Mat visImg = img.clone();
        // Show results
        if(data->bVis) {
            for(const auto& box : bBoxes) {
                // Draw rectangle displaying the bounding box
                int top, left, width, height;
                top = box.roi.y;
                left = box.roi.x;
                width = box.roi.width;
                height = box.roi.height;
                cv::rectangle(visImg, cv::Point(left, top), cv::Point(left+width, top+height), cv::Scalar(0, 255, 0), 2);

                string label = cv::format("%.2f", box.confidence);
                if (box.classID < static_cast<int>(classes.size())) {
                    label += classes[box.classID] + ":";
                }

                // Display label at the top of the bounding box
                int baseLine;
                cv::Size labelSize = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);
                top = max(top, labelSize.height);
                rectangle(visImg, cv::Point(left, static_cast<int>(top - round(1.5*labelSize.height))),
                        cv::Point(static_cast<int>(left + round(1.5*labelSize.width)), top + baseLine),
                        cv::Scalar(255, 255, 255), cv::FILLED);
                cv::putText(visImg, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0,0,0), 1);
            }
        }
        cvmat_to_pwbuffer(visImg, data->buffer, data->frame_width, data->frame_height);
        success = true;
    } catch (const std::exception& e){
        cout << "detection failed: " << e.what() << endl;
        // print ("detection failed: %s\n", e.what());
        success = false;
    }
    
    data->callback(data->buffer, data->user_data, success);

    delete data;
}

// Async entry point
void detectObjects_async(struct pw_buffer *out_buffer,
                        float confThreshold,
                        float nmsThreshold,
                        const char* basePath,
                        const char* classesFile,
                        uint32_t frame_width,
                        uint32_t frame_height,
                        bool bVis,
                        detection_callback_t callback,
                        struct impl* user_data)
{
    // thread data structure
    AsyncDetectionData* data = new AsyncDetectionData{
        .buffer        = out_buffer,
        .confThreshold = confThreshold,
        .nmsThreshold  = nmsThreshold,
        .basePath      = std::string(basePath),
        .classesFile   = std::string(classesFile),
        .frame_width   = frame_width,
        .frame_height  = frame_height,
        .bVis          = bVis,
        .callback      = callback,
        .user_data     = user_data
    };

    std::thread detection_thread(detectobjects_worker_thread, data);
    detection_thread.detach();
}