#include "nn/onnx_model_base.h"
#include "nn/autobackend.h"
#include <opencv2/opencv.hpp>
#include <vector>

#include "utils/augment.h"
#include "constants.h"
#include "utils/common.h"

void plot_results(cv::Mat img, std::vector<YoloResults>& results,
                  std::unordered_map<int, std::string>& names,
                  const cv::Size& shape
                  ) {

    cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);

    for (const auto& res : results) {
        float left = res.bbox.x;
        float top = res.bbox.y;

        // Draw bounding box
        rectangle(img, res.bbox, cv::Scalar(0, 255, 0), 2);

        // Draw mask if available
        if (res.mask.rows && res.mask.cols > 0) {
            mask(res.bbox).setTo(cv::Scalar(255), res.mask);
        }
    }

    cv::imshow("mask", mask);
}


int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_file or 'webcam'>" << std::endl;
        return -1;
    }

    cv::VideoCapture cap;
    if (std::string(argv[1]) == "webcam") {
        cap.open(0);  // Open the webcam
        if (!cap.isOpened()) {
            std::cerr << "Error: Unable to open webcam." << std::endl;
            return -1;
        }
    } else {
        // Try to open the provided image file
        cap.open(argv[1]);
        if (!cap.isOpened()) {
            std::cerr << "Error: Unable to open image file or webcam." << std::endl;
            return -1;
        }
    }

    const std::string& onnx_provider = OnnxProviders::CUDA;
    const std::string& onnx_logid = "yolov8_inference2";
    float mask_threshold = 0.5f;
    float conf_threshold = 0.30f;
    float iou_threshold = 0.45f;
	int conversion_code = cv::COLOR_BGR2RGB;

    const std::string& modelPath = "best_1920_1088.onnx";

    AutoBackendOnnx model(modelPath.c_str(), onnx_logid.c_str(), onnx_provider.c_str());
    std::unordered_map<int, std::string> names = model.getNames();

    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Unable to capture frame from webcam." << std::endl;
            break;
        }

        // Process the frame
        std::vector<YoloResults> objs = model.predict_once(frame, conf_threshold, iou_threshold, mask_threshold, conversion_code);

        cv::Size show_shape = frame.size();
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
        plot_results(frame, objs, names, show_shape);

        cv::imshow("Webcam Feed", frame);

        if (cv::waitKey(1) == 27) {  // Break the loop if the 'Esc' key is pressed
            break;
        }
    }

    cv::waitKey();

    return -1;
}
