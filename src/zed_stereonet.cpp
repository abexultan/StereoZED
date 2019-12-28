// #include <torch/script.h> 
#include <sl/Camera.hpp>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include "zed_stereonet.h"

using namespace sl;
using namespace std;


int main(int argc, const char* argv[]) {


  if (argc != 2) {
    std::cerr << "usage: zed_stereonet <path-to-exported-script-module>\n";
    return -1;
  }

  Camera zed;

  InitParameters param;
  param.camera_resolution = RESOLUTION_HD720;
  param.sdk_verbose = true;
  param.camera_fps = 30;

  ERROR_CODE err = zed.open(param);
  if (err != SUCCESS) {
    cout << toString(err) << endl;
    zed.close();
    return 1; // Quit if an error occurred
  }

  Mat zed_image_L, zed_image_R;
  cv::Mat left, right, output_cv;
  std::vector<torch::jit::IValue> inputs;
  at::Tensor output;

  torch::jit::script::Module module = load_module(argv[1]);
  module.to(at::kCUDA);
  cv::namedWindow("Disparity Display");
  char key = ' ';
    while (key != 'q') {
          if (zed.grab() == SUCCESS) {
            // Retrieve images
            zed.retrieveImage(zed_image_L, VIEW_LEFT);
            zed.retrieveImage(zed_image_R, VIEW_RIGHT);
            left = slMat2cvMat(zed_image_L);
            right = slMat2cvMat(zed_image_R);
            inputs = cvMat2moduleinput(left, right);
            output = module.forward(inputs).toTensor();
            output_cv = tensor2cvMat(output/255.0);
            cv::imshow("Disparity Display", output_cv);
          }
          key = cv::waitKey(5);
}
  cv::destroyAllWindows();
  zed.close();
  return EXIT_SUCCESS;
}
