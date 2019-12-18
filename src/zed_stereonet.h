#include <sl/Camera.hpp>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>

using namespace sl;
using namespace std;

torch::jit::script::Module load_module(std::string path);
cv::Mat slMat2cvMat(Mat& input);
cv::Mat tensor2cvMat(at::Tensor tensor_input);
std::vector<torch::jit::IValue> cvMat2moduleinput(cv::Mat left, cv::Mat right);

std::vector<torch::jit::IValue> cvMat2moduleinput(cv::Mat left, cv::Mat right){
  cv::Mat left_norm, right_norm;
  cv::cvtColor(left, left, CV_BGR2RGB);
  cv::cvtColor(right, right, CV_BGR2RGB);
  left.convertTo(left_norm, CV_32FC3, 1.f/255);
  right.convertTo(right_norm, CV_32FC3, 1.f/255);

  at::Tensor left_tensor = torch::from_blob(left_norm.data, {1, left_norm.rows, left_norm.cols, 3});
  at::Tensor right_tensor = torch::from_blob(right_norm.data, {1, right_norm.rows, right_norm.cols, 3});
  
  left_tensor = left_tensor.permute({0, 3, 1, 2});
  right_tensor = right_tensor.permute({0, 3, 1, 2});
  at::Tensor image = torch::cat({left_tensor, right_tensor}, 3);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(image.to(at::kCUDA));

  return inputs;
}

cv::Mat tensor2cvMat(at::Tensor tensor_input){
  at::Tensor tensor_input_transf = tensor_input.to(torch::kCPU);
  tensor_input_transf = tensor_input_transf.squeeze().detach();
  tensor_input_transf = tensor_input_transf.to(torch::kFloat32);
  int64_t rows = tensor_input_transf.sizes()[0];
  int64_t cols = tensor_input_transf.sizes()[1];
  cv::Mat out_image = cv::Mat::zeros(rows, cols, CV_32FC1);
  std::memcpy(out_image.data, tensor_input_transf.data_ptr(), sizeof(float)*tensor_input_transf.numel());
  
  return out_image;
}

torch::jit::script::Module load_module(std::string path){
  // Loading model.
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    return module = torch::jit::load(path);
      }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
                              } 
}

cv::Mat slMat2cvMat(Mat& input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
        case MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
        case MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
        case MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
        case MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
        case MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
        case MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
        case MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
        case MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
        default: break;
    }
    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
}
