#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>

void normalize_(cv::Mat img, std::vector<float>& input_image_)
{
	int row = img.rows;
	int col = img.cols;
	input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				input_image_[c * row * col + i * col + j] = pix;  // ((x / 255.) - 0.5) / 0.5
			}
		}
	}
}

int main(int argc, char* argv[]) {
    std::string model_path = "./model.onnx";
    std::string imgPath = "./img2.jpeg";
   
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Default");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    Ort::Session *ort_session = nullptr;
    ort_session = new Ort::Session(env, model_path.c_str(), sessionOptions); // CPU

    size_t numInputNodes = ort_session->GetInputCount();
    size_t numOutputNodes = ort_session->GetOutputCount();
    Ort::AllocatorWithDefaultOptions allocator;
	std::vector<char*> input_names;       // save input names from model
	std::vector<char*> output_names;      // save output names from model
	std::vector<std::vector<int64_t>> input_node_dims;  // >=1 outputs   save input dims 
	std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs  save output dims
    std::vector<float> input_image_;         // input image data after resize and scale

    for (int i = 0; i < numInputNodes; i++)
    {
        input_names.push_back(ort_session->GetInputName(i, allocator));   // get input names
        Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_node_dims.push_back(input_dims);
    }
    for (int i = 0; i < numOutputNodes; i++)
    {
        output_names.push_back(ort_session->GetOutputName(i, allocator));
        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }

    const int height_ = input_node_dims[0][2]; 
    const int width_ = input_node_dims[0][3];
    const int dims = output_node_dims[0][1];

    Ort::Value input_tensor_{nullptr};
    std::array<int64_t, 4> input_shape_{1, 3, height_, width_};   // mode input shape NCHW = 1x3xHxW

    Ort::Value output_tensor_{nullptr};
    std::array<int64_t, 2> output_shape_{1, dims}; // model output shape

    // load data 
    cv::Mat img = cv::imread(imgPath);
    cv::resize(img, img, cv::Size(height_, width_));     // resize
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);        // bgr -> rgb
    std::cout << "dim:" << img.rows << " " << img.cols << " " << img.channels() << std::endl;
    normalize_(img, input_image_);

    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
    std::vector<Ort::Value> ort_outputs = ort_session->Run(Ort::RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());  

	Ort::Value &predictions = ort_outputs.at(0);
	auto pred_dims = predictions.GetTensorTypeAndShapeInfo().GetShape();
	const float* pdata = predictions.GetTensorMutableData<float>();
    // show features from given image
    for(int i = 0; i < pred_dims.at(1); ++i)
    {
        std::cout << pdata[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}

