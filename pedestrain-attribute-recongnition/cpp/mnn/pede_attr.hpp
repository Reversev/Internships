#ifndef face_extractor__hpp
#define face_extractor__hpp

#include <fstream>
#include <sstream>
#include <iostream>
#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
using namespace std;

struct Pede_attr_config
{
	int width;
	int height;
	float confThreshold; // matching threshold
	string modelpath;
};


class Pede_attr
{
public:
	Pede_attr(Pede_attr_config config);   // configure face model
	~Pede_attr();
    bool run(const vector<uint8_t*> input_imgs, vector<vector<float>>& results, const int ori_height, const int ori_width);
    void extract(const uint8_t* frame, vector<float>& feat, const int ori_height, const int ori_width);  // extract per frame

private:
	int inpWidth;
	int inpHeight;

	int precision  = 0;
    int power      = 0;
    int memory     = 0;

	float confThreshold;
	string model_path;
	
	MNN::CV::ImageProcess::Config preconfig;
	MNN::ScheduleConfig netconfig;
	MNN::BackendConfig backendConfig;
	MNN::Tensor* input;         // input image data after resize and scale
	
	MNN::CV::Matrix transform;
	std::shared_ptr<MNN::CV::ImageProcess> pretreat = nullptr;
	shared_ptr<MNN::Interpreter> enet_ptr = nullptr;
	MNN::Session* mnn_session = nullptr;  // initialize model session: model path
	float mean_vals[3] = {123.675f, 116.28f, 103.53f};
    float norm_vals[3] = {58.395f, 57.12f, 57.375f};

};
#endif /* face_extractor__hpp */