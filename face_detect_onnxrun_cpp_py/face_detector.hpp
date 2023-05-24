#ifndef face_detector_hpp
#define face_detector_hpp
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;
using namespace Ort;


struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	string modelpath;
};

typedef struct PointInfo
{
	Point pt;
	float score;
} PointInfo;

typedef struct BoxInfo
{
	// retangle
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	// keypoints
	PointInfo kpt1;
	PointInfo kpt2;
	PointInfo kpt3;
	PointInfo kpt4;
	PointInfo kpt5;
} BoxInfo;

class YOLOV7_face
{
public:
	YOLOV7_face(Net_config config);   // configure face model
	void detect(Mat& frame);          // detect per frame 
private:
	int inpWidth;
	int inpHeight;
	int nout;    // calculate output number 
	int num_proposal;   // proposed rect from model

	float confThreshold;
	float nmsThreshold;
	vector<float> input_image_;
	void normalize_(Mat img);           // scale face image 
	void nms(vector<BoxInfo>& input_boxes);  // nms 
	bool has_postprocess; 

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "YOLOV7_face");    // set logger 
	Ort::Session *ort_session = nullptr;  // initialize model session: model path
	SessionOptions sessionOptions = SessionOptions();  // initialize optins session: num_thresh and op graph level 
	vector<char*> input_names;       // save input names from model
	vector<char*> output_names;      // save output names from model
	vector<vector<int64_t>> input_node_dims; // >=1 outputs   save input dims
	vector<vector<int64_t>> output_node_dims; // >=1 outputs  save output dims
};

#endif /* face_detector_hpp */
