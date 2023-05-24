#include <unistd.h>
#include"face_detector.hpp"


int main(int argc, char* argv[])
{
	if (argc < 3) {
		cout << "USAGE:" << endl;
		cout << argv[0] << " required: <input_mode> <img path> <face_detect model path> " << endl;
		exit(1);
	}

	int input_mode = atoi(argv[1]); // 0 image 1 video
	string image_list = argv[2];
	string dmodel_path = argv[3];

	// mkdir save folder 
	string save_foler = "result_imgs";
	if (0 != access(save_foler.c_str(), 0)) {
		system("mkdir -p result_imgs");
	}
	
	// load detect model
	Net_config YOLOV7_face_cfg = {0.3, 0.4, dmodel_path};   // confThreshold nmsThreshold modelpath
	YOLOV7_face net(YOLOV7_face_cfg);

	if (input_mode == 0){
		Mat srcimg = imread(image_list);
		net.detect(srcimg);
		imwrite(save_foler + "/res_output.jpg", srcimg);
	}
	else{
		cout << "Not support this input mode." << endl;
		exit(1);
	}
}
