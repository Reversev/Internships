#include <unistd.h>
#include <sys/time.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <numeric>
#include "pede_attr1.hpp"


int main(int argc, char* argv[])
{
	if (argc < 2) {
		printf("USAGE: %s", argv[0]);
		printf(" required: <image path> <pedestrian attribute model path>\n");
		exit(1);
	}

	string image_path = argv[1];
	string model_path = argv[2];

	// read gallery names and features
	struct timeval tpstart, tpend;
    float timeuse = 0.f;
	const float conf_thresh = 0.5f; 
	vector<float> t_ts, d_ts, e_ts;  

	size_t index = image_path.rfind("/");
	string img_name = image_path.substr(index + 1);
	cv::Mat img = imread(image_path, cv::IMREAD_COLOR);   
	if (img.empty()) {
		cout << ("read image " + image_path + "error!\n");
		exit(1);
	}

	// initialize Pede_attr model
	Pede_attr_config Pede_attr_cfg = {conf_thresh, model_path};   // confThreshold modelpath
	shared_ptr<Pede_attr> pede_attr_share_ptr(new Pede_attr(Pede_attr_cfg), default_delete<Pede_attr>());
	Pede_attr* pede_attr_ptr = pede_attr_share_ptr.get();

	vector<cv::Mat> tmp_imgs;
	vector<vector<float>> feats;
	tmp_imgs.push_back(img);  // 1 
	gettimeofday(&tpstart, NULL);
	pede_attr_ptr->run(tmp_imgs, feats);
	// for (int k=0; k<feats[0].size(); ++k){
	// 	cout << feats[0][k] << " ";
	// }
	// cout << endl;
	tmp_imgs.clear();
	feats.clear(); 

	gettimeofday(&tpend, NULL);
	timeuse = 1000000 * (tpend.tv_sec - tpstart.tv_sec) + tpend.tv_usec - tpstart.tv_usec;
	timeuse /= 1000;
	printf("total extract time: %.2fms\n", timeuse);

  	return 0;
}





    
    
    