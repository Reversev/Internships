#include"pede_attr.hpp"

bool Pede_attr::run(const vector<uint8_t*> input_imgs, vector<vector<float>>& results, const int ori_height, const int ori_width) {
  for (size_t i = 0; i < input_imgs.size(); i++) {
    vector<float> feat;
    extract(input_imgs[i], feat, ori_height, ori_width);
    results.push_back(feat);
    feat.clear();
  }
  return true;
}

Pede_attr::Pede_attr(Pede_attr_config config)
{
    this->inpWidth = config.width;
    this->inpHeight = config.height;
	this->confThreshold = config.confThreshold;
	this->model_path = config.modelpath;

	this->enet_ptr = shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_path.c_str()));
	this->backendConfig.precision = (MNN::BackendConfig::PrecisionMode) this->precision;
	this->backendConfig.power = (MNN::BackendConfig::PowerMode) this->power;
	this->backendConfig.memory = (MNN::BackendConfig::MemoryMode) this->memory;

	this->netconfig.type = MNN_FORWARD_AUTO;
	this->netconfig.numThread = 2;
	this->netconfig.backendConfig = &this->backendConfig; 

	this->mnn_session = enet_ptr->createSession(netconfig);
	input = enet_ptr->getSessionInput(mnn_session, nullptr);
    enet_ptr->resizeTensor(input, {1, 3, this->inpWidth, this->inpWidth});
	enet_ptr->resizeSession(mnn_session);

	preconfig.sourceFormat = MNN::CV::BGR;
    preconfig.destFormat = MNN::CV::RGB;
    preconfig.filterType = MNN::CV::BILINEAR;
    memcpy(preconfig.mean, mean_vals, 3 * sizeof(float));
    memcpy(preconfig.normal, norm_vals, 3 * sizeof(float));
    this->pretreat = std::shared_ptr<MNN::CV::ImageProcess> (MNN::CV::ImageProcess::create(preconfig));
}


Pede_attr::~Pede_attr(){ enet_ptr->releaseModel(); }


void Pede_attr::extract(const uint8_t* frame, vector<float>& feat, const int ori_height, const int ori_width)
{
    pretreat->setMatrix(transform);
    pretreat->convert(frame, this->inpWidth, this->inpHeight, 0, this->input);

	this->enet_ptr->runSession(this->mnn_session);

	MNN::Tensor *tensor_scores = enet_ptr->getSessionOutput(this->mnn_session, NULL);

    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
    
    tensor_scores->copyToHostTensor(&tensor_scores_host);
    // post processing steps
    auto pdata  = tensor_scores_host.host<float>();
    // show features from given image
    for(int i = 0; i <  tensor_scores->shape()[1]; ++i)
    {
        feat.push_back(pdata[i]);
    }


}

