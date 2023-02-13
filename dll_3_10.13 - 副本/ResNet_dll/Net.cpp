#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <regex>
#include <Windows.h>
#include <unordered_map>

#define DEVICE 0


const char* INPUT_NAME = "data";
const char* OUTPUT_NAME = "prob";

#define ERR -1;
#define OK 0;


using namespace std;
using namespace nvinfer1;

static Logger gLogger;

struct return_data
{
	ICudaEngine* eng;
	int img_h;
	int img_w;
	int cls;

}get_par = { 0, 0, 0, 0 };


/// <summary>
/// 模型解密
/// </summary>
/// <param name="inCode">要解密的字符串</param>
/// <returns> 返回解密结果</returns>
string Decrypt_name(string inCode)
{
	string out_layer_name;
	unsigned int iSize = inCode.size();
	/*for (int i = 0; i < iSize; i++)
	{
		int asc = inCode[i] - 1;
		char ch = (char)asc;
		out_layer_name += ch;
		string cc = out_layer_name;
	}*/
	for (int i = 0; i < iSize; i++)
	{
		out_layer_name += (char)(inCode[i] - 1);
		string cc = out_layer_name;
	}

	return out_layer_name;
}

//ResNet

/// <summary>
/// 加载bin文件
/// </summary>
/// <param name="file"> bin文件的路径地址</param>
/// <returns> 加载完成的网络结构及其权重</returns>
std::unordered_map<std::string, Weights> load_weight(const string file)
{
	std::unordered_map<std::string, Weights> weightMap;

	//open wts file
	ifstream input(file);
	assert(input.is_open() && "Unable to load weight file.");

	//read number of weight blbs
	int32_t count;
	input >> count;
	assert(count > 0 && "Invalid weight map file.");

	while (count--)
	{
		Weights wt{ DataType::kFLOAT, nullptr, 0 };
		uint32_t size;

		//read name and type of blob
		string name;
		input >> name >> dec >> size;
		wt.type = DataType::kFLOAT;
		string layer_name;
		layer_name = Decrypt_name(name);

		//load blob
		uint32_t* val = reinterpret_cast<uint32_t*> (malloc(sizeof(val) * size));
		for (uint32_t x = 0, y = size; x < y; ++x)
		{
			input >> hex >> val[x];
		}
		wt.values = val;
		wt.count = size;
		weightMap[layer_name] = wt;
	}
	return weightMap;
}

//bin文件转换成二进制
bool text_to_binary(const char* infilename, const char* outfilename)
{
	std::ifstream in(infilename);
	std::ofstream out(outfilename, std::ios::binary);

	uint32_t line_count;
	if (!(in >> line_count))
	{
		return false;
	}
	if (out.write(reinterpret_cast<const char*>(&line_count), sizeof(line_count)))
	{
		cout << (&line_count);
	}
	for (uint32_t l = 0; l < line_count; ++l)
	{
		std::string name;

		uint32_t num_values;
		if (!(in >> name >> std::dec >> num_values))
		{
			return false;
		}

		std::vector<uint32_t> values(num_values);
		for (uint32_t i = 0; i < num_values; ++i)
		{
			if (!(in >> std::hex >> values[i]))
			{
				return false;
			}
		}

		uint32_t name_size = static_cast<uint32_t>(name.size());

		bool a = out.write(reinterpret_cast<const char*>(&name_size), sizeof(name_size)) &&
			out.write(name.data(), name.size()) &&
			out.write(reinterpret_cast<const char*>(&num_values), sizeof(num_values)) &&
			out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(values[0]));
		if (a == false)
		{
			return false;
		}

	}
	return true;
}

//网络读取并加载二进制模型
std::unordered_map<std::string, Weights> read_weights(const char* infilename)
{
	std::unordered_map<std::string, Weights> weightMap;

	std::ifstream in(infilename, std::ios::binary);
	if (!in)
	{
		std::cerr << "Error: Could not open input file '" << infilename << "'\n";
	}

	uint32_t line_count;
	if (!in.read(reinterpret_cast<char*>(&line_count), sizeof(line_count)))
	{
		std::cerr << "Error: line count too short '" << infilename << "'\n";
	}

	for (uint32_t l = 0; l < line_count; ++l)
	{
		uint32_t name_size;
		if (!in.read(reinterpret_cast<char*>(&name_size), sizeof(name_size)))
		{
			std::cerr << "Error: read name size error '" << infilename << "'\n";
		}
		std::string name(name_size, 0);
		if (!in.read(const_cast<char*>(name.data()), name_size))
		{
			std::cerr << "Error: name data error '" << infilename << "'\n";
		}

		uint32_t num_values;
		if (!in.read(reinterpret_cast<char*>(&num_values), sizeof(num_values)))
		{
			std::cerr << "Error: read name values error '" << infilename << "'\n";
		}

		// Normally I would use float* values = new float[num_values]; here which
		// requires delete [] ptr; to free the memory later.
		// I used malloc to match the original example since I don't know who is
		// responsible to clean things up later, and TensorRT might use free(ptr)
		// Makes no real difference as long as new/delete ro malloc/free are matched up.
		float* values = reinterpret_cast<float*>(malloc(num_values * sizeof(*values)));
		if (!in.read(reinterpret_cast<char*>(values), num_values * sizeof(*values)))
		{
			std::cerr << "Error: Could not open input file '" << infilename << "'\n";
		}
		string layer_name = Decrypt_name(name);
		weightMap[layer_name] = Weights{ DataType::kFLOAT, values, num_values };
	}
	return weightMap;
}

/// <summary>
/// 构建BN层
/// </summary>
/// <param name="network"> TensorRT的网络结构</param>
/// <param name="weightMap"> 构建的模型的weightMap</param>
/// <param name="input"> 权重的值</param>
/// <param name="lname"> 在模型里相应的层数</param>
/// <param name="eps"> 一个特别小的数，用于防止分母为0</param>
/// <returns> 返回构建好的层</returns>
IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input, string lname, float eps)
{
	float* gamma = (float*)weightMap[lname + ".weight"].values;
	float* beta = (float*)weightMap[lname + ".bias"].values;
	float* mean = (float*)weightMap[lname + ".running_mean"].values;
	float* var = (float*)weightMap[lname + ".running_var"].values;
	int len = weightMap[lname + ".running_var"].count;

	float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		scval[i] = gamma[i] / sqrt(var[i] + eps);
	}
	Weights scale{ DataType::kFLOAT, scval, len };

	float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
	}
	Weights shift{ DataType::kFLOAT, shval, len };

	float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
	for (int i = 0; i < len; i++) {
		pval[i] = 1.0;
	}
	Weights power{ DataType::kFLOAT, pval, len };

	weightMap[lname + ".scale"] = scale;
	weightMap[lname + ".shift"] = shift;
	weightMap[lname + ".power"] = power;
	IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
	assert(scale_1);
	return scale_1;
}

/// <summary>
/// 构建ResNet的block
/// </summary>
/// <param name="network"> 要构建网络结构</param>
/// <param name="weightMap"> 构建的模型的weightMap</param>
/// <param name="input"> 权重的值</param>
/// <param name="inch"> 输入特征图的层数</param>
/// <param name="outch"> 输出特征图的层数</param>
/// <param name="stride"> 步长</param>
/// <param name="lname"> 在模型里相应的层数</param>
/// <returns> 返回构建好的网络层</returns>
IActivationLayer* basicBlock(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, string lname)
{
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

	IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ 3, 3 }, weightMap[lname + "conv1.weight"], emptywts);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ stride, stride });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);
	IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
	assert(relu1);

	IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ 3 ,3 }, weightMap[lname + "conv2.weight"], emptywts);
	assert(conv2);
	conv2->setPaddingNd(DimsHW{ 1,1 });
	IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

	IElementWiseLayer* ew1;
	if (inch != outch)
	{
		IConvolutionLayer* conv3 = network->addConvolutionNd(input, outch, DimsHW{ 1,1 }, weightMap[lname + "downsample.0.weight"], emptywts);
		assert(conv3);
		conv3->setStrideNd(DimsHW{ stride, stride });
		IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "downsample.1", 1e-5);
		ew1 = network->addElementWise(*bn3->getOutput(0), *bn2->getOutput(0), ElementWiseOperation::kSUM);
	}
	else {
		ew1 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
	}

	IActivationLayer* relu2 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
	assert(relu2);

	return relu2;
}

ICudaEngine* CreateEngine_ResNet(int mode_select, unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, string wts_path, int INPUT_H, int INPUT_W, int cudaSetPrecision, string train_exe_version)
{
	INetworkDefinition* network = builder->createNetworkV2(0U);

	//CREATE INPUT TENSOR OF SHAPE {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
	ITensor* data = network->addInput(INPUT_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
	assert(data);

	auto start1 = chrono::steady_clock::now();

	std::unordered_map<std::string, Weights> weightMap;
	//这三个版本，转出来的bin文件用了二进制保存，读取方式不一样，之后的版本还是用之前的方式读取。ResNet也一样.
	if (train_exe_version == "1.0.1" || train_exe_version == "1.0.2" || train_exe_version == "1.0.3")
	{
		weightMap = read_weights(wts_path.c_str()); //读取二进制数据
	}
	else
	{
		weightMap = load_weight(wts_path);
	}

	//std::unordered_map<std::string, Weights> weightMap = read_weights(wts_path.c_str());
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

	auto end1 = chrono::steady_clock::now();
	double total_time1 = chrono::duration<double, milli>(end1 - start1).count();
	cout << "load weight: " << total_time1 << " ms" << endl;

	if (mode_select == 1)
	{
		IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{ 7, 7 }, weightMap["conv1.weight"], emptywts);
		assert(conv1);
		conv1->setStrideNd(DimsHW{ 2, 2 });
		conv1->setPaddingNd(DimsHW{ 3, 3 });

		IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

		IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
		assert(relu1);

		IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 3 ,3 });
		assert(pool1);
		pool1->setStrideNd(DimsHW{ 2, 2 });
		pool1->setPaddingNd(DimsHW{ 1, 1 });

		IActivationLayer* relu2 = basicBlock(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.");
		IActivationLayer* relu3 = basicBlock(network, weightMap, *relu2->getOutput(0), 64, 64, 1, "layer1.1.");

		IActivationLayer* relu4 = basicBlock(network, weightMap, *relu3->getOutput(0), 64, 128, 2, "layer2.0.");
		IActivationLayer* relu5 = basicBlock(network, weightMap, *relu4->getOutput(0), 128, 128, 1, "layer2.1.");

		IActivationLayer* relu6 = basicBlock(network, weightMap, *relu5->getOutput(0), 128, 256, 2, "layer3.0.");
		IActivationLayer* relu7 = basicBlock(network, weightMap, *relu6->getOutput(0), 256, 256, 1, "layer3.1.");

		IActivationLayer* relu8 = basicBlock(network, weightMap, *relu7->getOutput(0), 256, 512, 2, "layer4.0.");
		IActivationLayer* relu9 = basicBlock(network, weightMap, *relu8->getOutput(0), 512, 512, 1, "layer4.1.");

		IPoolingLayer* pool2 = network->addPoolingNd(*relu9->getOutput(0), PoolingType::kAVERAGE, DimsHW{ 7,7 });
		assert(pool2);
		pool2->setStrideNd(DimsHW{ 1,1 });

		IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 2, weightMap["fc.weight"], weightMap["fc.bias"]);
		assert(fc1);

		fc1->getOutput(0)->setName(OUTPUT_NAME);
		cout << "wait..." << endl;
		network->markOutput(*fc1->getOutput(0));
	}

	else if (mode_select == 2)
	{
		IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{ 7, 7 }, weightMap["conv1.weight"], emptywts);
		assert(conv1);
		conv1->setStrideNd(DimsHW{ 2, 2 });
		conv1->setPaddingNd(DimsHW{ 3, 3 });

		IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

		IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
		assert(relu1);

		IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 3 ,3 });
		assert(pool1);
		pool1->setStrideNd(DimsHW{ 2, 2 });
		pool1->setPaddingNd(DimsHW{ 1, 1 });

		IActivationLayer* relu2 = basicBlock(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.");
		IActivationLayer* relu3 = basicBlock(network, weightMap, *relu2->getOutput(0), 64, 64, 1, "layer1.1.");
		IActivationLayer* relu4 = basicBlock(network, weightMap, *relu3->getOutput(0), 64, 64, 1, "layer1.2.");

		IActivationLayer* relu5 = basicBlock(network, weightMap, *relu4->getOutput(0), 64, 128, 2, "layer2.0.");
		IActivationLayer* relu6 = basicBlock(network, weightMap, *relu5->getOutput(0), 128, 128, 1, "layer2.1.");
		IActivationLayer* relu7 = basicBlock(network, weightMap, *relu6->getOutput(0), 128, 128, 1, "layer2.2.");
		IActivationLayer* relu8 = basicBlock(network, weightMap, *relu7->getOutput(0), 128, 128, 1, "layer2.3.");

		IActivationLayer* relu9 = basicBlock(network, weightMap, *relu8->getOutput(0), 128, 256, 2, "layer3.0.");
		IActivationLayer* relu10 = basicBlock(network, weightMap, *relu9->getOutput(0), 256, 256, 1, "layer3.1.");
		IActivationLayer* relu11 = basicBlock(network, weightMap, *relu10->getOutput(0), 256, 256, 1, "layer3.2.");
		IActivationLayer* relu12 = basicBlock(network, weightMap, *relu11->getOutput(0), 256, 256, 1, "layer3.2.");
		IActivationLayer* relu13 = basicBlock(network, weightMap, *relu12->getOutput(0), 256, 256, 1, "layer3.4.");
		IActivationLayer* relu14 = basicBlock(network, weightMap, *relu13->getOutput(0), 256, 256, 1, "layer3.5.");

		IActivationLayer* relu15 = basicBlock(network, weightMap, *relu14->getOutput(0), 256, 512, 2, "layer4.0.");
		IActivationLayer* relu16 = basicBlock(network, weightMap, *relu15->getOutput(0), 512, 512, 1, "layer4.1.");
		IActivationLayer* relu17 = basicBlock(network, weightMap, *relu16->getOutput(0), 512, 512, 1, "layer4.2.");

		IPoolingLayer* pool2 = network->addPoolingNd(*relu17->getOutput(0), PoolingType::kAVERAGE, DimsHW{ 7,7 });
		assert(pool2);
		pool2->setStrideNd(DimsHW{ 1,1 });

		IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 2, weightMap["fc.weight"], weightMap["fc.bias"]);
		assert(fc1);

		fc1->getOutput(0)->setName(OUTPUT_NAME);
		cout << "wait..." << endl;
		network->markOutput(*fc1->getOutput(0));
	}

	//build engine
	builder->setMaxBatchSize(maxBatchSize);
	config->setMaxWorkspaceSize(1 << 20);

	if (cudaSetPrecision == 16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}

	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

	//destroy network
	network->destroy();

	//release host memory
	for (auto& mem : weightMap)
	{
		free((void*)(mem.second.values));
	}

	return engine;
}

ICudaEngine* APIToModel_Res(int mode_select, unsigned int maxBatchSize, string wts_path, int input_H, int input_W, int cudaSetPrecision, string train_exe_version)
{
	//create builder
	IBuilder* builder = createInferBuilder(gLogger);
	IBuilderConfig* config = builder->createBuilderConfig();

	// create model to populate the network, then set the outputs anf create an engine
	ICudaEngine* engine = CreateEngine_ResNet(mode_select, maxBatchSize, builder, config, DataType::kFLOAT, wts_path, input_H, input_W, cudaSetPrecision, train_exe_version);
	assert(engine != nullptr);

	// destroy 
	builder->destroy();
	config->destroy();

	return engine;
}

//UNet

ILayer* outConv(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname, int num_cls) {
	// Weights emptywts{DataType::kFLOAT, nullptr, 0};

	IConvolutionLayer* conv1 = network->addConvolutionNd(input, num_cls, DimsHW{ 1, 1 }, weightMap[lname + ".weight"], weightMap[lname + ".bias"]);
	assert(conv1);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 0, 0 });
	conv1->setNbGroups(1);
	return conv1;
}

ILayer* block1(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, std::string lname, int midch) {
	IConvolutionLayer* conv1 = network->addConvolutionNd(input, midch, DimsHW{ ksize, ksize }, weightMap[lname + ".features.0.weight"], weightMap[lname + ".features.0.bias"]);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	conv1->setNbGroups(1);
	IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

	IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ ksize, ksize }, weightMap[lname + ".features.2.weight"], weightMap[lname + ".features.2.bias"]);
	conv2->setStrideNd(DimsHW{ 1, 1 });
	conv2->setPaddingNd(DimsHW{ 1, 1 });
	conv2->setNbGroups(1);
	IActivationLayer* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
	return relu2;
}
ILayer* block2(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, std::string lname, int midch) {

	IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{ 2, 2 });
	assert(pool1);

	IConvolutionLayer* conv1 = network->addConvolutionNd(*pool1->getOutput(0), midch, DimsHW{ ksize, ksize }, weightMap[lname + ".features.5.weight"], weightMap[lname + ".features.5.bias"]);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	conv1->setNbGroups(1);
	IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

	IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".features.7.weight"], weightMap[lname + ".features.7.bias"]);
	conv2->setStrideNd(DimsHW{ 1, 1 });
	conv2->setPaddingNd(DimsHW{ 1, 1 });
	conv2->setNbGroups(1);
	IActivationLayer* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
	return relu2;
}
ILayer* block3(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, std::string lname, int midch) {

	IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{ 2, 2 });
	assert(pool1);

	IConvolutionLayer* conv1 = network->addConvolutionNd(*pool1->getOutput(0), midch, DimsHW{ ksize, ksize }, weightMap[lname + ".features.10.weight"], weightMap[lname + ".features.10.bias"]);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	conv1->setNbGroups(1);
	IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

	IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ ksize, ksize }, weightMap[lname + ".features.12.weight"], weightMap[lname + ".features.12.bias"]);
	conv2->setStrideNd(DimsHW{ 1, 1 });
	conv2->setPaddingNd(DimsHW{ 1, 1 });
	conv2->setNbGroups(1);
	IActivationLayer* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);

	IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch, DimsHW{ ksize, ksize }, weightMap[lname + ".features.14.weight"], weightMap[lname + ".features.14.bias"]);
	conv3->setStrideNd(DimsHW{ 1, 1 });
	conv3->setPaddingNd(DimsHW{ 1, 1 });
	conv3->setNbGroups(1);
	IActivationLayer* relu3 = network->addActivation(*conv3->getOutput(0), ActivationType::kRELU);
	return relu3;
}
ILayer* block4(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, std::string lname, int midch) {

	IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{ 2, 2 });
	assert(pool1);
	IConvolutionLayer* conv1 = network->addConvolutionNd(*pool1->getOutput(0), midch, DimsHW{ ksize, ksize }, weightMap[lname + ".features.17.weight"], weightMap[lname + ".features.17.bias"]);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	conv1->setNbGroups(1);
	IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

	IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".features.19.weight"], weightMap[lname + ".features.19.bias"]);
	conv2->setStrideNd(DimsHW{ 1, 1 });
	conv2->setPaddingNd(DimsHW{ 1, 1 });
	conv2->setNbGroups(1);
	IActivationLayer* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);

	IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".features.21.weight"], weightMap[lname + ".features.21.bias"]);
	conv3->setStrideNd(DimsHW{ 1, 1 });
	conv3->setPaddingNd(DimsHW{ 1, 1 });
	conv3->setNbGroups(1);
	IActivationLayer* relu3 = network->addActivation(*conv3->getOutput(0), ActivationType::kRELU);
	return relu3;
}
ILayer* block5(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, std::string lname, int midch) {
	IPoolingLayer* pool1 = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{ 2, 2 });
	assert(pool1);
	IConvolutionLayer* conv1 = network->addConvolutionNd(*pool1->getOutput(0), midch, DimsHW{ ksize, ksize }, weightMap[lname + ".features.24.weight"], weightMap[lname + ".features.24.bias"]);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	conv1->setNbGroups(1);
	IActivationLayer* relu1 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

	IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".features.26.weight"], weightMap[lname + ".features.26.bias"]);
	conv2->setStrideNd(DimsHW{ 1, 1 });
	conv2->setPaddingNd(DimsHW{ 1, 1 });
	conv2->setNbGroups(1);
	IActivationLayer* relu2 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);

	IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".features.28.weight"], weightMap[lname + ".features.28.bias"]);
	conv3->setStrideNd(DimsHW{ 1, 1 });
	conv3->setPaddingNd(DimsHW{ 1, 1 });
	conv3->setNbGroups(1);
	IActivationLayer* relu3 = network->addActivation(*conv3->getOutput(0), ActivationType::kRELU);
	IPoolingLayer* pool2 = network->addPoolingNd(*relu3->getOutput(0), PoolingType::kMAX, DimsHW{ 2, 2 });
	return relu3;
}

ILayer* doubleConv1(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, std::string lname, int midch) {
	IConvolutionLayer* conv1 = network->addConvolutionNd(input, midch, DimsHW{ ksize, ksize }, weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	conv1->setNbGroups(1);
	IConvolutionLayer* conv2 = network->addConvolutionNd(*conv1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
	conv2->setStrideNd(DimsHW{ 1, 1 });
	conv2->setPaddingNd(DimsHW{ 1, 1 });
	conv2->setNbGroups(1);
	assert(conv2);

	return conv2;
}
ILayer* up1(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input1, ITensor& input2, int resize, int outch, int midch, std::string lname) {
	float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * resize * 2 * 2));
	for (int i = 0; i < resize * 2 * 2; i++) {
		deval[i] = 1.0;
	}
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	Weights deconvwts1{ DataType::kFLOAT, deval, resize * 2 * 2 };
	IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(input1, resize, DimsHW{ 2, 2 }, deconvwts1, emptywts);
	deconv1->setStrideNd(DimsHW{ 2, 2 });
	deconv1->setNbGroups(resize);
	//weightMap["deconvwts." + lname] = deconvwts1;

	ILayer* pad1 = network->addPaddingNd(*deconv1->getOutput(0), DimsHW{ 0,0 }, DimsHW{ 0, 0 });
	ITensor* inputTensors[] = { &input2,pad1->getOutput(0) };
	auto cat = network->addConcatenation(inputTensors, 2);
	assert(cat);

	IConvolutionLayer* conv1 = network->addConvolutionNd(*cat->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	conv1->setNbGroups(1);
	IConvolutionLayer* conv2 = network->addConvolutionNd(*conv1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
	conv2->setStrideNd(DimsHW{ 1, 1 });
	conv2->setPaddingNd(DimsHW{ 1, 1 });
	conv2->setNbGroups(1);
	assert(conv2);
	return conv2;
}
ILayer* up2(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input1, ITensor& input2, int resize, int outch, int midch, std::string lname) {
	float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * resize * 2 * 2));
	for (int i = 0; i < resize * 2 * 2; i++) {
		deval[i] = 1.0;
	}
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	Weights deconvwts1{ DataType::kFLOAT, deval, resize * 2 * 2 };
	IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(input1, resize, DimsHW{ 2, 2 }, deconvwts1, emptywts);
	deconv1->setStrideNd(DimsHW{ 2, 2 });
	deconv1->setNbGroups(resize);

	ILayer* pad1 = network->addPaddingNd(*deconv1->getOutput(0), DimsHW{ 0,0 }, DimsHW{ 0, 0 });
	ITensor* inputTensors[] = { &input2,pad1->getOutput(0) };
	auto cat = network->addConcatenation(inputTensors, 2);
	assert(cat);

	IConvolutionLayer* conv1 = network->addConvolutionNd(*cat->getOutput(0), resize, DimsHW{ 3, 3 }, weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	conv1->setNbGroups(1);
	IConvolutionLayer* conv2 = network->addConvolutionNd(*conv1->getOutput(0), resize, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
	conv2->setStrideNd(DimsHW{ 1, 1 });
	conv2->setPaddingNd(DimsHW{ 1, 1 });
	conv2->setNbGroups(1);
	assert(conv2);
	return conv2;
}
ILayer* up3(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input1, ITensor& input2, int resize, int outch, int midch, std::string lname) {
	float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * resize * 2 * 2 * 2));
	for (int i = 0; i < resize * 2 * 2 * 2; i++) {
		deval[i] = 1.0;
	}
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	Weights deconvwts1{ DataType::kFLOAT, deval, resize * 2 * 2 * 2 };
	IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(input1, resize * 2, DimsHW{ 2, 2 }, deconvwts1, emptywts);
	deconv1->setStrideNd(DimsHW{ 2, 2 });
	deconv1->setNbGroups(resize * 2);


	ILayer* pad1 = network->addPaddingNd(*deconv1->getOutput(0), DimsHW{ 0,0 }, DimsHW{ 0, 0 });
	ITensor* inputTensors[] = { &input2,pad1->getOutput(0) };
	auto cat = network->addConcatenation(inputTensors, 2);
	assert(cat);

	IConvolutionLayer* conv1 = network->addConvolutionNd(*cat->getOutput(0), resize, DimsHW{ 3, 3 }, weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	conv1->setNbGroups(1);
	IConvolutionLayer* conv2 = network->addConvolutionNd(*conv1->getOutput(0), resize, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
	conv2->setStrideNd(DimsHW{ 1, 1 });
	conv2->setPaddingNd(DimsHW{ 1, 1 });
	conv2->setNbGroups(1);
	assert(conv2);
	return conv2;

}
ILayer* up4(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input1, ITensor& input2, int resize, int outch, int midch, std::string lname) {
	float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * resize * 2 * 2 * 2));
	for (int i = 0; i < resize * 2 * 2 * 2; i++) {
		deval[i] = 1.0;
	}
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	Weights deconvwts1{ DataType::kFLOAT, deval, resize * 2 * 2 * 2 };
	IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(input1, resize * 2, DimsHW{ 2, 2 }, deconvwts1, emptywts);
	deconv1->setStrideNd(DimsHW{ 2, 2 });
	deconv1->setNbGroups(resize * 2);

	ILayer* pad1 = network->addPaddingNd(*deconv1->getOutput(0), DimsHW{ 0,0 }, DimsHW{ 0, 0 });
	ITensor* inputTensors[] = { &input2,pad1->getOutput(0) };
	auto cat = network->addConcatenation(inputTensors, 2);
	assert(cat);
	if (midch == resize) {
		ILayer* dcov1 = doubleConv1(network, weightMap, *cat->getOutput(0), outch, 3, lname, outch);
		assert(dcov1);
		return dcov1;
	}
	else {
		IConvolutionLayer* conv1 = network->addConvolutionNd(*cat->getOutput(0), resize, DimsHW{ 3, 3 }, weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
		conv1->setStrideNd(DimsHW{ 1, 1 });
		conv1->setPaddingNd(DimsHW{ 1, 1 });
		conv1->setNbGroups(1);
		IConvolutionLayer* conv2 = network->addConvolutionNd(*conv1->getOutput(0), resize, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
		conv2->setStrideNd(DimsHW{ 1, 1 });
		conv2->setPaddingNd(DimsHW{ 1, 1 });
		conv2->setNbGroups(1);
		assert(conv2);
		return conv2;
	}
}

ILayer* up5(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input1, ITensor& input2, int resize, int outch, int midch, std::string lname) {
	float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * resize * 2 * 2 * 2));
	for (int i = 0; i < resize * 2 * 2 * 2; i++) {
		deval[i] = 1.0;
	}
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	Weights deconvwts1{ DataType::kFLOAT, deval, resize * 2 * 2 * 2 };
	IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(input1, resize * 2, DimsHW{ 2, 2 }, deconvwts1, emptywts);
	deconv1->setStrideNd(DimsHW{ 2, 2 });
	deconv1->setNbGroups(resize * 2);

	ILayer* pad1 = network->addPaddingNd(*deconv1->getOutput(0), DimsHW{ 0,0 }, DimsHW{ 0, 0 });
	ITensor* inputTensors[] = { &input2,pad1->getOutput(0) };
	auto cat = network->addConcatenation(inputTensors, 2);
	assert(cat);

	IConvolutionLayer* conv1 = network->addConvolutionNd(*cat->getOutput(0), resize, DimsHW{ 3, 3 }, weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	conv1->setNbGroups(1);
	IConvolutionLayer* conv2 = network->addConvolutionNd(*conv1->getOutput(0), resize, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
	conv2->setStrideNd(DimsHW{ 1, 1 });
	conv2->setPaddingNd(DimsHW{ 1, 1 });
	conv2->setNbGroups(1);
	assert(conv2);
	return conv2;
}

ILayer* doubleConv1_V1(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, std::string lname, int midch) {
	IConvolutionLayer* conv1 = network->addConvolutionNd(input, midch, DimsHW{ ksize, ksize }, weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	conv1->setNbGroups(1);

	IActivationLayer* relu11 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

	IConvolutionLayer* conv2 = network->addConvolutionNd(*relu11->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
	conv2->setStrideNd(DimsHW{ 1, 1 });
	conv2->setPaddingNd(DimsHW{ 1, 1 });
	conv2->setNbGroups(1);
	assert(conv2);
	IActivationLayer* relu12 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
	return relu12;
}

ILayer* up1_V1(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input1, ITensor& input2, int resize, int outch, int midch, std::string lname) {
	float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * resize * 2 * 2));
	for (int i = 0; i < resize * 2 * 2; i++) {
		deval[i] = 1.0;
	}
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	Weights deconvwts1{ DataType::kFLOAT, deval, resize * 2 * 2 };
	IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(input1, resize, DimsHW{ 2, 2 }, deconvwts1, emptywts);
	deconv1->setStrideNd(DimsHW{ 2, 2 });
	deconv1->setNbGroups(resize);
	//weightMap["deconvwts." + lname] = deconvwts1;

	ILayer* pad1 = network->addPaddingNd(*deconv1->getOutput(0), DimsHW{ 0,0 }, DimsHW{ 0, 0 });
	ITensor* inputTensors[] = { &input2,pad1->getOutput(0) };
	auto cat = network->addConcatenation(inputTensors, 2);
	assert(cat);

	IConvolutionLayer* conv1 = network->addConvolutionNd(*cat->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	conv1->setNbGroups(1);

	IActivationLayer* relu11 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

	IConvolutionLayer* conv2 = network->addConvolutionNd(*relu11->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
	conv2->setStrideNd(DimsHW{ 1, 1 });
	conv2->setPaddingNd(DimsHW{ 1, 1 });
	conv2->setNbGroups(1);
	assert(conv2);
	IActivationLayer* relu12 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
	return relu12;
}
ILayer* up2_V1(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input1, ITensor& input2, int resize, int outch, int midch, std::string lname) {
	float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * resize * 2 * 2));
	for (int i = 0; i < resize * 2 * 2; i++) {
		deval[i] = 1.0;
	}
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	Weights deconvwts1{ DataType::kFLOAT, deval, resize * 2 * 2 };
	IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(input1, resize, DimsHW{ 2, 2 }, deconvwts1, emptywts);
	deconv1->setStrideNd(DimsHW{ 2, 2 });
	deconv1->setNbGroups(resize);

	ILayer* pad1 = network->addPaddingNd(*deconv1->getOutput(0), DimsHW{ 0,0 }, DimsHW{ 0, 0 });
	ITensor* inputTensors[] = { &input2,pad1->getOutput(0) };
	auto cat = network->addConcatenation(inputTensors, 2);
	assert(cat);

	IConvolutionLayer* conv1 = network->addConvolutionNd(*cat->getOutput(0), resize, DimsHW{ 3, 3 }, weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	conv1->setNbGroups(1);

	IActivationLayer* relu11 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

	IConvolutionLayer* conv2 = network->addConvolutionNd(*relu11->getOutput(0), resize, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
	conv2->setStrideNd(DimsHW{ 1, 1 });
	conv2->setPaddingNd(DimsHW{ 1, 1 });
	conv2->setNbGroups(1);
	assert(conv2);

	IActivationLayer* relu12 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
	return relu12;
}
ILayer* up3_V1(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input1, ITensor& input2, int resize, int outch, int midch, std::string lname) {
	float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * resize * 2 * 2 * 2));
	for (int i = 0; i < resize * 2 * 2 * 2; i++) {
		deval[i] = 1.0;
	}
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	Weights deconvwts1{ DataType::kFLOAT, deval, resize * 2 * 2 * 2 };
	IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(input1, resize * 2, DimsHW{ 2, 2 }, deconvwts1, emptywts);
	deconv1->setStrideNd(DimsHW{ 2, 2 });
	deconv1->setNbGroups(resize * 2);


	ILayer* pad1 = network->addPaddingNd(*deconv1->getOutput(0), DimsHW{ 0,0 }, DimsHW{ 0, 0 });
	ITensor* inputTensors[] = { &input2,pad1->getOutput(0) };
	auto cat = network->addConcatenation(inputTensors, 2);
	assert(cat);

	IConvolutionLayer* conv1 = network->addConvolutionNd(*cat->getOutput(0), resize, DimsHW{ 3, 3 }, weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	conv1->setNbGroups(1);

	IActivationLayer* relu11 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

	IConvolutionLayer* conv2 = network->addConvolutionNd(*relu11->getOutput(0), resize, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
	conv2->setStrideNd(DimsHW{ 1, 1 });
	conv2->setPaddingNd(DimsHW{ 1, 1 });
	conv2->setNbGroups(1);
	assert(conv2);
	IActivationLayer* relu12 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
	return relu12;

}
ILayer* up4_V1(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input1, ITensor& input2, int resize, int outch, int midch, std::string lname) {
	float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * resize * 2 * 2 * 2));
	for (int i = 0; i < resize * 2 * 2 * 2; i++) {
		deval[i] = 1.0;
	}
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	Weights deconvwts1{ DataType::kFLOAT, deval, resize * 2 * 2 * 2 };
	IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(input1, resize * 2, DimsHW{ 2, 2 }, deconvwts1, emptywts);
	deconv1->setStrideNd(DimsHW{ 2, 2 });
	deconv1->setNbGroups(resize * 2);

	ILayer* pad1 = network->addPaddingNd(*deconv1->getOutput(0), DimsHW{ 0,0 }, DimsHW{ 0, 0 });
	ITensor* inputTensors[] = { &input2,pad1->getOutput(0) };
	auto cat = network->addConcatenation(inputTensors, 2);
	assert(cat);
	if (midch == resize) {
		ILayer* dcov1 = doubleConv1_V1(network, weightMap, *cat->getOutput(0), outch, 3, lname, outch);
		assert(dcov1);
		return dcov1;
	}
	else {
		IConvolutionLayer* conv1 = network->addConvolutionNd(*cat->getOutput(0), resize, DimsHW{ 3, 3 }, weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
		conv1->setStrideNd(DimsHW{ 1, 1 });
		conv1->setPaddingNd(DimsHW{ 1, 1 });
		conv1->setNbGroups(1);

		IActivationLayer* relu11 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

		IConvolutionLayer* conv2 = network->addConvolutionNd(*relu11->getOutput(0), resize, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
		conv2->setStrideNd(DimsHW{ 1, 1 });
		conv2->setPaddingNd(DimsHW{ 1, 1 });
		conv2->setNbGroups(1);
		assert(conv2);
		IActivationLayer* relu12 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
		return relu12;
	}
}

ILayer* up5_V1(INetworkDefinition* network, std::unordered_map<std::string, Weights>& weightMap, ITensor& input1, ITensor& input2, int resize, int outch, int midch, std::string lname) {
	float* deval = reinterpret_cast<float*>(malloc(sizeof(float) * resize * 2 * 2 * 2));
	for (int i = 0; i < resize * 2 * 2 * 2; i++) {
		deval[i] = 1.0;
	}
	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	Weights deconvwts1{ DataType::kFLOAT, deval, resize * 2 * 2 * 2 };
	IDeconvolutionLayer* deconv1 = network->addDeconvolutionNd(input1, resize * 2, DimsHW{ 2, 2 }, deconvwts1, emptywts);
	deconv1->setStrideNd(DimsHW{ 2, 2 });
	deconv1->setNbGroups(resize * 2);

	ILayer* pad1 = network->addPaddingNd(*deconv1->getOutput(0), DimsHW{ 0,0 }, DimsHW{ 0, 0 });
	ITensor* inputTensors[] = { &input2,pad1->getOutput(0) };
	auto cat = network->addConcatenation(inputTensors, 2);
	assert(cat);

	IConvolutionLayer* conv1 = network->addConvolutionNd(*cat->getOutput(0), resize, DimsHW{ 3, 3 }, weightMap[lname + ".conv1.weight"], weightMap[lname + ".conv1.bias"]);
	conv1->setStrideNd(DimsHW{ 1, 1 });
	conv1->setPaddingNd(DimsHW{ 1, 1 });
	conv1->setNbGroups(1);

	IActivationLayer* relu11 = network->addActivation(*conv1->getOutput(0), ActivationType::kRELU);

	IConvolutionLayer* conv2 = network->addConvolutionNd(*relu11->getOutput(0), resize, DimsHW{ 3, 3 }, weightMap[lname + ".conv2.weight"], weightMap[lname + ".conv2.bias"]);
	conv2->setStrideNd(DimsHW{ 1, 1 });
	conv2->setPaddingNd(DimsHW{ 1, 1 });
	conv2->setNbGroups(1);
	assert(conv2);
	IActivationLayer* relu12 = network->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
	return relu12;
}


/// <summary>
/// 构建engine
/// </summary>
/// <param name="model_mode"> 选择模型：0是UNet,1是ResNet</param>
/// <param name="maxBatchSize"> 最大batch</param>
/// <param name="builder"> TRT创建的builder</param>
/// <param name="config"> TRT创建的Config</param>
/// <param name="dt"> 数据类型</param>
/// <param name="wts_path"> 权重路径</param>
/// <param name="num_classes"> 缺陷类别数</param>
/// <param name="INPUT_H"> 模型指定推理图像的高</param>
/// <param name="INPUT_W"> 模型指定推理图像的宽</param>
/// <returns> 返回构建好的engine</returns>
ICudaEngine* createEngine_UNet(int model_mode, unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, string wts_path, int num_classes, int INPUT_H, int INPUT_W, int cudaSetPrecision, string train_exe_version) {
	INetworkDefinition* network = builder->createNetworkV2(0U);

	// Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
	ITensor* data = network->addInput(INPUT_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
	assert(data);


	auto start1 = chrono::steady_clock::now();
	std::unordered_map<std::string, Weights> weightMap;
	//这三个版本，转出来的bin文件用了二进制保存，读取方式不一样，之后的版本还是用之前的方式读取。ResNet也一样.
	if (train_exe_version == "1.0.1" || train_exe_version == "1.0.2" || train_exe_version == "1.0.3")
	{
		weightMap = read_weights(wts_path.c_str());
	}
	else
	{
		weightMap = load_weight(wts_path);
	}

	Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
	auto end1 = chrono::steady_clock::now();
	double total_time1 = chrono::duration<double, milli>(end1 - start1).count();
	cout << "load weight: " << total_time1 << " ms" << endl;

	if (train_exe_version == "0")
	{
		if (model_mode == 1)
		{
			std::cout << "wait..." << std::endl;
			// build network
			auto x1 = block1(network, weightMap, *data, 64, 3, "vgg", 64);
			auto x2 = block2(network, weightMap, *x1->getOutput(0), 128, 3, "vgg", 128);
			auto x3 = block3(network, weightMap, *x2->getOutput(0), 256, 3, "vgg", 256);
			auto x4 = block4(network, weightMap, *x3->getOutput(0), 256, 3, "vgg", 256);
			auto x5 = block5(network, weightMap, *x4->getOutput(0), 256, 3, "vgg", 256);

			ILayer* x6 = up1(network, weightMap, *x5->getOutput(0), *x4->getOutput(0), 256, 256, 256, "up_concat4");
			ILayer* x7 = up2(network, weightMap, *x6->getOutput(0), *x3->getOutput(0), 256, 256, 256, "up_concat3");
			ILayer* x8 = up3(network, weightMap, *x7->getOutput(0), *x2->getOutput(0), 128, 128, 128, "up_concat2");
			ILayer* x9 = up4(network, weightMap, *x8->getOutput(0), *x1->getOutput(0), 64, 64, 64, "up_concat1");
			ILayer* x10 = outConv(network, weightMap, *x9->getOutput(0), INPUT_H * INPUT_W * num_classes, "final", num_classes);

			x10->getOutput(0)->setName(OUTPUT_NAME);
			network->markOutput(*x10->getOutput(0));
		}

		if (model_mode == 2)
		{
			std::cout << "wait..." << std::endl;
			// build network
			auto x1 = block1(network, weightMap, *data, 64, 3, "vgg", 64);
			auto x2 = block2(network, weightMap, *x1->getOutput(0), 128, 3, "vgg", 128);
			auto x3 = block3(network, weightMap, *x2->getOutput(0), 256, 3, "vgg", 256);
			auto x4 = block4(network, weightMap, *x3->getOutput(0), 512, 3, "vgg", 512);
			auto x5 = block5(network, weightMap, *x4->getOutput(0), 512, 3, "vgg", 512);

			ILayer* x6 = up1(network, weightMap, *x5->getOutput(0), *x4->getOutput(0), 512, 512, 512, "up_concat4");
			ILayer* x7 = up5(network, weightMap, *x6->getOutput(0), *x3->getOutput(0), 256, 256, 256, "up_concat3");
			ILayer* x8 = up3(network, weightMap, *x7->getOutput(0), *x2->getOutput(0), 128, 128, 128, "up_concat2");
			ILayer* x9 = up4(network, weightMap, *x8->getOutput(0), *x1->getOutput(0), 64, 64, 64, "up_concat1");
			ILayer* x10 = outConv(network, weightMap, *x9->getOutput(0), INPUT_H * INPUT_W * num_classes, "final", num_classes);
			x10->getOutput(0)->setName(OUTPUT_NAME);
			network->markOutput(*x10->getOutput(0));
		}
	}

	else
	{
		if (model_mode == 1)
		{
			std::cout << "wait..." << std::endl;
			// build network
			auto x1 = block1(network, weightMap, *data, 64, 3, "vgg", 64);
			auto x2 = block2(network, weightMap, *x1->getOutput(0), 128, 3, "vgg", 128);
			auto x3 = block3(network, weightMap, *x2->getOutput(0), 256, 3, "vgg", 256);
			auto x4 = block4(network, weightMap, *x3->getOutput(0), 256, 3, "vgg", 256);
			auto x5 = block5(network, weightMap, *x4->getOutput(0), 256, 3, "vgg", 256);

			ILayer* x6 = up1_V1(network, weightMap, *x5->getOutput(0), *x4->getOutput(0), 256, 256, 256, "up_concat4");
			ILayer* x7 = up2_V1(network, weightMap, *x6->getOutput(0), *x3->getOutput(0), 256, 256, 256, "up_concat3");
			ILayer* x8 = up3_V1(network, weightMap, *x7->getOutput(0), *x2->getOutput(0), 128, 128, 128, "up_concat2");
			ILayer* x9 = up4_V1(network, weightMap, *x8->getOutput(0), *x1->getOutput(0), 64, 64, 64, "up_concat1");
			ILayer* x10 = outConv(network, weightMap, *x9->getOutput(0), INPUT_H * INPUT_W * num_classes, "final", num_classes);

			x10->getOutput(0)->setName(OUTPUT_NAME);
			network->markOutput(*x10->getOutput(0));
		}

		if (model_mode == 2)
		{
			std::cout << "wait..." << std::endl;
			// build network
			auto x1 = block1(network, weightMap, *data, 64, 3, "vgg", 64);
			auto x2 = block2(network, weightMap, *x1->getOutput(0), 128, 3, "vgg", 128);
			auto x3 = block3(network, weightMap, *x2->getOutput(0), 256, 3, "vgg", 256);
			auto x4 = block4(network, weightMap, *x3->getOutput(0), 512, 3, "vgg", 512);
			auto x5 = block5(network, weightMap, *x4->getOutput(0), 512, 3, "vgg", 512);

			ILayer* x6 = up1_V1(network, weightMap, *x5->getOutput(0), *x4->getOutput(0), 512, 512, 512, "up_concat4");
			ILayer* x7 = up5_V1(network, weightMap, *x6->getOutput(0), *x3->getOutput(0), 256, 256, 256, "up_concat3");
			ILayer* x8 = up3_V1(network, weightMap, *x7->getOutput(0), *x2->getOutput(0), 128, 128, 128, "up_concat2");
			ILayer* x9 = up4_V1(network, weightMap, *x8->getOutput(0), *x1->getOutput(0), 64, 64, 64, "up_concat1");
			ILayer* x10 = outConv(network, weightMap, *x9->getOutput(0), INPUT_H * INPUT_W * num_classes, "final", num_classes);
			x10->getOutput(0)->setName(OUTPUT_NAME);
			network->markOutput(*x10->getOutput(0));
		}
	}


	// Build engine
	builder->setMaxBatchSize(maxBatchSize);
	config->setMaxWorkspaceSize(1 << 20);  // 16MB


	if (cudaSetPrecision == 16)
	{
		config->setFlag(BuilderFlag::kFP16);
	}

	auto start2 = chrono::steady_clock::now();
	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

	auto end2 = chrono::steady_clock::now();
	double total_time2 = chrono::duration<double, milli>(end2 - start2).count();
	cout << "build engine: " << total_time2 << " ms" << endl;
	//ICudaEngine* engine = builder->buildCudaEngine(*network);
	// Don't need the network any more
	network->destroy();

	// Release host memory
	for (auto& mem : weightMap)
	{
		free((void*)(mem.second.values));
	}
	return engine;
}


/// <summary>
/// 构建引擎
/// </summary>
/// <param name="mode_select"> 使用的模型</param>
/// <param name="maxBatchSize"> 最大batch</param>
/// <param name="wts_path"> bin模型的路径</param>
/// <param name="num_classes"> 类别</param>
/// <param name="input_H"> 输入图像的高</param>
/// <param name="input_W"> 输入图像的宽</param>
/// <returns> 返回构建好的引擎</returns>
ICudaEngine* APIToModel_UNet(int mode_select, unsigned int maxBatchSize, string wts_path, int num_classes, int input_H, int input_W, int cudaSetPrecision, string train_exe_version)
{
	// Create builder
	IBuilder* builder = createInferBuilder(gLogger);
	IBuilderConfig* config = builder->createBuilderConfig();
	// Create model to populate the network, then set the outputs and create an engine
	// ICudaEngine* engine = (CREATENET(NET))(maxBatchSize, builder, config, DataType::kFLOAT);

	ICudaEngine* engine = createEngine_UNet(mode_select, maxBatchSize, builder, config, DataType::kFLOAT, wts_path, num_classes, input_H, input_W, cudaSetPrecision, train_exe_version);
	assert(engine != nullptr);

	// Close everything down
	builder->destroy();
	config->destroy();

	return engine;
}

/// <summary>
/// UTF转GB
/// </summary>
/// <param name="str">字符指针</param>
/// <returns></returns>
string UTF8ToGB(const char* str)
{
	string result;
	WCHAR* strSrc;
	LPSTR szRes;

	//获得临时变量的大小
	int i = MultiByteToWideChar(CP_UTF8, 0, str, -1, NULL, 0);
	strSrc = new WCHAR[i + 1];
	MultiByteToWideChar(CP_UTF8, 0, str, -1, strSrc, i);

	//获得临时变量的大小
	i = WideCharToMultiByte(CP_ACP, 0, strSrc, -1, NULL, 0, NULL, NULL);
	szRes = new CHAR[i + 1];
	WideCharToMultiByte(CP_ACP, 0, strSrc, -1, szRes, i, NULL, NULL);

	result = szRes;
	delete[]strSrc;
	delete[]szRes;

	return result;
}


int startsWith(string s, string sub) {
	return s.find(sub) == 0 ? 1 : 0;
}

int endsWith(string s, string sub) {
	return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
}

vector<string> split(const string& str, const string& delim) {
	vector<string> res;
	if ("" == str) return res;
	//先将要切割的字符串从string类型转换为char*类型
	char* strs = new char[str.length() + 1];
	strcpy(strs, str.c_str());

	char* d = new char[delim.length() + 1];
	strcpy(d, delim.c_str());

	char* p = strtok(strs, d);
	while (p) {
		string s = p; //分割得到的字符串转换为string类型
		res.push_back(s); //存入结果数组
		p = strtok(NULL, d);
	}

	return res;
}



/// <summary>
/// 加载bin文件，解析权重，在当前的显卡上构建引擎。
/// </summary>
/// <param name="root_path"> bin文件所在根目录</param>
/// <param name="model_num"> 确定要加载的是哪个模型</param>
/// <returns> 以结构体的形式返回构建的engine,该模型要推理图像的高宽以及类别</returns>
return_data convert_bin(string root_path, int model_num, int cudaSetPrecision)
{
	ICudaEngine* eng;
	int mode = 0;
	int INPUT_h = 0;
	int INPUT_w = 0;
	int cls2 = 2;
	float threshold = 0.5;
	int area_threshold = 100;

	string wts_path = "";
	string engine_path = "";
	string train_exe_version = "0";
	char* trtModelStream{ nullptr };


	if (model_num == 0)
	{
		//Unet, 对应的encode.txt; encode.bin
		fstream in(root_path + "\\encode.txt");
		string s;
		if (in.fail())
		{
			cout << "open file error" << endl;
		}

		wts_path = root_path + "\\encode.bin";
		engine_path = root_path;
		while (std::getline(in, s), '\n')
		{
			string str = UTF8ToGB(s.c_str()).c_str();
			if (startsWith(str, "num_classes"))
			{
				std::vector<string> res = split(str, ":");
				cls2 = stoi(res[1]);
			}

			else if (startsWith(str, "image_H_W"))
			{
				std::vector<string> res = split(str, ":");
				std::vector<string> res1 = split(res[1], ",");
				INPUT_h = stoi(res1[0]);
				INPUT_w = stoi(res1[1]);
			}

			else if (startsWith(str, "pixel_value"))
			{
				std::vector<string> res = split(str, ":");
				threshold = stof(res[1]);
			}

			else if (startsWith(str, "area_value"))
			{
				std::vector<string> res = split(str, ":");
				area_threshold = stoi(res[1]);
			}

			else if (startsWith(str, "select_mode"))
			{
				std::vector<string> res = split(str, ":");
				string train_mode = res[1];
				//根据配置文件确定要转化的模型是哪一种；
				if (train_mode == "LT") {
					mode = 1;
				}
				else if (train_mode == "MT")
				{
					mode = 2;
				}
			}

			else if (startsWith(str, "version"))
			{
				std::vector<string> res = split(str, ":");
				train_exe_version = res[1];
			}

			else if (startsWith(str, "END"))
			{
				break;
			}
		}
		in.close();
	}

	else if (model_num == 1)
	{

		fstream in(root_path + "\\encode_R.txt");
		string s;
		if (in.fail())
		{
			cout << "open file error" << endl;
		}

		wts_path = root_path + "\\encode_R.bin";
		engine_path = root_path;

		while (getline(in, s), '\n')
		{
			string str = UTF8ToGB(s.c_str()).c_str();
			if (startsWith(str, "image_H_W"))
			{
				std::vector<string> res = split(str, ":");
				std::vector<string> ree = split(res[1], ",");
				INPUT_h = stoi(ree[0]);
				INPUT_w = stoi(ree[1]);
			}
			else if (startsWith(str, "pixel_value"))
			{
				std::vector<string> res_tred = split(str, ":");
				threshold = stof(res_tred[1]);
			}
			else if (startsWith(str, "select_mode"))
			{
				std::vector<string> res = split(str, ":");
				string train_mode = res[1];
				//根据配置文件确定要转化的模型是哪一种；
				if (train_mode == "LT") {
					mode = 1;
				}
				else if (train_mode == "MT")
				{
					mode = 2;
				}
			}
			else if (startsWith(str, "version"))
			{
				std::vector<string> res = split(str, ":");
				train_exe_version = res[1];
			}
			else if (startsWith(str, "END"))
			{
				break;
			}
		}in.close();
	}


	//UNet
	if (model_num == 0)
	{
		int OUTPUT_SIZE = cls2 * INPUT_h * INPUT_w;
		eng = APIToModel_UNet(mode, 1, wts_path, cls2, INPUT_h, INPUT_w, cudaSetPrecision, train_exe_version);
		get_par.eng = eng;
		get_par.img_h = INPUT_h;
		get_par.img_w = INPUT_w;
		get_par.cls = cls2;
	}

	//ResNet
	if (model_num == 1)
	{
		eng = APIToModel_Res(mode, 1, wts_path, INPUT_h, INPUT_w, cudaSetPrecision, train_exe_version);

		get_par.eng = eng;
		get_par.img_h = INPUT_h;
		get_par.img_w = INPUT_w;
		get_par.cls = 2;
	}

	return get_par;
}
