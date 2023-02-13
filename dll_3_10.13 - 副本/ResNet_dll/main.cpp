#include "windows.h"
#include <iostream>
#include <chrono>
#include <time.h>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"
#include <opencv2/opencv.hpp>
#include <regex>
#include <algorithm>
#include <chrono>
#include <highgui/highgui_c.h>
#include "Net.h"
#include <ctime>
#include <device_launch_parameters.h>
#include <io.h>
#include <direct.h>
#include <omp.h>
#include <sstream>
#include "encrypt_bin2seq.h"

#define DEVICE0 0
#define DEVICE1 1
#define OK 0;
#define ERR -1;
#define KEY 0x9b;
#define BATCH_SIZE 1

const string DLL_VERSION = "1.0.7"; // ��ǰ���ɵ�DLL�汾��;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
const int model_select = 0;
int tcount = 0;
int true_classes = 0;
int deviceCount = 0;

cv::Mat uchar2Mat[15][12];
bool Flags = false;
bool Handle_Flags = false;
bool WARM_UP_FLAGS = false;


static Logger gLogger;
using namespace std;
using namespace cv;
using namespace nvinfer1;


struct EngineParameter_
{
	float* tdata;
	float* prob;
	IRuntime* runtime;
	ICudaEngine* engine;
	IExecutionContext* context;
	cudaStream_t stream;
	void* buffer[2];

	int input_index;
	int output_index;
};

EngineParameter_ struct_cam[15][12] = { 0 };

template<class T>
int length(T& arr)
{
	return sizeof(arr) / sizeof(arr[0]);
}

struct axi
{
	float x, y;
}Num[4] = { 0 };

struct Scaling_ratio
{
	int top;
	int bottom;
	int left;
	int right;
	double ratio;
}get_ratio = { 0 };

struct return_data
{
	ICudaEngine* eng = nullptr;
	int img_h;
	int img_w;
	int cls;
} sEng[15][12] = { 0 };

list<return_data> aa;

/// <summary>
/// ��������Ĺ���
/// </summary>
/// <param name="ax_A"> ��һ�������ֵ</param>
/// <param name="ax_B"> �ڶ��������ֵ</param>
/// <returns>��С���󷵻������ֵ</returns>
float cmp(axi ax_A, axi ax_B)
{
	if (ax_A.x == ax_B.x)
		return ax_A.y < ax_B.y;
	return ax_A.x < ax_B.x;
};


/// <summary>
/// ͼ��Ԥ����
/// </summary>
/// <param name="img"> ��Ҫ�����ͼ��</param>
/// <param name="data"> ����������ָ�룬���������ʽ���resize��ͼ���ֵ</param>
/// <param name="in_H"> resize��ͼ��ĸ�</param>
/// <param name="in_W"> resize��ͼ��Ŀ�</param>
void proccess_img_ResNet(cv::Mat img, float* data, int in_H, int in_W)
{
	cv::Mat imgDst;
	cv::resize(img, imgDst, cv::Size(in_W, in_H), cv::INTER_CUBIC);

	//�ȶ�ֵ�����ٹ�һ�� ---> ����ֵ�����Ա�׼��
	int i = 0;
	for (int row = 0; row < in_H; ++row)
	{
		uchar* uc_pixel = imgDst.data + row * imgDst.step;
		for (int col = 0; col < in_W; ++col) {
			data[i] = ((float)uc_pixel[2] / 255 - 0.485) / 0.229;
			data[i + in_H * in_W] = ((float)uc_pixel[1] / 255 - 0.456) / 0.224;
			data[i + 2 * in_H * in_W] = ((float)uc_pixel[0] / 255 - 0.406) / 0.225;
			uc_pixel += 3;
			++i;
		}
	}
}


/// <summary>
/// UNet�����ͼ��Ԥ������
/// </summary>
/// <param name="img"> ��Ҫ�����ͼ��</param>
/// <param name="data"> ����������ָ�룬���������ʽ���resize��ͼ���ֵ</param>
/// <param name="in_H"> resize��ͼ��ĸ�</param>
/// <param name="in_W"> resize��ͼ��Ŀ�</param>
/// <returns> �Խṹ���������ʽ����resize����������ͼ��ֱ���ٵĿ�ȣ�������ͼ������ű���</returns>
Scaling_ratio proccess_img_UNet(cv::Mat img, float* data, int in_H, int in_W)
{
	double ih = img.rows;
	double iw = img.cols;
	double scale = min(in_H / ih, in_W / iw);
	int nw = int(iw * scale);
	if (nw < in_W) nw += 1;
	int nh = int(ih * scale);


	int topborder = (in_H - nh) / 2;
	int bootomborder = (in_H - nh) / 2;
	int leftborder = (in_W - nw) / 2;
	int rightborder = (in_W - nw) / 2;

	if ((in_H - nh) % 2 != 0)
	{
		bootomborder += 1;
	}
	if ((in_W - nw) % 2 != 0)
	{
		rightborder += 1;
	}

	//cv::Mat new_img;
	cv::resize(img, img, Size(nw, nh));
	Mat U_img = Mat::zeros(in_H, in_W, CV_8UC3);
	cv::copyMakeBorder(img, U_img, topborder, bootomborder, leftborder, rightborder, cv::BORDER_CONSTANT, cv::Scalar::all(0));

	int i = 0;
	for (int row = 0; row < in_H; ++row)
	{
		uchar* uc_pixel = U_img.data + row * U_img.step;
		for (int col = 0; col < in_W; ++col) {
			data[i] = (float)uc_pixel[2] / 255.0;
			data[i + in_H * in_W] = (float)uc_pixel[1] / 255.0;
			data[i + 2 * in_H * in_W] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;
		}
	}

	Scaling_ratio total;
	total.top = (in_H - nh) / 2;
	total.bottom = (in_H - nh) / 2;
	total.left = (in_W - nw) / 2;
	total.right = (in_W - nw) / 2;
	total.ratio = scale;
	return  total;
}


/// <summary>
/// �����������к����֣�����������������ֵӳ�䵽��0,1�������ڣ���ÿ����ʡ�
/// </summary>
/// <param name="in_H"></param>
/// <param name="in_W"></param>
/// <param name="tcls_num"></param>
/// <param name="tprob"></param>
/// <param name="tout"></param>
void softmax_UNet(int in_H, int in_W, int tcls_num, float* tprob, double* tout)
{
	for (int t = 0; t < in_H * in_W; t++)
	{
		float sum1 = 0.;
		for (int p = 0; p < tcls_num; p++)
		{
			sum1 += exp(tprob[t + in_H * in_W * p]);
		}

		for (int y = 0; y < tcls_num; y++)
		{
			tout[t + in_H * in_W * y] = tprob[t + in_H * in_W * y] / sum1;
		}

	}
}

/// <summary>
/// �ҳ���ͨ��ͼ���е�ÿ������ֵ��ߵ���һ��ͨ����������ͨ�����ر��255
/// </summary>
/// <param name="in_H"> ����ͼ��ĸ�</param>
/// <param name="in_W"> ����ͼ��Ŀ�</param>
/// <param name="tcls_num"> ѵ����ͼ�������</param>
/// <param name="threshold"> ������ֵ</param>
/// <param name="out"> ��������ͨ������������ָ��</param>
void argmax_UNet(int in_H, int in_W, int tcls_num, float threshold, float* out)
{
	for (int t = 0; t < in_H * in_W; t++)
	{
		double max_value = -INFINITY;
		int count_c = 0;
		for (int k = 0; k < tcls_num; k++)
		{
			if (max_value < out[t + in_H * in_W * k] && out[t + in_H * in_W * k] > threshold)
			{
				max_value = out[t + in_H * in_W * k];
				count_c = k;
			}
		}
		out[t] = count_c;

		for (int v = 0; v < tcls_num; v++)
		{
			if (v == count_c && v != 0)
			{
				out[t + in_H * in_W * v] = 255;
			}
			else {
				out[t + in_H * in_W * v] = 0;
			}
		}
	}
}

/// <summary>
/// �쳣������
/// </summary>
/// <param name="err"></param> ���񵽵Ĵ�����Ϣ
/// <param name="file"></param> �������ڵ��ļ�
/// <param name="line"></param> �������ڵ���
static void HandleError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess) {
		Handle_Flags = true;
		ofstream receiveData;
		receiveData.open("D:\\cudaError.txt", ios::app);
		time_t now = time(0);
		receiveData << ctime(&now) << "  Error: " << int(err) << "---" << cudaGetErrorString(err) << "in " << file << "  at line: " << line << endl;
		receiveData.close();
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__))


/// <summary>
/// UNet����Ҫ������
/// </summary>
/// <param name="context"> cuda�й�����������</param>
/// <param name="engine"> cuda�й���������</param>
/// <param name="stream"> cuda��</param>
/// <param name="buffers"> cuda�Ļ���ռ�</param>
/// <param name="input"> �����Ҫ�����ͼ��</param>
/// <param name="output"> �����������</param>
/// <param name="in_H"> ����ͼ��ĸ�</param>
/// <param name="in_W"> ����ͼ��Ŀ�</param>
/// <param name="select_gpu"> ָ������Ҫ�õ�gpu�����Ҫ��LoadFileʱ��gpu��Ӧ�ϣ����򱨴�</param>
void doInference_ResNet(IExecutionContext& context, ICudaEngine& engine, cudaStream_t stream, void* buffers[2],
	float* input, float* output, int in_H, int in_W, int select_gpu, int inputIndex, int outputIndex) {

	if (deviceCount >= select_gpu)
	{
		if (select_gpu == 1)
		{
			cudaSetDevice(DEVICE0);
		}

		else if (select_gpu == 2)
		{
			cudaSetDevice(DEVICE1);
		}
	}
	else
	{
		cudaSetDevice(DEVICE0);
	}

	HANDLE_ERROR(cudaMemcpy(buffers[inputIndex], input, 1 * 3 * in_H * in_W * sizeof(float), cudaMemcpyHostToDevice));
	//HANDLE_ERROR(cudaStreamSynchronize(stream), cam, thrd);
	context.enqueue(1, buffers, stream, nullptr);
	HANDLE_ERROR(cudaMemcpy(output, buffers[outputIndex], 1 * 2 * sizeof(float), cudaMemcpyDeviceToHost));
	//HANDLE_ERROR(cudaStreamSynchronize(stream), cam, thrd);

	//HANDLE_ERROR(cudaMemcpyAsync(buffers[inputIndex], input, 1 * 3 * in_H * in_W * sizeof(float), cudaMemcpyHostToDevice, stream),cam,thrd);
	////HANDLE_ERROR(cudaStreamSynchronize(stream), cam, thrd);
	//context.enqueue(1, buffers, stream, nullptr);	
	//HANDLE_ERROR(cudaMemcpyAsync(output, buffers[outputIndex], 1 * 2 * sizeof(float), cudaMemcpyDeviceToHost, stream), cam, thrd);
	////HANDLE_ERROR(cudaStreamSynchronize(stream), cam, thrd);

}

/// <summary>
/// ResNet����Ҫ������
/// </summary>
/// <param name="context"> cuda�й�����������</param>
/// <param name="engine"> cuda�й���������</param>
/// <param name="input"> �����Ҫ�����ͼ��</param>
/// <param name="output"> �����������</param>
/// <param name="batchSize"> һ��������</param>
/// <param name="in_H"> ����ͼ��ĸ�</param>
/// <param name="in_W"> ����ͼ��Ŀ�</param>
/// <param name="out_size"> ����������������</param>
void doInference_UNet(IExecutionContext& context, ICudaEngine& engine, cudaStream_t stream, void* buffers[2], float* input,
	float* output, int batchSize, int in_H, int in_W, int out_size, int select_gpu, int inputIndex, int outputIndex)
{

	if (select_gpu == 1)
	{
		cudaSetDevice(DEVICE0);
	}

	else
	{
		cudaSetDevice(DEVICE1);
	}

	//engine = context.getEngine();
	//assert(engine.getNbBindings() == 2);
	//const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
	//const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	HANDLE_ERROR(cudaMemcpy(buffers[inputIndex], input, batchSize * 3. * in_H * in_W * sizeof(float), cudaMemcpyHostToDevice));
	context.enqueue(batchSize, buffers, stream, nullptr);
	HANDLE_ERROR(cudaMemcpy(output, buffers[outputIndex], 1. * batchSize * out_size * sizeof(float), cudaMemcpyDeviceToHost));
	//��ͬ����ͨ��cudaStreamSynchronize()��Э����
	cudaStreamSynchronize(stream);

	// Release stream and buffers
	/*cudaStreamDestroy(stream);
	HANDLE_ERROR(cudaFree(buffers[inputIndex]));
	HANDLE_ERROR(cudaFree(buffers[outputIndex]));*/
}

/// <summary>
/// ָ���������õ��Կ�
/// </summary>
/// <param name="deviceCount"></param> �����Կ�����
/// <param name="select_gpu"></param> ѡ����Կ����
void Match_device(int deviceCount, int select_gpu)
{
	if (deviceCount >= select_gpu)
	{
		if (select_gpu == 1)
		{
			cudaSetDevice(DEVICE0);
		}
		else if (select_gpu == 2)
		{
			cudaSetDevice(DEVICE1);
		}
	}
	else
	{
		cudaSetDevice(DEVICE0);
	}
}

/// <summary>
/// ģ��Ԥ�Ȼ��ƣ�������ʱ�Ὺ��һ���̣߳�������������Ĭ��ʹ�õ���cam:12, thread:12��
/// </summary>
/// <param name="num_classes"></param>  ȱ�������
/// <param name="INPUT_h"></param> resize�������ͼ��ĸ�
/// <param name="INPUT_w"></param>resize�������ͼ��Ŀ�
/// <param name="model_select"></param> ģ��ѡ��0ΪUNet,1ΪResNet
void warm_up(int num_classes, int INPUT_h, int INPUT_w, int select_gpu)
{
#pragma omp parallel num_threads(1)  //����1���̣߳�ÿ���̶߳�������{}�ڴ���
	{
	cv:Mat src = cv::Mat::zeros(INPUT_h, INPUT_w, CV_8UC3);

		int cam_cls = 12;
		int cam_thread = 12;
		cout << "warm up " << endl;
		int out_size = num_classes * INPUT_h * INPUT_w;
		cv::Mat black_img;
		black_img = Mat::zeros(INPUT_h, INPUT_w, CV_8UC3);
		while (true)
		{
			Sleep(4000);

			get_ratio = proccess_img_UNet(black_img, struct_cam[cam_cls][cam_thread].tdata, INPUT_h, INPUT_h);
			doInference_UNet(*struct_cam[cam_cls][cam_thread].context,
				*struct_cam[cam_cls][cam_thread].engine,
				struct_cam[cam_cls][cam_thread].stream,
				struct_cam[cam_cls][cam_thread].buffer,
				struct_cam[cam_cls][cam_thread].tdata,
				struct_cam[cam_cls][cam_thread].prob,
				1, INPUT_h, INPUT_h, out_size, select_gpu,
				struct_cam[cam_cls][cam_thread].input_index,
				struct_cam[cam_cls][cam_thread].output_index);
		}
	}
}


extern"C"
{
	/*******************************************************
		-�������ƣ�Loadfile

		-�������������ڿ���ģ����������棻

		-���������
				   file_path: �����ļ���·����
					 cam_cls: �����ĵڼ��������
				  cam_thread: ������Ҫ�������߳�(��0��ʼ����)��
				model_select:
							  ������Ϊ0ʱ��ʹ��UNet;
							  ������Ϊ1ʱ��ʹ��ResNet;

		-����������ޣ��˺��������ǿ������棬�޲������;
	*******************************************************/
	_declspec (dllexport)void Loadfile(const char* file_path,
		int cam_cls,
		int cam_thread,
		int select_gpu,
		int model_select)

	{
		int cudaSetPrecision = 32;
		int dev;
		//�ж��Կ�����
		Flags = true;
		char* trtModelStream{ nullptr };
		cudaGetDeviceCount(&deviceCount);
		int INPUT_h = 0;
		int INPUT_w = 0;
		int num_classes = 2;
		//ָ���������Կ����豸���ڵ��Կ���Ӧ��
		Match_device(deviceCount, select_gpu);

		//ʶ��ǰ�Կ��������Զ�����FP16��FP32
		for (dev = 0; dev < deviceCount; dev++)
		{
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, dev);

			if (deviceProp.major > 6)
			{
				cudaSetPrecision = 16;
			}
			else
			{
				cudaSetPrecision = 32;
			}
		}
		cout << cudaSetPrecision << endl;
		//����bin�ļ���
		string root_path = file_path;

		if (model_select == 0)
		{
			//Unet, ��Ӧ��encode.txt; encode.bin
			fstream in(root_path + "\\encode.txt");
			string s;
			if (in.fail())
			{
				cout << "open file error" << endl;
			}

			while (std::getline(in, s), '\n')
			{
				string str = UTF8ToGB(s.c_str()).c_str();
				if (startsWith(str, "image_H_W"))
				{
					std::vector<string> res = split(str, ":");
					std::vector<string> res1 = split(res[1], ",");
					INPUT_h = stoi(res1[0]);
					INPUT_w = stoi(res1[1]);
				}
				else if (startsWith(str, "num_classes"))
				{
					std::vector<string> res1 = split(str, ":");
					num_classes = stoi(res1[1]);
				}
				else if (startsWith(str, "true_classes"))
				{
					std::vector<string> res1 = split(str, ":");
					true_classes = stoi(res1[1]);
				}
				else if (startsWith(str, "END"))
				{
					break;
				}
			}
			in.close();

		}

		else if (model_select == 1)
		{

			fstream in(root_path + "\\encode_R.txt");
			string s;
			if (in.fail())
			{
				cout << "open file error" << endl;
			}

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
				else if (startsWith(str, "END"))
				{
					break;
				}
			}in.close();
		}


		string engine_name;
		if (model_select == 0)
		{
			engine_name = root_path + "\\encode.seq";
		}
		else if (model_select == 1)
		{
			engine_name = root_path + "\\encode_R.seq";
		}


		size_t size{ 0 };
		//�ж��ļ����Ƿ�����Ѿ����л��õ�ģ���ļ������û�о����л�һ�Σ����������
		if ((_access(engine_name.c_str(), 0)) == -1)
		{
			//���л�ģ��
			sEng[cam_cls][cam_thread] = convert_bin(root_path, model_select, cudaSetPrecision);

			IHostMemory* model_stream{ nullptr };
			assert(sEng[cam_cls][cam_thread].eng != nullptr);
			model_stream = sEng[cam_cls][cam_thread].eng->serialize();

			char* strc = new char[model_stream->size()];
			memcpy(strc, model_stream->data(), model_stream->size());

			Makecode(strc, model_stream->size());//�����л������������ļ����м���

			//������ܺ��seq
			std::ofstream p(engine_name, ios::binary);
			p.write(strc, model_stream->size());
			p.close();
			//char str[] = "guoke3915";
			//printf2char(str, strlen(str) + 1);
			//char *kakala = new char[serialize_str.size() + 1];
			//strncpy(kakala, serialize_str.c_str(), serialize_str.size() + 1);
			//int encLen = 0;
			//
			//cout << serialize_str.size() + 1 << endl;
			//char* enc = (char*)aes->Encrypt(kakala, serialize_str.size() + 1, encLen, false);
			////printf2char(enc, encLen);	
			//std::ofstream p(engine_name);
			//if (!p)
			//{
			//	std::cerr << "could not open plan output file" << std::endl;
			//}
			//cout << "write..." << endl;
			//p.write(enc, strlen(enc));
			//cout << strlen(enc)<< endl;

			model_stream->destroy();
			sEng[cam_cls][cam_thread].eng->destroy();
		}

		//�����л�
		char* trtmodel{ nullptr };
		std::ifstream file(engine_name, ios::binary);
		if (file.good()) {
			file.seekg(0, file.end);
			size = file.tellg();
			file.seekg(0, file.beg);
			trtmodel = new char[size];
			assert(trtmodel);
			file.read(trtmodel, size);
			file.close();
		}

		/*ifstream  readFile;
		readFile.open(engine_name, ios::binary);
		std::stringstream  streambuffer;
		streambuffer << readFile.rdbuf();
		string sdata(streambuffer.str());
		int nLen = sdata.length();*/

		Cutecode(trtmodel, size);//�����л������������ļ����н���

		struct_cam[cam_cls][cam_thread].runtime = createInferRuntime(gLogger);
		assert(struct_cam[cam_cls][cam_thread].runtime != nullptr);
		struct_cam[cam_cls][cam_thread].engine = struct_cam[cam_cls][cam_thread].runtime->deserializeCudaEngine(trtmodel, size);
		assert(struct_cam[cam_cls][cam_thread].engine != nullptr);
		struct_cam[cam_cls][cam_thread].context = struct_cam[cam_cls][cam_thread].engine->createExecutionContext();
		assert(struct_cam[cam_cls][cam_thread].context != nullptr);
		delete[] trtmodel;

		assert(struct_cam[cam_cls][cam_thread].engine->getNbBindings() == 2);

		struct_cam[cam_cls][cam_thread].input_index = struct_cam[cam_cls][cam_thread].engine->getBindingIndex(INPUT_BLOB_NAME);
		struct_cam[cam_cls][cam_thread].output_index = struct_cam[cam_cls][cam_thread].engine->getBindingIndex(OUTPUT_BLOB_NAME);

		if (model_select == 0)
		{
			// Create GPU buffers on device
			HANDLE_ERROR(cudaMalloc((void**)&struct_cam[cam_cls][cam_thread].buffer[struct_cam[cam_cls][cam_thread].input_index], 3. * INPUT_h * INPUT_w * sizeof(float)), cam_cls, cam_thread);
			HANDLE_ERROR(cudaMalloc((void**)&struct_cam[cam_cls][cam_thread].buffer[struct_cam[cam_cls][cam_thread].output_index], 1. * num_classes * INPUT_h * INPUT_w * sizeof(float)), cam_cls, cam_thread);

			// Create stream
			HANDLE_ERROR(cudaStreamCreate(&struct_cam[cam_cls][cam_thread].stream), cam_cls, cam_thread);

			HANDLE_ERROR(cudaMallocHost((void**)&struct_cam[cam_cls][cam_thread].tdata, 3 * INPUT_h * INPUT_w * sizeof(float)), cam_cls, cam_thread);
			HANDLE_ERROR(cudaMallocHost((void**)&struct_cam[cam_cls][cam_thread].prob, 1. * num_classes * INPUT_h * INPUT_w * sizeof(float)), cam_cls, cam_thread);
		}
		else if (model_select == 1)
		{
			// gpu�ϴ���������
			HANDLE_ERROR(cudaMalloc((void**)&struct_cam[cam_cls][cam_thread].buffer[struct_cam[cam_cls][cam_thread].input_index], 1. * 3 * INPUT_h * INPUT_w * sizeof(float)), cam_cls, cam_thread);
			HANDLE_ERROR(cudaMalloc((void**)&struct_cam[cam_cls][cam_thread].buffer[struct_cam[cam_cls][cam_thread].output_index], 1. * 2 * sizeof(float)), cam_cls, cam_thread);

			// ����stream
			HANDLE_ERROR(cudaStreamCreate(&struct_cam[cam_cls][cam_thread].stream), cam_cls, cam_thread);

			HANDLE_ERROR(cudaMallocHost((void**)&struct_cam[cam_cls][cam_thread].tdata, 3. * INPUT_h * INPUT_w * sizeof(float)), cam_cls, cam_thread);
			HANDLE_ERROR(cudaMallocHost((void**)&struct_cam[cam_cls][cam_thread].prob, 2. * sizeof(float)), cam_cls, cam_thread);
		}

		/***************************************************
			Ŀ�ģ�
				���ģ��ͣһ��֮��������ʱ����һ��ͼƬ������ʱ��ܳ��������
			���������
				Ԥ�Ȼ��ƣ�ģ�͵�һ�μ���ʱ����������һ�У�
				�Ͷ������һ��ģ�ͣ�Ȼ���ģ��ÿ��4s����һ�£�
				Ŀ���ǲ���gpuֹͣ���У��Ӷ��ﵽĿ�ģ�
				���Ǹ÷������������һ��ģ�͵��Դ��С��
		******************************************************/

		if (model_select == 0 && WARM_UP_FLAGS == false)
		{
			cout << "get in " << endl;

			WARM_UP_FLAGS = true;

			struct_cam[12][12].engine = struct_cam[cam_cls][cam_thread].engine;

			struct_cam[12][12].runtime = createInferRuntime(gLogger);

			struct_cam[12][12].context = struct_cam[12][12].engine->createExecutionContext();
			assert(struct_cam[12][12].context != nullptr);

			struct_cam[12][12].input_index = struct_cam[12][12].engine->getBindingIndex(INPUT_BLOB_NAME);
			struct_cam[12][12].output_index = struct_cam[12][12].engine->getBindingIndex(OUTPUT_BLOB_NAME);

			// Create GPU buffers on device
			HANDLE_ERROR(cudaMalloc((void**)&struct_cam[12][12].buffer[struct_cam[12][12].input_index], 3. * INPUT_h * INPUT_w * sizeof(float)), 12, 12);
			HANDLE_ERROR(cudaMalloc((void**)&struct_cam[12][12].buffer[struct_cam[12][12].output_index], 1. * num_classes * INPUT_h * INPUT_w * sizeof(float)), 12, 12);

			// Create stream
			HANDLE_ERROR(cudaStreamCreate(&struct_cam[12][12].stream), 12, 12);

			HANDLE_ERROR(cudaMallocHost((void**)&struct_cam[12][12].tdata, 3 * INPUT_h * INPUT_w * sizeof(float)), 12, 12);
			HANDLE_ERROR(cudaMallocHost((void**)&struct_cam[12][12].prob, 1. * num_classes * INPUT_h * INPUT_w * sizeof(float)), 12, 12);

			thread task1(warm_up, num_classes, INPUT_h, INPUT_w, select_gpu);
			task1.detach();
		}
	}

	/*******************************************************************************************************************************************
		-�������ƣ�Result

		-�������������ڶ������ͼ�������������������

		-���������
							  img1: ������Ҫ�����ͼƬ;
						camera_num: �ڵڼ�������Ͻ�������;
						thread_num: �ڵڼ����߳��Ͻ�������;
						select_gpu: �������ĸ�GPU��(Ҫ��Loadfileʱѡ���һ��);
					defect_classes��ѵ��ʱ�ֶ����õ�ȱ������;
						 threshold���趨���������ֵ;
					area_threshold: �����ֵ(ֻ��UNet������);
						   width_1: ԭͼ�Ŀ�;
						  height_1: ԭͼ�ĸ�;
						   INPUT_H: ģ��Ҫ�������ͼ��ĸ�;
						   INPUT_W: ģ��Ҫ�������ͼ��Ŀ�.
					  model_select:
									������Ϊ0ʱ��ʹ��UNet;
									������Ϊ1ʱ��ʹ��ResNet;
							  envs: Ϊ0ʱ�����ȱ�ݵ���������;Ϊ1ʱ�����ģ�͵���Ӿ�������.

		-�������������ļ��Ľ��,�����dll˵���ĵ�;

	*********************************************************************************************************************************************/

	_declspec (dllexport)char* Result(BYTE* img1,
		int camera_num,
		int thread_num,
		int select_gpu,
		int defect_classes,
		float threshold,
		int area_threshold,
		int width_1,
		int height_1,
		int INPUT_H,
		int INPUT_W,
		int model_select,
		int envs)
	{
		//��Ҫʹ��UNet��������
		if (model_select == 0)
		{
			std::string get_axis = "";

			int out_size = defect_classes * INPUT_H * INPUT_W;
			cv::Mat InputImg;
			InputImg = cv::Mat(height_1, width_1, CV_8UC3, (uchar*)img1);

			cv::Mat MaskImg = cv::Mat(INPUT_H, INPUT_W, CV_8UC1);

			//ͼ��Ԥ����
			Scaling_ratio get_ratio;
			get_ratio = proccess_img_UNet(InputImg, struct_cam[camera_num][thread_num].tdata, INPUT_H, INPUT_W);


			////����	
			doInference_UNet(*struct_cam[camera_num][thread_num].context,
				*struct_cam[camera_num][thread_num].engine,
				struct_cam[camera_num][thread_num].stream,
				struct_cam[camera_num][thread_num].buffer,
				struct_cam[camera_num][thread_num].tdata,
				struct_cam[camera_num][thread_num].prob,
				1, INPUT_H, INPUT_W, out_size, select_gpu,
				struct_cam[camera_num][thread_num].input_index,
				struct_cam[camera_num][thread_num].output_index);


			//���ؼ��ķ���
			//softmax_UNet(INPUT_H, INPUT_W, defect_classes, prob, out);

			if (true_classes > 1)
			{
				defect_classes = true_classes;
			}

			//��ȡÿ��������֮�Ķ���ߵ�����
			argmax_UNet(INPUT_H, INPUT_W, defect_classes, threshold, struct_cam[camera_num][thread_num].prob);

			//����ÿ����ͼȻ��תͼƬ		
			uchar* ptmp_0 = NULL;

			//�������ȱ�����ת��Ϊͼ���ҵ���С��Ӿ��Σ��������õ������ֵ�����������ε��ĸ�����㣬���������ַ�����ʽ���س�ȥ��
			for (int p = 0; p < defect_classes; p++)
			{
				for (int i = 0; i < INPUT_H; i++)
				{
					ptmp_0 = MaskImg.ptr<uchar>(i);
					//ptmp_0 = MaskImg.ptr<uchar>(i);
					for (int j = 0; j < INPUT_W; j++)
					{
						ptmp_0[j] = (struct_cam[camera_num][thread_num].prob)[i * INPUT_W + j + INPUT_W * INPUT_H * p];
					}
				}

				//��ȡÿ��ͼ��ÿ��ȱ�ݵ���С��Ӿ���
				vector<vector<Point> > contours;
				vector<Vec4i> hierarchy;
				cv::findContours(MaskImg, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
				float defect_h;
				float defect_w;

				if (envs == 0)
				{
					for (int v = 0; v < contours.size(); v++)
					{
						get_axis += to_string(p);
						for (int b = 0; b < contours[v].size(); b++)
						{
							get_axis += ",";
							get_axis += to_string(((double)(contours[v][b].x) - get_ratio.left) / get_ratio.ratio);
							get_axis += ",";
							get_axis += to_string(((double)(contours[v][b].y) - get_ratio.top) / get_ratio.ratio);
						}
						get_axis += ",";
						get_axis += to_string(abs(contourArea(contours[v])));
						get_axis += ";";
					}
				}
				else if (envs == 1)
				{
					for (int i = 0; i < contours.size(); i++)
					{
						//��ȡ��С��Ӿ��ε�����
						RotatedRect rect = minAreaRect(contours[i]);
						Point2f Pt[4];
						rect.points(Pt);
						for (int i = 0; i < 4; i++)
						{
							Num[i].x = Pt[i].x;
							Num[i].y = Pt[i].y;
						}
						sort(Num, Num + 4, cmp);


						double raw_0x = ((double)(Pt[0].x) - get_ratio.left) / get_ratio.ratio;
						double raw_0y = ((double)(Pt[0].y) - get_ratio.top) / get_ratio.ratio;

						double raw_1x = ((double)(Pt[1].x) - get_ratio.left) / get_ratio.ratio;
						double raw_1y = ((double)(Pt[1].y) - get_ratio.top) / get_ratio.ratio;

						double raw_2x = ((double)(Pt[2].x) - get_ratio.left) / get_ratio.ratio;
						double raw_2y = ((double)(Pt[2].y) - get_ratio.top) / get_ratio.ratio;

						double raw_3x = ((double)(Pt[3].x) - get_ratio.left) / get_ratio.ratio;
						double raw_3y = ((double)(Pt[3].y) - get_ratio.top) / get_ratio.ratio;


						//��ȡ��Ӿ��εĳ������Լ����
						double tmp_a = pow(pow((raw_3x - raw_0x), 2) + pow((raw_3y - raw_0y), 2), 0.5);
						double tmp_b = pow(pow((raw_3x - raw_2x), 2) + pow((raw_3y - raw_2y), 2), 0.5);
						if (tmp_a > tmp_b)
						{
							defect_h = tmp_b;
							defect_w = tmp_a;
						}
						else
						{
							defect_h = tmp_a;
							defect_w = tmp_b;
						}

						double resized_area = abs(contourArea(contours[i]));
						int resized_true_img = (INPUT_H - (get_ratio.top + get_ratio.bottom)) * (INPUT_W - (get_ratio.left + get_ratio.right));
						double defect_area = width_1 * height_1 * resized_area / resized_true_img;


						//�жϼ�⵽��ĳ��Ϳ��Ƿ�������������ֵ�������ڣ��ͼ�¼������ꡢ�ߡ�����������������ü���
						if (defect_area > area_threshold)
						{
							get_axis += to_string(p);
							get_axis += ",";
							get_axis += to_string(double(raw_0x));
							get_axis += ",";
							get_axis += to_string(double(raw_0y));
							get_axis += ",";
							get_axis += to_string(double(raw_1x));
							get_axis += ",";
							get_axis += to_string(double(raw_1y));
							get_axis += ",";
							get_axis += to_string(double(raw_2x));
							get_axis += ",";
							get_axis += to_string(double(raw_2y));
							get_axis += ",";
							get_axis += to_string(double(raw_3x));
							get_axis += ",";
							get_axis += to_string(double(raw_3y));
							get_axis += ",";
							get_axis += to_string(defect_h);
							get_axis += ",";
							get_axis += to_string(defect_w);
							get_axis += ",";
							get_axis += to_string(defect_area);
							get_axis += ";";
						}
					}
				}
			}

			return strdup(get_axis.c_str());
		}
		else if (model_select == 1)
		{

			uchar2Mat[camera_num][thread_num] = cv::Mat(height_1, width_1, CV_8UC3, img1);
			//Ԥ����					
			proccess_img_ResNet(uchar2Mat[camera_num][thread_num], struct_cam[camera_num][thread_num].tdata, INPUT_H, INPUT_W);

			doInference_ResNet(*struct_cam[camera_num][thread_num].context,
				*struct_cam[camera_num][thread_num].engine,
				struct_cam[camera_num][thread_num].stream,
				struct_cam[camera_num][thread_num].buffer,
				struct_cam[camera_num][thread_num].tdata,
				struct_cam[camera_num][thread_num].prob,
				INPUT_H, INPUT_W, select_gpu,
				struct_cam[camera_num][thread_num].input_index,
				struct_cam[camera_num][thread_num].output_index);

			//softmax
			float tout[2] = { 0,0 };
			tout[0] = exp(struct_cam[camera_num][thread_num].prob[0]) / (exp(struct_cam[camera_num][thread_num].prob[0]) + exp(struct_cam[camera_num][thread_num].prob[1]));
			tout[1] = exp(struct_cam[camera_num][thread_num].prob[1]) / (exp(struct_cam[camera_num][thread_num].prob[0]) + exp(struct_cam[camera_num][thread_num].prob[1]));

			if (typeid(tout[0]) == typeid(float))
			{
				string get_result = "0,0,0,0,0,0,0,0,0,0,0,0;";

				if (tout[1] > tout[0])
				{
					if (tout[1] <= 1 && tout[1] >= 0)
					{
						get_result = "0," + to_string(tout[1]) + ",0,0,0,0,0,0,0,0,0,0;";
					}
					else
					{
						get_result = "0,0.501,0,0,0,0,0,0,0,0,0,0;";
					}
				}
				else {

					if (tout[0] >= 0 && tout[0] <= 1)
					{
						get_result = "1," + to_string(tout[0]) + ",0,0,0,0,0,0,0,0,0,0;";
					}
					else
					{
						get_result = "1,0.501,0,0,0,0,0,0,0,0,0,0;";
					}
				}
				return strdup(get_result.c_str());
			}
			else
			{
				string get_result = "0,0.501,0,0,0,0,0,0,0,0,0,0;";
				return strdup(get_result.c_str());
			}
		}
	}

	/*******************************************************
		-�������ƣ�ReleaseEng

		-���������������ͷ�������̣߳�

		-���������camera_cls: �ͷŵĵڼ��������
					num_thread: �ͷŴ�����ϵڼ����̣߳�

		-�����������

	*******************************************************/
	_declspec (dllexport)void ReleaseEng(int camera_cls, int num_thread)
	{
		if (Flags == true)
		{
			struct_cam[camera_cls][num_thread].context->destroy();
			struct_cam[camera_cls][num_thread].engine->destroy();
		}
	}


	/*******************************************************
	-�������ƣ�getVersion

	-�������������ڵ�ǰdll�İ汾�ţ�

	-�����������

	-���������char*���͵İ汾�ţ���:1.0.2

*******************************************************/
	_declspec (dllexport)char* getVersion()
	{
		return (char*)DLL_VERSION.c_str();
	}
};

