#include <Windows.h>
#include"train.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <io.h>
#include <iostream>
#include <vector>
#include <string>
#include <vector>
#include "dirent.h"
#include <locale.h>
#include <thread>
#include <stdint.h>
#include <fstream>


using namespace std;
using namespace cv;

char* kkk1;


void thread01()
{
	struct dirent* ptr;
	DIR* dir;
	string PATH = "C:\\Users\\Yadong\\Desktop\\123\\3\\bmp";

	dir = opendir(PATH.c_str());
	vector<string> files;

	while ((ptr = readdir(dir)) != NULL)
	{
		//跳过'.'和'..'两个目录
		if (ptr->d_name[0] == '.')
			continue;
		//cout << ptr->d_name << endl;
		files.push_back(ptr->d_name);
	}

	while (true)
	{
		for (int i = 0; i < files.size(); ++i)
		{
			cv::Mat img1 = cv::imread(PATH + "\\" + files[i]);
			unsigned char* uchar_img1 = img1.data;

			auto start111 = chrono::steady_clock::now();
			//string kk2 = Result((BYTE*)uchar_img1, 1, 1, 1, 5, 0.5, 10, 2426, 1674, 544, 544, 0, 1);
			string kk2 = Result((BYTE*)uchar_img1, 1, 1, 1, 9, 0.5, 0, 1440, 1080, 384, 384, 0, 1);
			auto end222 = chrono::steady_clock::now();
			double total_time = chrono::duration<double, milli>(end222 - start111).count();

			std::cout << "thread_1: " << total_time << endl;

			/*if (sizeof(kk2) > 10)
			{
				cout << "thread_1： " << kk2 << endl;
			}*/
		}
	}
	closedir(dir);
	ReleaseEng(1, 1);
}


void thread02()
{
	struct dirent* ptr;
	DIR* dir;
	string PATH = "C:\\Users\\Administrator\\Desktop\\123\\2\\bmp";

	dir = opendir(PATH.c_str());
	vector<string> files;

	while ((ptr = readdir(dir)) != NULL)
	{
		//跳过'.'和'..'两个目录
		if (ptr->d_name[0] == '.')
			continue;
		//cout << ptr->d_name << endl;
		files.push_back(ptr->d_name);
	}

	int time = 0;
	while (true)
	{
		for (int i = 0; i < files.size(); ++i)
		{
			cv::Mat img1 = cv::imread(PATH + "\\" + files[i]);
			unsigned char* uchar_img1 = img1.data;

			Sleep(20);

			auto start111 = chrono::steady_clock::now();

			string kk2 = Result((BYTE*)uchar_img1, 1, 2, 1, 9, 0.5, 0, 1440, 1080, 416, 416, 0, 1);

			auto end222 = chrono::steady_clock::now();
			double total_time = chrono::duration<double, milli>(end222 - start111).count();
			std::cout << "thread_2: " << total_time << endl;

			time += 1;
			if (time >= 10000)
			{
				break;
			}

			/*if (sizeof(kk2) > 10)
			{
				cout << "thread_1： " << kk2 << endl;
			}*/
		}
		break;
	}
	closedir(dir);
	ReleaseEng(1, 2);
}


void thread03()
{
	struct dirent* ptr;
	DIR* dir;

	string PATH = "D:\\test\\dafenlei\\111\\Project\\NGImage";

	dir = opendir(PATH.c_str());
	vector<string> files;

	while ((ptr = readdir(dir)) != NULL)
	{
		//跳过'.'和'..'两个目录
		if (ptr->d_name[0] == '.')
			continue;
		//cout << ptr->d_name << endl;
		files.push_back(ptr->d_name);
	}
	while (1)
	{
		for (int i = 0; i < files.size(); ++i)
		{
			cv::Mat img1 = cv::imread(PATH + "\\" + files[i]);
			unsigned char* uchar_img1 = img1.data;
			//推理1

			auto start2 = chrono::steady_clock::now();

			string kk2 = Result((BYTE*)uchar_img1, 0, 0, 1, 2, 0.5, 10, 1300, 800, 224, 224, 1, 1);

			auto end2 = chrono::steady_clock::now();
			double duration_millsecond = std::chrono::duration<double, std::milli>(end2 - start2).count();
			//std::cout << duration_millsecond << "毫秒" << std::endl;

			if (sizeof(kk2) > 10)
			{
				cout << "thread_22222 " << kk2 << endl;
			}
		}
	}
	closedir(dir);
}



int main()
{
	setlocale(LC_ALL, "LC_CTYPE=.utf8");

	/*while (true)
	{
		kkk1 = getVersion();
		cout << kkk1 << endl;
	}*/


	// UNet
	//string bin_path1 = "D:\\test\\korea";
	string bin_path1 = "C:\\Users\\Yadong\\Desktop\\123\\3";
	//string bin_path2 = "C:\\Users\\Administrator\\Desktop\\123\\2";


	Loadfile(bin_path1.c_str(), 1, 1, 1, 0);
	//Loadfile(bin_path2.c_str(), 1, 2, 1, 0);

	thread task01(thread01);
	//thread task02(thread02);

	task01.join();
	//task02.join();

	//ResNet
	/*string bin_path3 = "D:\\test\\dafenlei\\111\\Project";
	Loadfile(bin_path3.c_str(), 0, 0, 1, 1);
	thread task03(thread03);
	task03.join();*/

	return 0;
}
