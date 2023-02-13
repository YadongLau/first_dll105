#pragma once
#include <iostream>
#include "cuda_runtime_api.h"

using namespace nvinfer1;
using namespace std;

struct return_data;

return_data convert_bin(string root_path, int model_num, int cudaSetPrecision);

int startsWith(string s, string sub);

int endsWith(string s, string sub);

vector<string> split(const string& str, const string& delim);

string UTF8ToGB(const char* str);

