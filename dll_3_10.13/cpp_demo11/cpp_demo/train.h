#include <Windows.h> 
#pragma comment(lib,"engine2dll.lib")

extern"C" _declspec(dllimport) void Loadfile(const char* file_path, int cam_cls, int cam_thread, int select_gpu, int model_select);

extern"C" _declspec(dllimport) char* Result(BYTE * img1, int camera_num, int thread_num, int select_gpu, int defect_classes, float threshold, int area_threshold, int width_1, int height_1, int INPUT_H, int INPUT_W, int model_select, int envs);

extern"C" _declspec(dllimport) void ReleaseEng(int camera_cls, int num_thread);

extern"C" _declspec(dllimport) char* getVersion(void);