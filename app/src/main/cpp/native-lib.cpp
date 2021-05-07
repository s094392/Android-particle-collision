#include <jni.h>
#include <string>
#include<android/log.h>
#include <CL/cl.hpp>
#include "CL/cl.h"


#define LOG    "pllab_opencl-jni"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG,__VA_ARGS__)

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {

    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;
    int gpu = 0;
    cl_platform_id platform;
    cl_int err;
    err = clGetPlatformIDs(1, &platform, NULL);

//    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);

    char buff[100];
    snprintf(buff, sizeof(buff), "Platform: %d", err);
    std::string buffAsStdStr = buff;

    return env->NewStringUTF(buffAsStdStr.c_str());
}