#include <jni.h>
#include <string>
#include<android/log.h>
#include <CL/cl.hpp>

#include "CL/cl.h"

int opencl_VecAdd();

#define LOG    "pllab_opencl-jni"
#define LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG,__VA_ARGS__)

static const char *PROGRAM_SOURCE[] = {
        "__kernel void buffer_addition(__global const float *matrix_a,\n",
        "                              __global const float *matrix_b,\n",
        "                              __global       float *matrix_c)\n",
        "{\n",
        "    const int wid_x = get_global_id(0);\n",
        "    matrix_c[wid_x] = matrix_a[wid_x] + matrix_b[wid_x];\n",
        "}\n",
};

static const cl_uint PROGRAM_SOURCE_LEN = sizeof(PROGRAM_SOURCE) / sizeof(const char *);

//matrix_t make_matrix(int width, int height) {
//    matrix_t res;
//    res.width = width;
//    res.height = height;
//    for (int i = 0; i < res.width * res.height; i++) {
//        res.elements.push_back(2);
//    }
//}

//void hello() {
//    const matrix_t matrix_a = make_matrix(3, 3);
//    const matrix_t matrix_b = make_matrix(3, 3);
//    const size_t matrix_size = matrix_a.width * matrix_a.height;
//    const size_t matrix_bytes = matrix_size * sizeof(cl_float);
//
//    cl_wrapper wrapper;
//    cl_program program = wrapper.make_program(PROGRAM_SOURCE, PROGRAM_SOURCE_LEN);
//    cl_kernel kernel = wrapper.make_kernel("buffer_addition", program);
//    cl_context context = wrapper.get_context();
//    cl_command_queue command_queue = wrapper.get_command_queue();
//
//    cl_int err = CL_SUCCESS;
//
//    cl_mem_ion_host_ptr matrix_a_ion_buf = wrapper.make_ion_buffer(matrix_bytes);
//    std::memcpy(matrix_a_ion_buf.ion_hostptr, matrix_a.elements.data(), matrix_bytes);
//    cl_mem matrix_a_mem = clCreateBuffer(
//            context,
//            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
//            matrix_bytes,
//            &matrix_a_ion_buf,
//            &err
//    );
//
//    LOGI("err# 1: %d\n", err);
//
//    cl_mem_ion_host_ptr matrix_b_ion_buf = wrapper.make_ion_buffer(matrix_bytes);
//    std::memcpy(matrix_b_ion_buf.ion_hostptr, matrix_b.elements.data(), matrix_bytes);
//    cl_mem matrix_b_mem = clCreateBuffer(
//            context,
//            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
//            matrix_bytes,
//            &matrix_b_ion_buf,
//            &err
//    );
//
//    LOGI("err# 2: %d\n", err);
//    cl_mem_ion_host_ptr matrix_c_ion_buf = wrapper.make_ion_buffer(matrix_bytes);
//    cl_mem matrix_c_mem = clCreateBuffer(
//            context,
//            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
//            matrix_bytes,
//            &matrix_c_ion_buf,
//            &err
//    );
//    LOGI("err# 3: %d\n", err);
//
//}

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