#include <jni.h>
#include <string>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <vuda_runtime.hpp>

#define LOG_TAG "ndk-build"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_foo(JNIEnv *env, jobject /* this */) {
    // assign a device to the thread
    cudaSetDevice(0);
    // allocate memory on the device
    const int N = 5000;
    int a[N], b[N], c[N];
    for (int i = 0; i < N; ++i) {
        a[i] = -i;
        b[i] = i * i;
    }
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void **) &dev_a, N * sizeof(int));
    cudaMalloc((void **) &dev_b, N * sizeof(int));
    cudaMalloc((void **) &dev_c, N * sizeof(int));
    // copy the arrays a and b to the device
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    // run kernel (vulkan shader module)
    const int blocks = 128;
    const int threads = 128;
    const int stream_id = 0;
    vuda::launchKernel("/storage/emulated/0/Android/data/com.example.myapplication/files/add.spv", "main", stream_id, blocks, threads, dev_a, dev_b, dev_c, N);
    // copy result to host
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // do something useful with the result in array c ...
    for (int i=0; i<N; i++) {
        LOGI("%d ", c[i]);
    }
    // free memory on device
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}