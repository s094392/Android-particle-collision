/*
 * Copyright 2013 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GLES3JNI_H
#define GLES3JNI_H 1

#include <android/log.h>
#include <math.h>
#include <GLES3/gl31.h>

#define DEBUG 1

#define LOG_TAG "ndk-build"
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#if DEBUG
#define ALOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)
#else
#define ALOGV(...)
#endif

// ----------------------------------------------------------------------------
// Types, functions, and data used by both ES2 and ES3 renderers.
// Defined in gles3jni.cpp.

#define MAX_INSTANCES_PER_SIDE 16
// #define MAX_INSTANCES   (MAX_INSTANCES_PER_SIDE * MAX_INSTANCES_PER_SIDE)
#define MAX_INSTANCES 10000
#define TWO_PI          (2.0 * M_PI)
#define MAX_ROT_SPEED   (0.3 * TWO_PI)

// This demo uses three coordinate spaces:
// - The model (a quad) is in a [-1 .. 1]^2 space
// - Scene space is either
//    landscape: [-1 .. 1] x [-1/(2*w/h) .. 1/(2*w/h)]
//    portrait:  [-1/(2*h/w) .. 1/(2*h/w)] x [-1 .. 1]
// - Clip space in OpenGL is [-1 .. 1]^2
//
// Conceptually, the quads are rotated in model space, then scaled (uniformly)
// and translated to place them in scene space. Scene space is then
// non-uniformly scaled to clip space. In practice the transforms are combined
// so vertices go directly from model to clip space.

struct Vertex {
    GLfloat pos[2];
    GLubyte rgba[4];
};
extern Vertex QUAD[12];

// returns true if a GL error occurred
extern bool checkGlError(const char* funcName);
extern GLuint createShader(GLenum shaderType, const char* src);
extern GLuint createProgram(const char* vtxSrc, const char* fragSrc);

// ----------------------------------------------------------------------------
// Interface to the ES2 and ES3 renderers, used by JNI code.

class Renderer {
public:
    virtual ~Renderer();
    void resize(int w, int h);
    void render(bool use_vuda);
    int n = 1000;

protected:
    Renderer();

    // return a pointer to a buffer of MAX_INSTANCES * sizeof(vec2).
    // the buffer is filled with per-instance offsets, then unmapped.
    virtual float* mapOffsetBuf() = 0;
    virtual void unmapOffsetBuf() = 0;
    // return a pointer to a buffer of MAX_INSTANCES * sizeof(vec4).
    // the buffer is filled with per-instance scale and rotation transforms.
//    virtual float* mapTransformBuf() = 0;
//    virtual void unmapTransformBuf() = 0;

    virtual Vertex* mapQUADBuf() = 0;
    virtual void unmapQUADBuf() = 0;

    virtual void draw(unsigned int numInstances) = 0;


    int circleEdge = 12;
    float radius = 0.01;
    float twicePi = 2.0 * 3.142;
    float yScale = 1.0;
    float yBound = 1.0;

    float x[10000];
    float y[10000];
    float dx[10000];
    float dy[10000];
    float new_dx[10000];
    float new_dy[10000];

private:
    void step(bool use_vuda);
    float distance2(int, int);
    uint64_t mLastFrameNs;
};

extern Renderer* createES3Renderer();

#endif // GLES3JNI_H
