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

#include <jni.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "gles3jni.h"
#include <GLES3/gl3.h>

Vertex QUAD[12] = {
    // Square with diagonal < 2 so that it fits in a [-1 .. 1]^2 square
    // regardless of rotation.
    {{-0.7f, -0.7f}, {0x00, 0xFF, 0x00}},
    {{-0.7f,  0.7f}, {0xFF, 0x00, 0x00}},
    {{ 0.0f,  1.3f}, {0xFF, 0xFF, 0xFF}},
    {{ 0.7f,  0.7f}, {0xFF, 0xFF, 0xFF}},
    {{ 0.7f, -0.7f}, {0x00, 0x00, 0xFF}},
};

bool checkGlError(const char* funcName) {
    GLint err = glGetError();
    if (err != GL_NO_ERROR) {
        ALOGE("GL error after %s(): 0x%08x\n", funcName, err);
        return true;
    }
    return false;
}

GLuint createShader(GLenum shaderType, const char* src) {
    GLuint shader = glCreateShader(shaderType);
    if (!shader) {
        checkGlError("glCreateShader");
        return 0;
    }
    glShaderSource(shader, 1, &src, NULL);

    GLint compiled = GL_FALSE;
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLint infoLogLen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLen);
        if (infoLogLen > 0) {
            GLchar* infoLog = (GLchar*)malloc(infoLogLen);
            if (infoLog) {
                glGetShaderInfoLog(shader, infoLogLen, NULL, infoLog);
                ALOGE("Could not compile %s shader:\n%s\n",
                        shaderType == GL_VERTEX_SHADER ? "vertex" : "fragment",
                        infoLog);
                free(infoLog);
            }
        }
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

GLuint createProgram(const char* vtxSrc, const char* fragSrc) {
    GLuint vtxShader = 0;
    GLuint fragShader = 0;
    GLuint program = 0;
    GLint linked = GL_FALSE;

    vtxShader = createShader(GL_VERTEX_SHADER, vtxSrc);
    if (!vtxShader)
        goto exit;

    fragShader = createShader(GL_FRAGMENT_SHADER, fragSrc);
    if (!fragShader)
        goto exit;

    program = glCreateProgram();
    if (!program) {
        checkGlError("glCreateProgram");
        goto exit;
    }
    glAttachShader(program, vtxShader);
    glAttachShader(program, fragShader);

    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        ALOGE("Could not link program");
        GLint infoLogLen = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLen);
        if (infoLogLen) {
            GLchar* infoLog = (GLchar*)malloc(infoLogLen);
            if (infoLog) {
                glGetProgramInfoLog(program, infoLogLen, NULL, infoLog);
                ALOGE("Could not link program:\n%s\n", infoLog);
                free(infoLog);
            }
        }
        glDeleteProgram(program);
        program = 0;
    }

exit:
    glDeleteShader(vtxShader);
    glDeleteShader(fragShader);
    return program;
}

static void printGlString(const char* name, GLenum s) {
    const char* v = (const char*)glGetString(s);
    ALOGV("GL %s: %s\n", name, v);
}

// ----------------------------------------------------------------------------

Renderer::Renderer()
: mLastFrameNs(0)
{
//    memset(mScale, 0, sizeof(mScale));
//    memset(mAngularVelocity, 0, sizeof(mAngularVelocity));
//    memset(mAngles, 0, sizeof(mAngles));



}

Renderer::~Renderer() {
}

void Renderer::resize(int w, int h) {
//    auto offsets = mapOffsetBuf();
//    calcSceneParams(w, h, offsets);
//    unmapOffsetBuf();

    yScale = (float)w / h;
    yBound = (float)h / w;

    auto quad = mapQUADBuf();
    for (int i = 0; i < circleEdge; i++) {
        quad[i].pos[0] = radius * cos(i * twicePi / circleEdge);
        quad[i].pos[1] = radius * sin(i * twicePi / circleEdge) * yScale;
    }
    unmapQUADBuf();

    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        x[i] = (rand() % (2000 - 1) - 1000) / 1000.0;
        y[i] = ((rand() % (2000 - 1) - 1000) / 1000.0) * yBound;
        dx[i] = (rand() % 2000 - 1000) / 200000.0;;
        dy[i] = (rand() % 2000 - 1000) / 200000.0;;
        new_dx[i] = dx[i];
        new_dy[i] = dy[i];
    }

    auto offsets = mapOffsetBuf();
    for (int i = 0; i < n; i++) {
        offsets[2 * i] = x[i];
        offsets[2 * i + 1] = y[i] * yScale;
    }
    unmapOffsetBuf();

//    // Auto gives a signed int :-(
//    for (auto i = (unsigned)0; i < mNumInstances; i++) {
//        mAngles[i] = drand48() * TWO_PI;
//        mAngularVelocity[i] = MAX_ROT_SPEED * (2.0*drand48() - 1.0);
//    }

    mLastFrameNs = 0;

    glViewport(0, 0, w, h);


}

//void Renderer::calcSceneParams(unsigned int w, unsigned int h,
//        float* offsets) {
//    // number of cells along the larger screen dimension
//    const float NCELLS_MAJOR = MAX_INSTANCES_PER_SIDE;
//    // cell size in scene space
//    const float CELL_SIZE = 2.0f / NCELLS_MAJOR;
//
//    // Calculations are done in "landscape", i.e. assuming dim[0] >= dim[1].
//    // Only at the end are values put in the opposite order if h > w.
//    const float dim[2] = {fmaxf(w,h), fminf(w,h)};
//    const float aspect[2] = {dim[0] / dim[1], dim[1] / dim[0]};
//    const float scene2clip[2] = {1.0f, aspect[0]};
//    const int ncells[2] = {
//            static_cast<int>(NCELLS_MAJOR),
//            (int)floorf(NCELLS_MAJOR * aspect[1])
//    };
//
//    float centers[2][MAX_INSTANCES_PER_SIDE];
//    for (int d = 0; d < 2; d++) {
//        auto offset = -ncells[d] / NCELLS_MAJOR; // -1.0 for d=0
//        for (auto i = 0; i < ncells[d]; i++) {
//            centers[d][i] = scene2clip[d] * (CELL_SIZE*(i + 0.5f) + offset);
//        }
//    }
//
//    int major = w >= h ? 0 : 1;
//    int minor = w >= h ? 1 : 0;
//    // outer product of centers[0] and centers[1]
//    for (int i = 0; i < ncells[0]; i++) {
//        for (int j = 0; j < ncells[1]; j++) {
//            int idx = i*ncells[1] + j;
//            offsets[2*idx + major] = centers[0][i];
//            offsets[2*idx + minor] = centers[1][j];
//        }
//    }
//
//    mNumInstances = ncells[0] * ncells[1];
//    mScale[major] = 0.5f * CELL_SIZE * scene2clip[0];
//    mScale[minor] = 0.5f * CELL_SIZE * scene2clip[1];
//}

float Renderer::distance2(int i, int j) {
    return (x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j]);
}

void Renderer::step() {
    timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    auto nowNs = now.tv_sec*1000000000ull + now.tv_nsec;

    if (mLastFrameNs > 0) {
        float dt = float(nowNs - mLastFrameNs) * 0.0000005f;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (i == j) continue;
                if (distance2(i, j) <= 4 * radius * radius) {
                    if ((x[i] - x[j]) * (dx[i] - dx[j]) + (y[i] - y[j]) * (dy[i] - dy[j]) <= 0) {
                        float dot = (dx[i] - dx[j]) * (x[i] - x[j]) + (dy[i] - dy[j]) * (y[i] - y[j]);
                        new_dx[i] = dx[i] - dot / distance2(i, j) * (x[i] - x[j]);
                        new_dy[i] = dy[i] - dot / distance2(i, j) * (y[i] - y[j]);
                        new_dx[j] = dx[j] - dot / distance2(i, j) * (x[j] - x[i]);
                        new_dy[j] = dy[j] - dot / distance2(i, j) * (y[j] - y[i]);
                    }
                }
            }
        }
        for (int i = 0; i < n; i++) {
            if (x[i] + radius >= 1) new_dx[i] = -abs(new_dx[i]);
            if (x[i] - radius <= -1) new_dx[i] = abs(new_dx[i]);
            if (y[i] + radius >= yBound) new_dy[i] = -abs(new_dy[i]);
            if (y[i] - radius <= -yBound) new_dy[i] = abs(new_dy[i]);

            dx[i] = new_dx[i];
            dy[i] = new_dy[i];
            x[i] += dx[i] * dt;
            y[i] += dy[i] * dt;
        }

        auto offsets = mapOffsetBuf();
        for (int i = 0; i < n; i++) {
            offsets[2 * i] = x[i];
            offsets[2 * i + 1] = y[i] * yScale;
        }
        unmapOffsetBuf();

//        for (unsigned int i = 0; i < mNumInstances; i++) {
//            mAngles[i] += mAngularVelocity[i] * dt;
//            if (mAngles[i] >= TWO_PI) {
//                mAngles[i] -= TWO_PI;
//            } else if (mAngles[i] <= -TWO_PI) {
//                mAngles[i] += TWO_PI;
//            }
//        }

//        float* transforms = mapTransformBuf();
//        for (unsigned int i = 0; i < mNumInstances; i++) {
//            float s = sinf(mAngles[i]);
//            float c = cosf(mAngles[i]);
//            transforms[4*i + 0] =  c * mScale[0];
//            transforms[4*i + 1] =  s * mScale[1];
//            transforms[4*i + 2] = -s * mScale[0];
//            transforms[4*i + 3] =  c * mScale[1];
//        }
//        unmapTransformBuf();
    }

    mLastFrameNs = nowNs;
}

void Renderer::render() {
    step();

    glClearColor(0.2f, 0.2f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    draw(n);
    checkGlError("Renderer::render");
}

// ----------------------------------------------------------------------------

static Renderer* g_renderer = NULL;

extern "C" {
    JNIEXPORT void JNICALL Java_com_android_gles3jni_GLES3JNILib_init(JNIEnv* env, jobject obj);
    JNIEXPORT void JNICALL Java_com_android_gles3jni_GLES3JNILib_resize(JNIEnv* env, jobject obj, jint width, jint height);
    JNIEXPORT void JNICALL Java_com_android_gles3jni_GLES3JNILib_step(JNIEnv* env, jobject obj);
};

#if !defined(DYNAMIC_ES3)
static GLboolean gl3stubInit() {
    return GL_TRUE;
}
#endif

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_GLES3JNILib_init(JNIEnv* env, jclass obj) {
    if (g_renderer) {
        delete g_renderer;
        g_renderer = NULL;
    }

    printGlString("Version", GL_VERSION);
    printGlString("Vendor", GL_VENDOR);
    printGlString("Renderer", GL_RENDERER);
    printGlString("Extensions", GL_EXTENSIONS);

    const char* versionStr = (const char*)glGetString(GL_VERSION);
    if (strstr(versionStr, "OpenGL ES 3.") && gl3stubInit()) {
        g_renderer = createES3Renderer();
    } else {
        ALOGE("Unsupported OpenGL ES version");
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_GLES3JNILib_resize(JNIEnv* env, jclass obj, jint width, jint height) {
    if (g_renderer) {
        g_renderer->resize(width, height);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_GLES3JNILib_step(JNIEnv* env, jclass obj) {
    if (g_renderer) {
        g_renderer->render();
    }
}