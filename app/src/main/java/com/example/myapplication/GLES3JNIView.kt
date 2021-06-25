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
package com.example.myapplication

import android.content.Context
import android.opengl.GLSurfaceView
import android.util.Log
import com.example.myapplication.GLES3JNILib.init
import com.example.myapplication.GLES3JNILib.resize
import com.example.myapplication.GLES3JNILib.step
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class GLES3JNIView(context: Context?, number: Int, use_vuda: Boolean) : GLSurfaceView(context) {
    private class Renderer(number: Int, use_vuda: Boolean) : GLSurfaceView.Renderer {
        override fun onDrawFrame(gl: GL10) {
            step(use_vuda)
            counter.logFrame()
        }

        override fun onSurfaceChanged(gl: GL10, width: Int, height: Int) {
            resize(width, height)
        }

        override fun onSurfaceCreated(gl: GL10, config: EGLConfig) {
            init(number)
        }

        private var use_vuda = use_vuda
        private var number = number
        private var counter = FPSCounter()
    }

    companion object {
        private const val TAG = "GLES3JNI"
        private const val DEBUG = true
    }

    init {
        // Pick an EGLConfig with RGB8 color, 16-bit depth, no stencil,
        // supporting OpenGL ES 2.0 or later backwards-compatible versions.
        setEGLConfigChooser(8, 8, 8, 0, 16, 0)
        setEGLContextClientVersion(3)
        setRenderer(Renderer(number, use_vuda))
    }
}


class FPSCounter {
    var startTime = System.nanoTime()
    var frames = 0
    fun logFrame() {
        frames++
        if (System.nanoTime() - startTime >= 1000000000) {
            Log.d("FPSCounter", "fps: $frames")
            frames = 0
            startTime = System.nanoTime()
        }
    }
}