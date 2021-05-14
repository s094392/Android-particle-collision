package com.example.myapplication


import android.app.ActivityManager
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity


class MainActivity : AppCompatActivity() {
    companion object {
        init {
            System.loadLibrary("native-lib")
        }
    }
    private val CONTEXT_CLIENT_VERSION = 3
    private var mGLSurfaceView: GLSurfaceView? = null
    private external  fun stringFromJNI(): String

    var mRenderer: RendererJNI? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        stringFromJNI()
        mGLSurfaceView = GLSurfaceView(this)
        mRenderer = RendererJNI(this)
        if (detectOpenGLES30()) {
            // 设置OpenGl ES的版本
            mGLSurfaceView!!.setEGLContextClientVersion(CONTEXT_CLIENT_VERSION)
            // 设置与当前GLSurfaceView绑定的Renderer
            mGLSurfaceView!!.setRenderer(mRenderer)
            // 设置渲染的模式
            mGLSurfaceView!!.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
        } else {
            Log.e("opengles30", "OpenGL ES 3.0 not supported on device.  Exiting...")
            finish()
        }
        setContentView(mGLSurfaceView)
    }

    override fun onResume() {
        super.onResume()
        mGLSurfaceView!!.onResume()
    }

    override fun onPause() {
        super.onPause()
        mGLSurfaceView!!.onPause()
    }

    private fun detectOpenGLES30(): Boolean {
        val am = getSystemService(ACTIVITY_SERVICE) as ActivityManager
        val info = am.deviceConfigurationInfo
        return info.reqGlEsVersion >= 0x30000
    }
}