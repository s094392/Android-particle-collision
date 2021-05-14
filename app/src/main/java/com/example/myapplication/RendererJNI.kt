package com.example.myapplication

import android.content.Context
import android.content.res.AssetManager
import android.opengl.GLSurfaceView
import android.util.Log
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10


class RendererJNI(context: Context) : GLSurfaceView.Renderer {
    companion object {
        init {
            System.loadLibrary("native-lib")
        }
    }
    private var mAssetMgr: AssetManager? = null
    private val mLogTag = "ndk-build"
    external fun foo()
    external fun glesInit()
    external fun glesRender()
    external fun glesResize(width: Int, height: Int)
    external fun readShaderFile(assetMgr: AssetManager?)

    override fun onSurfaceCreated(gl: GL10, config: EGLConfig) {
        readShaderFile(mAssetMgr)
        glesInit()
    }

    override fun onSurfaceChanged(gl: GL10, width: Int, height: Int) {
        glesResize(width, height)
    }

    override fun onDrawFrame(gl: GL10) {
        glesRender()
    }

    init {
        mAssetMgr = context.assets
        if (null == mAssetMgr) {
            Log.e(mLogTag, "getAssets() return null !")
        }
    }
}