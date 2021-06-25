package com.example.myapplication

import android.app.Activity
import android.os.Bundle

class GLES3JNILibActivity : Activity() {
    var mView: GLES3JNIView? = null
    override fun onCreate(icicle: Bundle?) {
        super.onCreate(icicle)
        val b = intent.extras
        val number = b!!.getInt("number")
        val vuda = b.getBoolean("vuda")
        mView = GLES3JNIView(application, number, vuda)
        setContentView(mView)
    }

    override fun onPause() {
        super.onPause()
        mView!!.onPause()
    }

    override fun onResume() {
        super.onResume()
        mView!!.onResume()
    }
}