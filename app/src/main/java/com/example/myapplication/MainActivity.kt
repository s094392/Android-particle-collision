package com.example.myapplication


import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity


class MainActivity : AppCompatActivity() {
    companion object {
        init {
            System.loadLibrary("native-lib")
        }
    }

    private external fun foo(): String

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        foo();
    }

}