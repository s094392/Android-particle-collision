package com.example.myapplication

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import androidx.appcompat.widget.SwitchCompat


class MainActivity : Activity() {
    override fun onCreate(icicle: Bundle?) {
        super.onCreate(icicle)
        setContentView(R.layout.activity_main)
        val button = findViewById<Button>(R.id.button)
        val number = findViewById<EditText>(R.id.editTextNumber)
        number.setText(3000.toString());
        val vuda = findViewById<SwitchCompat>(R.id.switch1)
        vuda.isChecked = true
        button.setOnClickListener {
            val i = Intent(this, GLES3JNILibActivity::class.java)
            i.putExtra("number", Integer.parseInt(number.text.toString()))
            i.putExtra("vuda", vuda.isChecked)
            startActivity(i)
        }
    }
}