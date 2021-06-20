# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.10.2)

# Declares and names the project.

project("myapplication")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        native-lib

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        native-lib.cpp)

add_library( # Sets the name of the library.
        gles3jni

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        gles3jni.cpp
        RendererES3.cpp
        )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.
find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log android)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libra ries, or system libraries.

target_link_libraries( # Specifies the target library.
        native-lib
        vulkan_lib
        opencl_lib
        cutils_lib
        cpp_lib
        base_lib
        GLESv3

        # Links the target library to the log library
        # included in the NDK.
        ${log-lib} android)


target_link_libraries( # Specifies the target library.
        gles3jni
        opencl_lib
        cutils_lib
        vulkan_lib
        cpp_lib
        base_lib
        GLESv3
        EGL

        # Links the target library to the log library
        # included in the NDK.
        ${log-lib} android)

include_directories(${CMAKE_SOURCE_DIR}/include/)
add_library(opencl_lib STATIC IMPORTED)
set_target_properties(opencl_lib PROPERTIES
        IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/arm64-v8a/libOpenCL.so)

add_library(cutils_lib STATIC IMPORTED)
set_target_properties(cutils_lib PROPERTIES
        IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/arm64-v8a/libcutils.so)

add_library(cpp_lib STATIC IMPORTED)
set_target_properties(cpp_lib PROPERTIES
        IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/arm64-v8a/libc++.so)

add_library(base_lib STATIC IMPORTED)
set_target_properties(base_lib PROPERTIES
        IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/arm64-v8a/libbase.so)

add_library(vulkan_lib STATIC IMPORTED)
set_target_properties(vulkan_lib PROPERTIES
        IMPORTED_LOCATION
        ${CMAKE_SOURCE_DIR}/jniLibs/arm64-v8a/libvulkan.so)

