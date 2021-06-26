# Android OpenGL and Vulkan particle collision example
A OpenGL particle collision  example with parallel computation using [vuda](https://github.com/jgbit/vuda), which reduces the time complexity to O(n).
This my course final project with [Tracy Liu](https://github.com/tracyliu1220).

<img src="https://github.com/s094392/Android-particle-collision/blob/main/media/Overview.png" width="80%">

## Particle collision
By [Tracy Liu](https://github.com/tracyliu1220), Reference: [https://github.com/tracyliu1220/2021-GPU-Final](https://github.com/tracyliu1220/2021-GPU-Final)

## Screenshot
It may only work on Google Pixel 4.

<img src="https://github.com/s094392/Android-particle-collision/blob/main/media/image1.png" width="20%"> <img src="https://github.com/s094392/Android-particle-collision/blob/main/media/image2.png" width="20%">

## Expermiment result
* Device: Pixel 4
* GPU: Adreno 640
* Screen Max FPS: 90

| # of Particles | w/o VUDA | Accumulate w/ VUDA |
| -------- | -------- | -------- |
| 1000     | 90 FPS     | 90 FPS     |
| 3000     | 23 FPS     | 90 FPS     |
| 6000     | 6 FPS     | 64 FPS     |
| 9000     | 3 FPS     | 45 FPS     |

## Vulkan Kernel
### Code
```c
#version 450 core

layout(local_size_x_id = 0) in;
layout(local_size_y_id = 100) in;
layout(constant_id = 1) const uint N = 1;

layout(set = 0, binding = 0) readonly buffer A { float x[]; };
layout(set = 0, binding = 1) readonly buffer B { float y[]; };
layout(set = 0, binding = 2) readonly buffer C { float dx[]; };
layout(set = 0, binding = 3) readonly buffer D { float dy[]; };
layout(set = 0, binding = 4) writeonly buffer E { float new_dx[]; };
layout(set = 0, binding = 5) writeonly buffer F { float new_dy[]; };

float distance2(uint i, uint j) {
    return (x[i] - x[j]) * (x[i] - x[j]) + (y[i] - y[j]) * (y[i] - y[j]);
}

void main(void) {
    uint i = gl_WorkGroupID.x * gl_WorkGroupSize.x + gl_LocalInvocationID.x;
    for (int j = 0; j < N; j++) {
        if (i < N) {
            if (i == j) return;
            if (distance2(i, j) <= 4 * 0.01 * 0.01) {
                if ((x[i] - x[j]) * (dx[i] - dx[j]) +
                        (y[i] - y[j]) * (dy[i] - dy[j]) <=
                    0) {
                    float dot = (dx[i] - dx[j]) * (x[i] - x[j]) +
                                (dy[i] - dy[j]) * (y[i] - y[j]);
                    new_dx[i] = dx[i] - dot / distance2(i, j) * (x[i] - x[j]);
                    new_dy[i] = dy[i] - dot / distance2(i, j) * (y[i] - y[j]);
                }
            }
        }
    }
}
```
### Build
```bash=
glslangValidator -V cal.comp -o cal.spv
xxd -i cal.spv cal.h
```
