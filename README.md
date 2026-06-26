# CUDA FFT Ocean Simulation

Real-time ocean surface simulation running entirely on the GPU, using the
**Inverse Fast Fourier Transform (IFFT)** with the **JONSWAP** spectrum for
wave energy distribution — the same approach used in AAA game engines like
those behind *Sea of Thieves* and *Assassin's Creed: Black Flag*.

## Overview

The simulation generates a statistically realistic ocean surface by:

1. Building a frequency-domain spectrum using the **JONSWAP** model based on wind speed, fetch length, and sea state
2. Evolving it over time using wave dispersion relations
3. Running an **IFFT on the GPU** via **cuFFT** to recover the spatial surface each frame
4. Rendering the result in real-time using **OpenGL** with CUDA/GL interop

An **ImGui** debug panel allows live tuning of wind parameters, wave amplitude,
and choppiness without restarting.

## Tech Stack

- **CUDA** + **cuFFT** — GPU compute and FFT
- **OpenGL** + **GLEW** — real-time rendering
- **GLFW** — windowing
- **GLM** — math
- **ImGui** + **ImPlot** — debug UI

## Dependencies

- CUDA Toolkit (tested on CUDA 12.x)
- GLFW 3
- GLEW
- OpenGL

External dependencies (`imgui`, `glfw`, `glm`) are in `external/`.

## Build

```bash
# Build GLFW first (one-time)
cd external/glfw && mkdir build && cd build
cmake .. && make

# Then build the project
cd ../../..
make
```

## Run

```bash
make run
# or directly:
./output/fft_program
```

## References

- Hasselmann et al., *Measurements of Wind-Wave Growth and Swell Decay during the Joint North Sea Wave Project (JONSWAP)*, 1973
- [FFT-Ocean](https://github.com/gasgiant/FFT-Ocean) by gasgiant
- [Water](https://github.com/GarrettGunnell/Water) by GarrettGunnell
- [Video reference 1](https://youtu.be/kGEqaX4Y4bQ)
- [Video reference 2](https://youtu.be/yPfagLeUa7k)
