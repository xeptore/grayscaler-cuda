build: main.cu config.h
	nvcc -g -G -ccbin /usr/bin/clang -Wno-deprecated-gpu-targets -m64 main.cu -o ./build/grayscale -arch=sm_35 -L/opt/cuda/lib64 -lcudart_static -ldl -lrt -lpthread -ljpeg
.PHONY: build