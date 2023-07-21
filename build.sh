#!/bin/bash

cd build && \
    export CXX=/root/armnn-devenv/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++ && \
    export CC=/root/armnn-devenv/gcc-arm-10.2-2020.11-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc && \
    cmake .. \
    -DCMAKE_CXX_FLAGS="-w -static-libgcc -static-libstdc++"  \
    -DBUILD_TESTS=0 \
    -DARMCOMPUTE_ROOT=$HOME/armnn-devenv/ComputeLibrary \
    -DARMCOMPUTE_BUILD_DIR=$HOME/armnn-devenv/ComputeLibrary/build/ \
    -DARMCOMPUTENEON=1 -DARMCOMPUTECL=0 -DARMNNREF=0 \
    -DONNX_GENERATED_SOURCES=$HOME/armnn-devenv/onnx \
    -DBUILD_ONNX_PARSER=0 \
    -DBUILD_TF_LITE_PARSER=1 \
    -DBUILD_ARMNN_TFLITE_DELEGATE=1 \
    -DTENSORFLOW_ROOT=$HOME/armnn-devenv/tensorflow \
    -DTFLITE_LIB_ROOT=$HOME/armnn-devenv/tflite/build \
    -DTF_LITE_SCHEMA_INCLUDE_PATH=$HOME/armnn-devenv/tflite \
    -DFLATBUFFERS_ROOT=$HOME/armnn-devenv/flatbuffers-arm64 \
    -DFLATC_DIR=$HOME/armnn-devenv/flatbuffers-1.12.0/build \
    -DPROTOBUF_ROOT=$HOME/armnn-devenv/google/x86_64_pb_install \
    -DPROTOBUF_LIBRARY_DEBUG=$HOME/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so.23.0.0 \
    -DPROTOBUF_LIBRARY_RELEASE=$HOME/armnn-devenv/google/arm64_pb_install/lib/libprotobuf.so.23.0.0 && \
    make -j$(nproc)

cd ..

# sshpass -p "odroid" scp ./build/libarmnn.so* odroid@192.168.0.4:/mnt/usb/odroid
# sshpass -p "linaro" scp ./build/libarmnn.so* linaro@192.168.0.232:/home/linaro/Desktop/chacha
# sshpass -p "root" scp -rp -P 6666 ./build/libarmnn.so* root@119.198.183.96:/root/chacha 
sshpass -p "root" scp -rp -P 6666 ./build/*.so* root@119.198.183.96:/root/chacha 
sshpass -p "root" scp -rp -P 6666 ./build/delegate/*.so* root@119.198.183.96:/root/chacha 

# cd ..
