FROM ubuntu:18.04
ENV TERM linux
ENV DEBIAN_FRONTEND noninteractive

# Forward system proxy setting
# ARG proxy
# ENV http_proxy $proxy
# ENV https_proxy $proxy

# Basic apt update
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends locales ca-certificates &&  rm -rf /var/lib/apt/lists/*

# Set the locale to en_US.UTF-8, because the Yocto build fails without any locale set.
RUN locale-gen en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Again, off the certificare
RUN echo "check_certificate = off" >> ~/.wgetrc
RUN echo "[global] \n\
trusted-host = pypi.python.org \n \
\t               pypi.org \n \
\t              files.pythonhosted.org" >> /etc/pip.conf

# Get basic packages
RUN apt-get update && apt-get install -y \
    apparmor \
    aufs-tools \
    automake \
    bash-completion \
    btrfs-tools \
    build-essential \
    cmake \
    createrepo \
    curl \
    dpkg-sig \
    g++ \
    gcc \
    git \
    iptables \
    jq \
    libapparmor-dev \
    libc6-dev \
    libcap-dev \
    libsystemd-dev \
    libyaml-dev \
    mercurial \
    net-tools \
    parallel \
    pkg-config \
    python-dev \
    python-mock \
    python-pip \
    python-setuptools \
    python-websocket \
    golang-go \
    iproute2 \
    iputils-ping \
    vim-common \
    vim \
    wget \
    libtool \
    unzip \
    scons \
    curl \
    autoconf \
    libtool \
    build-essential \
    g++ \ 
    cmake && rm -rf /var/lib/apt/lists/*

# Download the Android NDK and make a standalone toolchain
RUN mkdir -p /home/armnn-devenv/toolchains && \
    cd /home/armnn-devenv/toolchains && \
    wget https://dl.google.com/android/repository/android-ndk-r17b-linux-x86_64.zip && \
    unzip android-ndk-r17b-linux-x86_64.zip

ENV NDK /home/armnn-devenv/toolchains/android-ndk-r17b 

RUN $NDK/build/tools/make_standalone_toolchain.py \
    --arch arm64 \
    --api 26 \
    --stl=libc++ \
    --install-dir=/home/armnn-devenv/toolchains/aarch64-android-r17b

ENV PATH=/home/armnn-devenv/toolchains/aarch64-android-r17b/bin:$PATH

# Build the Boost C++ libraries
RUN mkdir /home/armnn-devenv/boost && \
    cd /home/armnn-devenv/boost && \
    wget https://boostorg.jfrog.io/artifactory/main/release/1.64.0/source/boost_1_64_0.tar.bz2 && \
    tar xvf boost_1_64_0.tar.bz2

RUN echo "using gcc : arm : aarch64-linux-android-clang++ ;" > /home/armnn-devenv/boost/user-config.jam && \
    cd /home/armnn-devenv/boost/boost_1_64_0 && \
    ./bootstrap.sh --prefix=/home/armnn-devenv/boost/install && \
    ./b2 install --user-config=/home/armnn-devenv/boost/user-config.jam \
    toolset=gcc-arm link=static cxxflags=-fPIC --with-filesystem \
    --with-test --with-log --with-program_options -j8

# Build the Compute Library
RUN cd /home/armnn-devenv && \
    git clone https://github.com/Piorosen/luna-acl.git ComputeLibrary && \
    cd ComputeLibrary && \
    scons arch=armv8.2-a neon=1 opencl=0 embed_kernels=1 extra_cxx_flags="-fPIC" \
    benchmark_tests=0 validation_tests=0 os=android Werror=0 -j8

# RUN mkdir /home/armnn-devenv/google && \
RUN mkdir -p /home/armnn-devenv/google && \
    cd /home/armnn-devenv/google && \
    git clone https://github.com/google/protobuf.git && \
    cd protobuf && \
    git checkout -b v3.5.2 v3.5.2 && \
    ./autogen.sh && \
    mkdir x86_build && \
    cd x86_build && \
    ../configure --prefix=/home/armnn-devenv/google/x86_pb_install && \
    make install -j8
    
RUN cd /home/armnn-devenv/google/protobuf && \
    mkdir arm64_build && cd arm64_build && \
    CC=aarch64-linux-android-clang \
    CXX=aarch64-linux-android-clang++ \
    CFLAGS="-fPIE -fPIC" LDFLAGS="-pie -llog" \
    ../configure --host=aarch64-linux-android \
    --prefix=/home/armnn-devenv/google/arm64_pb_install \
    --with-protoc=/home/armnn-devenv/google/x86_pb_install/bin/protoc && \
    make install -j8

# clone Tensorflow
RUN cd /home/armnn-devenv/google/ && \
    git clone https://github.com/tensorflow/tensorflow.git 

# Clone ARMNN
RUN cd /home/armnn-devenv/ && \
    git clone https://github.com/Piorosen/luna-armnn.git armnn 

# Generate TensorFlow protobuf definitions
RUN cd /home/armnn-devenv/google/tensorflow && \
    git checkout a0043f9262dc1b0e7dc4bdf3a7f0ef0bebc4891e && \
    /home/armnn-devenv/armnn/scripts/generate_tensorflow_protobuf.sh \
    /home/armnn-devenv/google/tf_pb /home/armnn-devenv/google/x86_pb_install

ENV PATH=/home/armnn-devenv/toolchains/android-ndk-r17b:$PATH

# RUN cd $HOME/armnn-devenv && \
#     wget -O flatbuffers-1.12.0.tar.gz https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz && \
#     tar xf flatbuffers-1.12.0.tar.gz && \
#     cd flatbuffers-1.12.0 && \
#     rm -f CMakeCache.txt

# Build Google Flatbuffers for ARMNN TFLite Parser
RUN cd /home/armnn-devenv && \
    wget -O flatbuffers-1.12.0.tar.gz https://github.com/google/flatbuffers/archive/v1.12.0.tar.gz && \
    tar xf flatbuffers-1.12.0.tar.gz && \
    mv flatbuffers-1.12.0 flatbuffers && \
    cd flatbuffers && \
    cd android && cd jni && \
    rm -rf Application.mk && \
    echo "APP_STL := c++_static" >> Application.mk && \
    echo "NDK_TOOLCHAIN_VERSION" := clang >> Application.mk && \
    echo "APP_CPPFLAGS :=-std=c++11" >> Application.mk && \
    echo "APP_ABI := arm64-v8a" >> Application.mk && \
    echo "APP_PLATFORM := android-23" >> Application.mk && \
    echo "NDK_PLATFORM=android-23" >> Application.mk && \
    cd ../ && ndk-build -B

COPY ./docker-entrypoint.sh /usr/bin
RUN chmod +x /usr/bin/docker-entrypoint.sh

RUN apt update && apt install -y openssh-server && rm -rf /var/lib/apt/lists/* && \
        mkdir /var/run/sshd && \ 
        echo 'root:root' | chpasswd && \
        sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
        sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config && \
        mkdir /root/.ssh

EXPOSE 22

# 
# RUN rm -rf /home/armnn-devenv/toolchains && \
#     mkdir /home/armnn-devenv/toolchains && \
#     cd /home/armnn-devenv/toolchains && \
#     wget "https://dl.google.com/android/repository/android-ndk-r25b-linux.zip?utm_source=developer.android.com&utm_medium=referral" -O android-ndk-r17b-linux-x86_64.zip && \
#     unzip android-ndk-r17b-linux-x86_64.zip && \ 
#     rm -rf android-ndk-r17b-linux-x86_64.zip

# ENV NDK=/home/armnn-devenv/toolchains/android-ndk-r25b 
# RUN rm -rf $NDK/build/tools/make_standalone_toolchain.py
# COPY ./make_standalone_toolchain.py $NDK/build/tools
# RUN python3 $NDK/build/tools/make_standalone_toolchain.py \
#     --arch arm64 \
#     --api 26 \
#     --stl=libc++ \
#     --install-dir=/home/armnn-devenv/toolchains/aarch64-android-r25b-yaho \
#     --force

# ENV PATH=/home/armnn-devenv/toolchains/aarch64-android-r25b-yaho/bin:$PATH

# BUILD ARMNN
RUN cd /home/armnn-devenv/armnn && \
    mkdir build && cd build && \
    export CXX=aarch64-linux-android-clang++ && \
    export CC=aarch64-linux-android-clang && \
    cmake .. \
    -DCMAKE_TRY_COMPILE_TARGET_TYPE="STATIC_LIBRARY" \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_SYSTEM_VERSION=26 \
    -DCMAKE_CXX_STANDARD=14 \
    -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
    -DBUILD_TESTS=0 \
    -DCMAKE_EXE_LINKER_FLAGS="-pie -llog -lz" \
    -DARMCOMPUTE_ROOT=/home/armnn-devenv/ComputeLibrary/ \
    -DARMCOMPUTE_BUILD_DIR=/home/armnn-devenv/ComputeLibrary/build \
    -DBOOST_ROOT=/home/armnn-devenv/boost/install/ \
    -DARMCOMPUTENEON=1 -DARMCOMPUTECL=0 -DARMNNREF=0 \
    -DTF_GENERATED_SOURCES=/home/armnn-devenv/google/tf_pb/ -DBUILD_TF_PARSER=1 \
    -DCMAKE_ANDROID_STANDALONE_TOOLCHAIN=/home/armnn-devenv/toolchains/aarch64-android-r17b/ \
    -DCMAKE_CXX_FLAGS="-w -fPIE -fPIC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPROTOBUF_ROOT=/home/armnn-devenv/google/arm64_pb_install/ && \
    make -j8





ENTRYPOINT ["/usr/sbin/sshd -D"]

#To do:
# 1. Flatbuffers build Application.mk hardcode value need to fix.


