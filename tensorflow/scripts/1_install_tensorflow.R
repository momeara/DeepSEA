


# Following these directions
# https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#pip-installation
system("
setenv TF_BINARY_URL 'https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl'
pip install --upgrade $TF_BINARY_URL
")
# gave this error:
# Cannot remove entries from nonexistent file /mnt/nfs/work/momeara/tools/anaconda2/lib/python2.7/site-packages/easy-install.pth


# This issue: https://github.com/tensorflow/tensorflow/issues/135 suggests
sytem("
curl https://bootstrap.pypa.io/ez_setup.py -o - | python
setenv TF_BINARY_URL 'https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl'
pip install --upgrade $TF_BINARY_URL
")
# success

system("python -m tensorflow")
# error:
# /mnt/nfs/work/momeara/tools/anaconda2/bin/python: /lib64/libc.so.6: version `GLIBC_2.14' not found (required by /mnt/nfs/work/momeara/tools/anaconda2/lib/python2.7/site-packages/tensorflow/python/_pywrap_tensorflow.so)

# gather information
system("cat /etc/centos-release")
# CentOS release 6.7 (Final)

system("ldd --version | head -n 1")
# ldd (GNU libc) 2.12

system("gcc --version | head -n 1")
gcc (GCC) 4.4.7 20120313 (Red Hat 4.4.7-16)



# http://stackoverflow.com/questions/33655731/error-while-importing-tensorflow-in-python2-7-in-ubuntu-12-04-glibc-2-17-not-f
#   http://stackoverflow.com/a/34897674/198401
system("
mkdir glibc_dependencies
cd glibc_dependencies
wget http://launchpadlibrarian.net/137699828/libc6_2.17-0ubuntu5_amd64.deb
wget http://launchpadlibrarian.net/137699829/libc6-dev_2.17-0ubuntu5_amd64.deb
wget ftp://rpmfind.net/linux/sourceforge/m/ma/magicspecs/apt/3.0/x86_64/RPMS.lib/libstdc++-4.8.2-7mgc30.x86_64.rpm
ar p libc6_2.17-0ubuntu5_amd64.deb data.tar.gz | tar zx
ar p libc6-dev_2.17-0ubuntu5_amd64.deb data.tar.gz | tar zx
rpm2cpio libstdc++-4.8.2-7mgc30.x86_64.rpm| cpio -idmv
")

#check that it worked

system("
setenv LD_TRACE_LOADED_OBJECTS 1
python
unsetenv LD_TRACE_LOADED_OBJECTS
")
#	linux-vdso.so.1 =>  (0x00007fff71bff000)
#	libpython2.7.so.1.0 => /mnt/nfs/work/momeara/tools/anaconda2/bin/../lib/libpython2.7.so.1.0 (0x00007ff2af3ba000)
#	libpthread.so.0 => /lib64/libpthread.so.0 (0x000000322a200000)
#	libdl.so.2 => /lib64/libdl.so.2 (0x000000322aa00000)
#	libutil.so.1 => /lib64/libutil.so.1 (0x0000003232600000)
#	libm.so.6 => /lib64/libm.so.6 (0x000000322a600000)
#	libc.so.6 => /lib64/libc.so.6 (0x0000003229e00000)
#	/lib64/ld-linux-x86-64.so.2 (0x0000003229a00000)

system("
setenv LD_TRACE_LOADED_OBJECTS 1
setenv LD_LIBRARY_PATH=\"glibc_dependencies/lib/x86_64-linux-gnu/:glibc_dependencies/usr/lib64/\"
python
unsetenv LD_TRACE_LOADED_OBJECTS
")
#	linux-vdso.so.1 =>  (0x00007fffab5be000)
#	libpython2.7.so.1.0 => /mnt/nfs/work/momeara/tools/anaconda2/bin/../lib/libpython2.7.so.1.0 (0x00007fecba784000)
#	libpthread.so.0 => /scratch/momeara/ts_dep/lib/x86_64-linux-gnu/libpthread.so.0 (0x00007fecba566000)
#	libdl.so.2 => /scratch/momeara/ts_dep/lib/x86_64-linux-gnu/libdl.so.2 (0x00007fecba362000)
#	libutil.so.1 => /scratch/momeara/ts_dep/lib/x86_64-linux-gnu/libutil.so.1 (0x00007fecba15f000)
#	libm.so.6 => /scratch/momeara/ts_dep/lib/x86_64-linux-gnu/libm.so.6 (0x00007fecb9e59000)
#	libc.so.6 => /scratch/momeara/ts_dep/lib/x86_64-linux-gnu/libc.so.6 (0x00007fecb9a91000)
#	/lib64/ld-linux-x86-64.so.2 (0x0000003229a00000)

system("
setenv LD_TRACE_LOADED_OBJECTS 1
setenv LD_LIBRARY_PATH=\"glibc_dependencies/lib/x86_64-linux-gnu/:glibc_dependencies/usr/lib64/\"
glibc_dependencies/lib/x86_64-linux-gnu/ld-2.17.so `which python`
unsetenv LD_TRACE_LOADED_OBJECTS
")
#	linux-vdso.so.1 =>  (0x00007ffff17ff000)
#	libpython2.7.so.1.0 => /mnt/nfs/work/momeara/tools/anaconda2/bin/../lib/libpython2.7.so.1.0 (0x00007f8d175e8000)
#	libpthread.so.0 => /scratch/momeara/ts_dep/lib/x86_64-linux-gnu/libpthread.so.0 (0x00007f8d173ca000)
#	libdl.so.2 => /scratch/momeara/ts_dep/lib/x86_64-linux-gnu/libdl.so.2 (0x00007f8d171c6000)
#	libutil.so.1 => /scratch/momeara/ts_dep/lib/x86_64-linux-gnu/libutil.so.1 (0x00007f8d16fc3000)
#	libm.so.6 => /scratch/momeara/ts_dep/lib/x86_64-linux-gnu/libm.so.6 (0x00007f8d16cbd000)
#	libc.so.6 => /scratch/momeara/ts_dep/lib/x86_64-linux-gnu/libc.so.6 (0x00007f8d168f5000)
#	/lib64/ld-linux-x86-64.so.2 => /scratch/momeara/ts_dep/lib/x86_64-linux-gnu/ld-2.17.so (0x00007f8d179d1000)
system("
setenv LD_LIBRARY_PATH=\"glibc_dependencies/lib/x86_64-linux-gnu/:glibc_dependencies/usr/lib64/\"
/scratch/momeara/ts_dep/lib/x86_64-linux-gnu/ld-2.17.so `which python` -m tensorflow
")
# error
#  sh: error while loading shared libraries: __vdso_time: invalid mode for dlopen(): Invalid argument
#  uname: error while loading shared libraries: __vdso_time: invalid mode for dlopen(): Invalid argument

# looks like sh and uname aren't using the right ld.so version:
system("
setenv LD_LIBRARY_PATH=\"glibc_dependencies/lib/x86_64-linux-gnu/:glibc_dependencies/usr/lib64/\"
uname
")
# error:
#  uname: error while loading shared libraries: __vdso_time: invalid mode for dlopen(): Invalid argument

system("
setenv LD_LIBRARY_PATH=\"glibc_dependencies/lib/x86_64-linux-gnu/:glibc_dependencies/usr/lib64/\"
/scratch/momeara/ts_dep/lib/x86_64-linux-gnu/ld-2.17.so `which uname`
")
# Linux

# this may be of interest: http://www.linuxquestions.org/questions/linux-general-1/glibc-backward-compatibility-4175445005/

# https://github.com/tensorflow/tensorflow/issues/2105
#  They are seeing 
#  ImportError: /lib64/libc.so.6: version `GLIBC_2.15' not found (required by /usr/local/python2710/lib/python2.7/site-packages/tensorflow/python/_pywrap_tensorflow.so)
#  upgraded to GLIBC_2.17
#  ls: error while loading shared libraries: __vdso_time: invalid mode for dlopen(): Invalid argument
# Linux sjs_88_78 2.6.32-504.23.4.el6.x86_64 #1 SMP Fri May 29 10:16:43 EDT 2015 x86_64 x86_64 x86_64 GNU/Linux
#  --> recommendation is to install from source



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#install from source
##############################################

# following instuctions here https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#installing-from-sources
system("
git clone https://github.com/tensorflow/tensorflow
")

# install Bazel
# http://www.bazel.io/docs/install.html

#check JDK version
system("
java -version | head -n 1
")
# java version "1.8.0_77"
# http://www.oracle.com/technetwork/java/javase/8u77-relnotes-2944725.html says this is JRE 8

system("
git clone https://github.com/bazelbuild/bazel.git
cd bazel
./compile.sh
")
# error
#  ERROR: /nfs/home/momeara/work/sea/DeepSEA/tensorflow/bazel/src/main/cpp/util/BUILD:4:1: C++ compilation of rule '//src/main/cpp/util:util' failed: gcc failed: error executing command
#  (cd /tmp/bazel.W9oai8kT/out/bazel && \
#     exec env - \
#       PATH=/mnt/nfs/work/momeara/tools/anaconda2/bin:/mnt/nfs/home/momeara/.local/bin:/mnt/nfs/home/momeara/work/seaware/env/bin:/mnt/nfs/home/momeara/opt/node-v0.10.33-linux-x64/bin:/mnt/nfs/work/momeara/sea/sea_virtual_env/bin:/mnt/nfs/home/momeara/opt/bin:/usr/lib64/qt-3.3/bin:/usr/kerberos/sbin:/usr/kerberos/bin:/usr/local/bin:/bin:/usr/bin \
#     /usr/bin/gcc -U_FORTIFY_SOURCE '-D_FORTIFY_SOURCE=1' -fstack-protector -Wall -Wl,-z,-relro,-z,now -B/usr/bin -B/usr/bin -Wunused-but-set-parameter -Wno-free-nonheap-object -fno-omit-frame-pointer '-std=c++0x' '-frandom-seed=bazel-out/local-fastbuild/bin/src/main/cpp/util/_objs/util/src/main/cpp/util/numbers.pic.o' -fPIC -DBLAZE_OPENSOURCE -iquote . -iquote bazel-out/local-fastbuild/genfiles -iquote external/bazel_tools -iquote bazel-out/local-fastbuild/genfiles/external/bazel_tools -isystem external/bazel_tools/tools/cpp/gcc3 -Wno-builtin-macro-redefined '-D__DATE__="redacted"' '-D__TIMESTAMP__="redacted"' '-D__TIME__="redacted"' -MD -MF bazel-out/local-fastbuild/bin/src/main/cpp/util/_objs/util/src/main/cpp/util/numbers.pic.d -c src/main/cpp/util/numbers.cc -o bazel-out/local-fastbuild/bin/src/main/cpp/util/_objs/util/src/main/cpp/util/numbers.pic.o): com.google.devtools.build.lib.shell.BadExitStatusException: Process exited with status 1.
#src/main/cpp/util/numbers.cc: In function 'bool blaze_util::safe_parse_positive_int(const char*, int*)':
#    src/main/cpp/util/numbers.cc:111: error: 'vmax' cannot appear in a constant-expression
#src/main/cpp/util/numbers.cc: In function 'bool blaze_util::safe_parse_negative_int(const char*, int*)':
#    src/main/cpp/util/numbers.cc:141: error: 'vmin' cannot appear in a constant-expression
#
#  I this is because gcc is only version 4.4

#install gcc 4.9
system("
cd ~/opt
wget http://gcc.skazkaforyou.com/releases/gcc-4.9.3/gcc-4.9.3.tar.gz
tar -xzvf gcc-4.9.3.tar.gz
cd gcc-4.9.3
./contrib/download_prerequisites 
./configure --prefix=/mnt/nfs/home/momeara/opt/
make -j 10
make install
")

# based on this suggestion
# https://groups.google.com/forum/#!topic/bazel-discuss/OtPDY-AF2J4
system("
git clone https://github.com/bazelbuild/bazel.git
cd bazel
setenv CC '/mnt/nfs/home/momeara/opt/bin/gcc'
setenv CXX '/mnt/nfs/home/momeara/opt/bin/g++' 
setenv EXTRA_BAZEL_ARGS '--verbose_failures --jobs=1'
setenv CPLUS_INCLUDE_PATH '/mnt/nfs/home/momeara/opt/include'
setenv C_INCLUDE_PATH '/mnt/nfs/home/momeara/opt/include'
setenv LIBRARY_PATH '/mnt/nfs/home/momeara/opt/lib:/mnt/nfs/home/momeara/opt/lib64'
setenv LD_LIBRARY_PATH '/mnt/nfs/home/momeara/opt/lib:/mnt/nfs/home/momeara/opt/lib64'
./compile.sh
")

system("
cd tensorflow
setenv CC '/mnt/nfs/home/momeara/opt/bin/gcc'
setenv CXX '/mnt/nfs/home/momeara/opt/bin/g++' 
setenv EXTRA_BAZEL_ARGS '--verbose_failures --jobs=1'
setenv CPLUS_INCLUDE_PATH '/mnt/nfs/home/momeara/opt/include'
setenv C_INCLUDE_PATH '/mnt/nfs/home/momeara/opt/include'
setenv LIBRARY_PATH '/mnt/nfs/home/momeara/opt/lib:/mnt/nfs/home/momeara/opt/lib64'
setenv LD_LIBRARY_PATH '/mnt/nfs/home/momeara/opt/lib:/mnt/nfs/home/momeara/opt/lib64'
./configure
../bazel/output/bazel build -c opt //tensorflow/cc:tutorials_example_trainer
../bazel/output/bazel build -c opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /scratch/momeara/tensorflow_pkg
")
# Please specify the location of python. [Default is /mnt/nfs/work/momeara/tools/anaconda2/bin/python]: <ENTER>
# Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] N
# Do you wish to build TensorFlow with GPU support? [y/N] N
#
# error
#  rsync: change_dir "/mnt/nfs/work/momeara/sea/DeepSEA/tensorflow/tensorflow/bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/external" failed: No such file or directory

# This recent issue is relevant: https://github.com/tensorflow/tensorflow/issues/2040
system("
cd ~/work/sea/DeepSEA/tensorflow/tensorflow
setenv CC '/mnt/nfs/home/momeara/opt/bin/gcc'
setenv CXX '/mnt/nfs/home/momeara/opt/bin/g++' 
setenv EXTRA_BAZEL_ARGS '--verbose_failures --jobs=1'
setenv CPLUS_INCLUDE_PATH '/mnt/nfs/home/momeara/opt/include'
setenv C_INCLUDE_PATH '/mnt/nfs/home/momeara/opt/include'
setenv LIBRARY_PATH '/mnt/nfs/home/momeara/opt/lib:/mnt/nfs/home/momeara/opt/lib64'
setenv LD_LIBRARY_PATH '/mnt/nfs/home/momeara/opt/lib:/mnt/nfs/home/momeara/opt/lib64'

./configure
../bazel/output/bazel clean
../bazel/output/bazel build -c opt //tensorflow/cc:tutorials_example_trainer
../bazel/output/bazel build -c opt //tensorflow/tools/pip_package:build_pip_package

cd /mnt/nfs/work/momeara/sea/DeepSEA/tensorflow/tensorflow/bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/
ln -s ../../../../../external .
cd /mnt/nfs/work/momeara/sea/DeepSEA/tensorflow/tensorflow

bazel-bin/tensorflow/tools/pip_package/build_pip_package /scratch/momeara/tensorflow_pkg
pip install /scratch/momeara/tensorflow_pkg/tensorflow-0.9.0-py2-none-any.whl

../bazel/output/bazel build tensorflow/tensorboard:tensorboard

cd tensorflow/tensorboard
bower install
")


#### now try to install with GPU support

# install cuDNN
# https://developer.nvidia.com/cudnn
#   download cudnn-7.5-linux-x64-v5.1-rc.tgz and copy to ~/work/sea/DeepSEA/
system("
cd ~/work/sea/DeepSEA/
tar xzvf cudnn-7.5-linux-x64-v5.1-rc.tgz
rm cudnn-7.5-linux-x64-v5.1-rc.tgz
")

system("
cd ~/work/sea/DeepSEA/tensorflow/tensorflow
setenv CC '/mnt/nfs/home/momeara/opt/bin/gcc'
setenv CXX '/mnt/nfs/home/momeara/opt/bin/g++' 
setenv EXTRA_BAZEL_ARGS '--verbose_failures --jobs=1'
setenv CPLUS_INCLUDE_PATH '/mnt/nfs/home/momeara/opt/include'
setenv C_INCLUDE_PATH '/mnt/nfs/home/momeara/opt/include'
setenv LIBRARY_PATH '/mnt/nfs/home/momeara/opt/lib:/mnt/nfs/home/momeara/opt/lib64'
setenv LD_LIBRARY_PATH '/usr/local/cuda-7.5/lib64:/mnt/nfs/home/momeara/opt/lib:/mnt/nfs/home/momeara/opt/lib64'
setenv PATH /usr/local/cuda-7.5/bin:$PATH

./configure
# Please specify the location of python. [Default is /mnt/nfs/work/momeara/tools/anaconda2/bin/python]: 
# Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] n
# No Google Cloud Platform support will be enabled for TensorFlow
# Do you wish to build TensorFlow with GPU support? [y/N] y
# GPU support will be enabled for TensorFlow
# Please specify which gcc nvcc should use as the host compiler. [Default is /mnt/nfs/home/momeara/opt/bin/gcc]:
# Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave empty to use system default]: 7.5
# Please specify the location where CUDA 7.5 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
# Please specify the Cudnn version you want to use. [Leave empty to use system default]: 5.1.3
# Please specify the location where cuDNN 5.1.3 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: /mnt/nfs/work/momeara/sea/DeepSEA/cuda
# Please specify a list of comma-separated Cuda compute capabilities you want to build with.
# You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
# Please note that each additional compute capability significantly increases your build time and binary size.
# [Default is: \"3.5,5.2\"]: 5.2

../bazel/output/bazel clean
../bazel/output/bazel build -c opt --config=cuda //tensorflow/cc:tutorials_example_trainer
../bazel/output/bazel build -c opt --config=cuda //tensorflow/tools/pip_package:build_pip_package


cd /mnt/nfs/work/momeara/sea/DeepSEA/tensorflow/tensorflow/bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/org_tensorflow/
ln -s ../../../../../external .
cd /mnt/nfs/work/momeara/sea/DeepSEA/tensorflow/tensorflow

bazel-bin/tensorflow/tools/pip_package/build_pip_package /scratch/momeara/tensorflow_pkg
pip install --upgrade /scratch/momeara/tensorflow_pkg/tensorflow-0.9.0-py2-none-any.whl

../bazel/output/bazel build -c opt --config=cuda tensorflow/tensorboard:tensorboard

")
# ERROR: /mnt/nfs/home/momeara/.cache/bazel/_bazel_momeara/ef8339021629a8146b3e301bb7dc3099/external/zlib_archive/BUILD:7:1: undeclared inclusion(s) in rule '@zlib_archive//:zlib':
# this rule is missing dependency declarations for the following files included by 'external/zlib_archive/zlib-1.2.8/uncompr.c':
#   '/mnt/nfs/home/momeara/opt/lib/gcc/x86_64-unknown-linux-gnu/4.9.3/include-fixed/limits.h'
#   '/mnt/nfs/home/momeara/opt/lib/gcc/x86_64-unknown-linux-gnu/4.9.3/include-fixed/syslimits.h'
#   '/mnt/nfs/home/momeara/opt/lib/gcc/x86_64-unknown-linux-gnu/4.9.3/include/stddef.h'
#   '/mnt/nfs/home/momeara/opt/lib/gcc/x86_64-unknown-linux-gnu/4.9.3/include/stdarg.h'.


# from this issue: https://github.com/tensorflow/tensorflow/issues/469
# follow modifications here: https://github.com/noisychannel/tensorflow_install/blob/master/tf_install.sh

# diff --git a/third_party/gpus/crosstool/CROSSTOOL b/third_party/gpus/crosstool/CROSSTOOL
# index 8db81a9..f3a55b7 100644
# --- a/third_party/gpus/crosstool/CROSSTOOL
# +++ b/third_party/gpus/crosstool/CROSSTOOL
# @@ -40,7 +40,7 @@ toolchain {
#  
#    tool_path { name: "ar" path: "/usr/bin/ar" }
#    tool_path { name: "compat-ld" path: "/usr/bin/ld" }
# -  tool_path { name: "cpp" path: "/usr/bin/cpp" }
# +  tool_path { name: "cpp" path: "/mnt/nfs/home/momeara/opt/bin/cpp" }
#    tool_path { name: "dwp" path: "/usr/bin/dwp" }
#    # As part of the TensorFlow release, we place some cuda-related compilation
#    # files in third_party/gpus/crosstool/clang/bin, and this relative
# @@ -51,16 +51,21 @@ toolchain {
#    # and the device compiler to use "-std=c++11".
#    cxx_flag: "-std=c++11"
#    linker_flag: "-lstdc++"
# -  linker_flag: "-B/usr/bin/"
# +  linker_flag: "-B/mnt/nfs/home/momeara/opt/bin/"
# +  linker_flag: "-L/mnt/nfs/home/momeara/opt/lib64/"
# +  linker_flag: "-Wl,-rpath,/mnt/nfs/home/momeara/opt/lib64/"
# +
#  
#    # TODO(bazel-team): In theory, the path here ought to exactly match the path
#    # used by gcc. That works because bazel currently doesn't track files at
#    # absolute locations and has no remote execution, yet. However, this will need
#    # to be fixed, maybe with auto-detection?
# -  cxx_builtin_include_directory: "/usr/lib/gcc/"
# -  cxx_builtin_include_directory: "/usr/local/include"
# +
# +  cxx_builtin_include_directory: "/mnt/nfs/home/momeara/opt/lib/gcc/"
# +  cxx_builtin_include_directory: "/mnt/nfs/home/momeara/opt/lib64/"
# +  cxx_builtin_include_directory: "/mnt/nfs/home/momeara/opt/include"
#    cxx_builtin_include_directory: "/usr/include"
# -  tool_path { name: "gcov" path: "/usr/bin/gcov" }
# +  tool_path { name: "gcov" path: "/mnt/nfs/home/momeara/opt/bin/gcov" }
#  
#    # C(++) compiles invoke the compiler (as that is the one knowing where
#    # to find libraries), but we provide LD so other rules can invoke the linker.
# @@ -70,8 +75,8 @@ toolchain {
#    tool_path { name: "objcopy" path: "/usr/bin/objcopy" }
#    objcopy_embed_flag: "-I"
#    objcopy_embed_flag: "binary"
# -  tool_path { name: "objdump" path: "/usr/bin/objdump" }
# -  tool_path { name: "strip" path: "/usr/bin/strip" }
# +  tool_path { name: "objdump" path: "/mnt/nfs/home/momeara/opt/bin/objdump" }
# +  tool_path { name: "strip" path: "/mnt/nfs/home/momeara/opt/bin/strip" }
#  
#    # Anticipated future default.
#    unfiltered_cxx_flag: "-no-canonical-prefixes"
# @@ -165,7 +170,7 @@ toolchain {
#  
#    tool_path { name: "ar" path: "/usr/bin/libtool" }
#    tool_path { name: "compat-ld" path: "/usr/bin/ld" }
# -  tool_path { name: "cpp" path: "/usr/bin/cpp" }
# +  tool_path { name: "cpp" path: "/mnt/nfs/home/momeara/opt/bin/cpp" }
#    tool_path { name: "dwp" path: "/usr/bin/dwp" }
#    tool_path { name: "gcc" path: "clang/bin/crosstool_wrapper_driver_is_not_gcc" }
#    cxx_flag: "-std=c++11"
# @@ -178,7 +183,7 @@ toolchain {
#    # TODO(ulfjack): This is wrong on so many levels. Figure out a way to auto-detect the proper
#    # setting from the local compiler, and also how to make incremental builds correct.
#    cxx_builtin_include_directory: "/"
# -  tool_path { name: "gcov" path: "/usr/bin/gcov" }
# +  tool_path { name: "gcov" path: "/mnt/nfs/home/momeara/opt/bin/gcov" }
#    tool_path { name: "ld" path: "/usr/bin/ld" }
#    tool_path { name: "nm" path: "/usr/bin/nm" }
#    tool_path { name: "objcopy" path: "/usr/bin/objcopy" }
# 
# diff --git a/third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc b/third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc
# index 071997c..60e3491 100755
# --- a/third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc
# +++ b/third_party/gpus/crosstool/clang/bin/crosstool_wrapper_driver_is_not_gcc
# @@ -47,12 +47,20 @@ import pipes
#  
#  # "configure" uses the specific format to substitute the following string.
#  # If you change it, make sure you modify "configure" as well.
# -CPU_COMPILER = ('/usr/bin/gcc')
# -GCC_HOST_COMPILER_PATH = ('/usr/bin/gcc')
# +# Unofficial setting. DO NOT SUBMIT!!!
# +# Unofficial setting. DO NOT SUBMIT!!!
# +# Unofficial setting. DO NOT SUBMIT!!!
# +# Unofficial setting. DO NOT SUBMIT!!!
# +CPU_COMPILER = ('/mnt/nfs/home/momeara/opt/bin/gcc')
# +# Unofficial setting. DO NOT SUBMIT!!!
# +# Unofficial setting. DO NOT SUBMIT!!!
# +# Unofficial setting. DO NOT SUBMIT!!!
# +# Unofficial setting. DO NOT SUBMIT!!!
# +GCC_HOST_COMPILER_PATH = ('/mnt/nfs/home/momeara/opt/bin/gcc')
#  
#  CURRENT_DIR = os.path.dirname(sys.argv[0])
#  NVCC_PATH = CURRENT_DIR + '/../../../cuda/bin/nvcc'
# -LLVM_HOST_COMPILER_PATH = ('/usr/bin/gcc')
# +LLVM_HOST_COMPILER_PATH = ('/mnt/nfs/home/momeara/opt/bin/gcc')
#  PREFIX_DIR = os.path.dirname(GCC_HOST_COMPILER_PATH)

# additionally, it said it couldn't find some programs in /usr/bin (ld, mn, as), so I symlinked them to /mnt/nfs/home/momeara/opt/bin and that seemd to work

