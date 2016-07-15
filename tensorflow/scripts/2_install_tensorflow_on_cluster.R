
# from Tent Balius June 8th 2016
# 7 GPUs on cluster
# http://wiki.docking.org/index.php/Gpus
#
# /nfs/work/tbalius/MOR/run_amber/run.pmemd_cuda_wraper.csh
#   gpu.q SGE queue: gpu.q
#   Teague's executable wrapper: /nfs/ge/bin/one-one-gpu
#
#    calls /nfs/ge/bin/request-gpu-device.py:
#      request-gpu-device.py
#      Teague Sterling 2015
      
#      Creates a simple, opt-in, process-based queue that can be 
#      accessed by the file system. Intended to allow a process
#      to request a specific core to limit to via taskset
#     
#      Example:
#      DEVICE_ID=./request-gpu-device.py 
#      some-cuda-command --device=$DEVICE_ID


##############################

# check on status of SGE queues:
qstat -g c
#  CLUSTER QUEUE                   CQLOAD   USED    RES  AVAIL  TOTAL aoACDS  cdsuE
#  --------------------------------------------------------------------------------
#  all.q                             0.55    555      0    471   1114      0     88
#  fast.q                            0.47      0      0      8      8      0      0
#  gpu.q                             0.54      0      0      7      7      0      0
#  int.q                             0.52      0      0     30     31      0      1

# start interactive session a GPU node
# http://gridscheduler.sourceforge.net/htmlman/htmlman1/qsub.html
qlogin -q gpu.q

### template for submitting a job to the gpu.q
cat << EOF > qsub.job.csh
#\$ -S /bin/csh
#\$ -cwd
#\$ -q gpu.q
#\$ -o stdout
#\$ -e stderr
cmd="/nfs/ge/bin/on-one-gpu - /path/to/gpu/executable"
set SCRATCH_DIR = /scratch
if ! (-d \$SCRATCH_DIR ) then
    SCRATCH_DIR=/tmp
endif
set username = `whoami`
set TASK_DIR = "\$SCRATCH_DIR/\${username}/\$JOB_ID"
echo \$TASK_DIR
mkdir -p \${TASK_DIR}
cd \${TASK_DIR}
pwd
$cmd
EOF
qsub qsub.job.csh

################################################################################



# In order to build or run TensorFlow with GPU support,
# both NVIDIA's Cuda Toolkit (>= 7.0) and cuDNN (>= v2) need to be installed.
#
# TensorFlow GPU support requires having a GPU card with NVidia Compute Capability >= 3.0.

# Do I have a card with NVidia Compute Capability?
# http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/

/sbin/lspci | grep -i nvidia
# 02:00.0 VGA compatible controller: NVIDIA Corporation GM204 [GeForce GTX 980] (rev a1)
# 02:00.1 Audio device: NVIDIA Corporation GM204 High Definition Audio Controller (rev a1)
# 03:00.0 VGA compatible controller: NVIDIA Corporation GM204 [GeForce GTX 980] (rev a1)
# 03:00.1 Audio device: NVIDIA Corporation GM204 High Definition Audio Controller (rev a1)
# 83:00.0 VGA compatible controller: NVIDIA Corporation GM204 [GeForce GTX 980] (rev a1)
# 83:00.1 Audio device: NVIDIA Corporation GM204 High Definition Audio Controller (rev a1)
# 84:00.0 VGA compatible controller: NVIDIA Corporation GM204 [GeForce GTX 980] (rev a1)
# 84:00.1 Audio device: NVIDIA Corporation GM204 High Definition Audio Controller (rev a1)

# This looks like it
#   https://developer.nvidia.com/cuda-gpus
#   http://www.geforce.com/hardware/notebook-gpus/geforce-gtx-980

#       Compute Capability 5.2

# The CUDA development environment relies on tight integration with the host development environment, including the host compiler and C runtime libraries, and is therefore only supported on distribution versions that have been qualified for this CUDA Toolkit release. - See more at: http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/#sthash.a1TI2WkV.dpuf
#
#  Distribution Kernel GCC   GLIBC
#  CentOS 6.x   2.6.32 4.4.7 2.12


cat /etc/*release
# CentOS release 6.8 (Final)
# LSB_VERSION=base-4.0-amd64:base-4.0-noarch:core-4.0-amd64:core-4.0-noarch:graphics-4.0-amd64:graphics-4.0-noarch:printing-4.0-amd64:printing-4.0-noarch
# CentOS release 6.8 (Final)
# CentOS release 6.8 (Final)
#   -> Distribution = CentOS 6.8 -->  OK

uname -a
# Linux n-1-126.cluster.ucsf.bkslab.org 2.6.32-573.18.1.el6.x86_64 #1 SMP Tue Feb 9 22:46:17 UTC 2016 x86_64 x86_64 x86_64 GNU/Linux
#   -> Kernel 2.6.32 --> OK

which gcc
# /mnt/nfs/home/momeara/opt/bin/gcc
#  -> This is the version of gcc that I built for tensorflow but it looks like it

gcc -v
# Using built-in specs.
# COLLECT_GCC=gcc
# COLLECT_LTO_WRAPPER=/mnt/nfs/home/momeara/opt/libexec/gcc/x86_64-unknown-linux-gnu/4.9.3/lto-wrapper
# Target: x86_64-unknown-linux-gnu
# Configured with: ./configure --prefix=/mnt/nfs/home/momeara/opt/
#     Thread model: posix
# gcc version 4.9.3 (GCC) --> too new


###############################

#install gcc 4.4.7
system("
cd ~/opt
wget http://gcc.skazkaforyou.com/releases/gcc-4.4.7/gcc-4.4.7.tar.gz
tar -xzf gcc-4.4.7.tar.gz
rm -rf gcc-4.4.7.tar.gz
cd gcc-4.4.7
./configure --prefix=/mnt/nfs/home/momeara/opt/
make -j 10
make install
gcc -v
")
# Using built-in specs.
# Target: x86_64-unknown-linux-gnu
# Configured with: ./configure --prefix=/mnt/nfs/home/momeara/opt/
# Thread model: posix
# gcc version 4.4.7 (GCC) 

#########################################

# it looks like the 7.5 drivers are already installed at /usr/local/cuda-7.5
# Set the paths and check that the drivers are installed:
system("
setenv PATH /usr/local/cuda-7.5/bin:$PATH
setenv LD_LIBRARY_PATH /usr/local/cuda-7.5/lib64
cat /proc/driver/nvidia/version
")
# NVRM version: NVIDIA UNIX x86_64 Kernel Module  352.79  Wed Jan 13 16:17:53 PST 2016
# GCC version:  gcc version 4.4.7 20120313 (Red Hat 4.4.7-16) (GCC)

system("
setenv PATH /usr/local/cuda-7.5/bin:$PATH
setenv LD_LIBRARY_PATH /usr/local/cuda-7.5/lib64
nvcc -V
")
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2015 NVIDIA Corporation
# Built on Tue_Aug_11_14:27:32_CDT_2015
# Cuda compilation tools, release 7.5, V7.5.17

# test compiling samples
system("
setenv PATH /usr/local/cuda-7.5/bin:$PATH
setenv LD_LIBRARY_PATH /usr/local/cuda-7.5/lib64
mkdir -p /scratch/$USER/cuda-7.5_samples
cp -r /usr/local/cuda-7.5/samples /scratch/$USER/cuda-7.5_samples
cd /scratch/momeara/cuda-7.5_samples/samples
make
")

##########################
# install python lockfile to run Teague's GPU script
system("
pip install lockfile
")

system("
/nfs/ge/bin/on-one-gpu - /scratch/momeara/cuda-7.5_samples/bin/x86_64/linux/release/deviceQuery
")
# Selected CUDA device ID: 1
# /scratch/momeara/cuda-7.5_samples/bin/x86_64/linux/release/deviceQuery Starting...
# 
#  CUDA Device Query (Runtime API) version (CUDART static linking)
# 
# cudaGetDeviceCount returned 38
# -> no CUDA-capable device is detected
# Result = FAIL

# If a CUDA-capable device and the CUDA Driver are installed but deviceQuery reports that no CUDA-capable devices are present, this likely means that the /dev/nvidia* files are missing or have the wrong permissions.

system("
ls -lh /dev/nvidia*
")
# crw-rw-rw-. 1 root root 195,   0 Mar 13 04:09 /dev/nvidia0
# crw-rw-rw-. 1 root root 195,   1 Mar 13 04:09 /dev/nvidia1
# crw-rw-rw-. 1 root root 195,   2 Mar 13 04:09 /dev/nvidia2
# crw-rw-rw-. 1 root root 195,   3 Mar 13 04:09 /dev/nvidia3
# crw-rw-rw-. 1 root root 195, 255 Mar 13 04:09 /dev/nvidiactl
# crw-rw-rw-. 1 root root 246,   0 Mar 14 12:00 /dev/nvidia-uvm

system("
/sbin/modinfo nvidia | grep version
")
# version:        352.93


system("
nvidia-smi -a
")
# Failed to initialize NVML: GPU access blocked by the operating system

# http://stackoverflow.com/questions/31078132/fail-to-use-devicequery-from-cuda-in-ubuntu
# http://www.zhiyuanlin.com/academic-notes/debugging-using-scala-cuda-in-spark2-ubuntu1404
#  That's because cuda-driver version mismatch.

# https://www.mail-archive.com/debian-bugs-dist@lists.debian.org/msg1408518.html
#  this suggests you can get info from dmesg

system("
dmesg | grep NVRM
")
# NVRM: API mismatch: the client has the version 352.93, but
# NVRM: this kernel module has the version 352.79.  Please
# NVRM: make sure that this kernel module and all NVIDIA driver
# NVRM: components have the same version.


# Driver version: 352.79
#  http://www.nvidia.com/download/driverResults.aspx/97645/en
#  http://us.download.nvidia.com/XFree86/Linux-x86_64/352.79/README/installdriver.html
system("
cd  ~/opt
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/352.79/NVIDIA-Linux-x86_64-352.79.run
")




#
#https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#optional-install-cuda-gpus-on-linux



# preliminary instructions for how to use GPUs with tensor flow
# https://www.tensorflow.org/versions/r0.9/how_tos/using_gpu/index.html


