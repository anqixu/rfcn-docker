# Setup base environment
FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
RUN sed -i'' 's/archive\.ubuntu\.com/ca\.archive\.ubuntu\.com/' /etc/apt/sources.list # switch to a more trusted and stable mirror
RUN apt-get -y update
RUN apt-get -y dist-upgrade
ENV TERM=xterm

# Install dependencies for Caffe & R-FCN / faster R-CNN
RUN apt-get -y install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
RUN apt-get -y install --no-install-recommends libboost-all-dev
RUN apt-get -y install libatlas-base-dev
RUN apt-get -y install libgflags-dev libgoogle-glog-dev liblmdb-dev
RUN apt-get -y install python python-setuptools cython python-numpy python-dev python-pip python-opencv python-scipy python-skimage-lib python-matplotlib ipython python-h5py python-leveldb python-networkx python-nose python-pandas python-dateutil python-protobuf python-gflags python-yaml python-six
RUN pip install --upgrade pip
RUN pip install easydict Pillow

# Create matplotlib cache file
RUN python -c 'import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot'

# Setup git (required for R-FCN / FRCNN
RUN apt-get -y install git
RUN git config --global user.email “hello@elementai.com”
RUN git config --global user.name “Element AI”

# Setup environment variables
ENV SRC_ROOT=/root/code
ENV RFCN_ROOT=$SRC_ROOT/py-R-FCN
ENV FRCN_ROOT=$SRC_ROOT/py-R-FCN
RUN mkdir -p $SRC_ROOT

# Pull R-FCN (which also contains codebase for FRCNN)
# NOTE: uncomment only one of the following OPTIONs

# OPTION A: original py-R-FCN codebase
WORKDIR $SRC_ROOT
RUN git clone https://github.com/Orpine/py-R-FCN.git $RFCN_ROOT
WORKDIR $RFCN_ROOT
RUN git clone https://github.com/Microsoft/caffe.git
# Check out the specific commit of caffe recommended by R-FCN
WORKDIR $RFCN_ROOT/caffe
RUN git checkout 1a2be8e

# OPTION B: py-R-FCN-multiGPU, with hook to MIOvision car dataset
#WORKDIR $SRC_ROOT
#RUN git clone -b miovision https://github.com/dgistJihunJung/py-R-FCN-multiGPU.git $RFCN_ROOT

# OPTION C: py-R-FCN-multiGPU + soft NMS codebase
#WORKDIR $SRC_ROOT
#RUN git clone https://github.com/bharatsingh430/soft-nms.git $RFCN_ROOT

# Patch and configure caffe
WORKDIR $FRCN_ROOT/caffe/include/caffe/layers
COPY patches/python_layer.hpp.patch python_layer.hpp.patch
RUN patch python_layer.hpp python_layer.hpp.patch
WORKDIR $RFCN_ROOT/caffe/python
RUN for req in $(cat requirements.txt); do pip install $req; done
WORKDIR $RFCN_ROOT/caffe
RUN mv Makefile.config.example Makefile.config

# Configure R-FCN / FRCNN
WORKDIR $RFCN_ROOT/lib
RUN make
COPY patches/Makefile.config.patch Makefile.config.patch







# TODO: @HERE

RUN patch Makefile.config Makefile.config.patch
# TODO: probably don't need the patches below
#COPY patches/Makefile.patch Makefile.patch
#RUN patch Makefile Makefile.patch

WORKDIR $RFCN_ROOT/caffe
RUN make -j8
RUN make pycaffe

# Fetch updated URLs of FRCNN models from github
RUN apt-get -y install wget
WORKDIR $FRCN_ROOT/data/scripts
RUN rm *
RUN wget https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/data/scripts/fetch_faster_rcnn_models.sh
RUN chmod +x fetch_faster_rcnn_models.sh
RUN wget https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/data/scripts/fetch_imagenet_models.sh
RUN chmod +x fetch_imagenet_models.sh
RUN wget https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/data/scripts/fetch_selective_search_data.sh
RUN chmod +x fetch_selective_search_data.sh

# Fetch VGG16 and COCO models from the web
WORKDIR $FRCN_ROOT
RUN ./data/scripts/fetch_faster_rcnn_models.sh
WORKDIR $FRCN_ROOT/data
RUN rm faster_rcnn_models.tgz
WORKDIR $FRCN_ROOT/data/faster_rcnn_models
RUN wget --content-disposition https://dl.dropboxusercontent.com/s/cotx0y81zvbbhnt/coco_vgg16_faster_rcnn_final.caffemodel?dl=0

# Fetch R-FCN VOC and COCO models from the web
WORKDIR $RFCN_ROOT/data
RUN wget --content-disposition https://www.dropbox.com/s/qvjinh7c6jpaovw/rfcn_models.tar.gz?dl=0
RUN tar zvxf rfcn_models.tar.gz
RUN rm rfcn_models.tar.gz
WORKDIR $RFCN_ROOT/data/rfcn_models
RUN wget --content-disposition https://www.dropbox.com/s/2ftauwlqbu92a0v/rfcn_model_coco.tar.gz?dl=0
RUN tar zvxf rfcn_model_coco.tar.gz
RUN rm rfcn_model_coco.tar.gz


# Copy over additional scripts
WORKDIR $FRCN_ROOT/tools
COPY patches/demo_headless.py demo_headless.py
RUN chmod +x demo_headless.py
COPY patches/demo_rfcn_headless.py demo_rfcn_headless.py
RUN chmod +x demo_rfcn_headless.py

# Convenience
#RUN apt-get -y install nano screen # TODO: re-enable
WORKDIR $SRC_ROOT
CMD ["tail", "-f", "/dev/null"]