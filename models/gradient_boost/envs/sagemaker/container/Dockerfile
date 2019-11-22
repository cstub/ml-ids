FROM nvidia/cuda:10.1-base

# Install Miniconda 3
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
    apt-get clean

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# Copy Conda environment file
COPY models/gradient_boost/project/conda.yaml /opt/ml/code/conda.yaml

# Install Conda environment
RUN conda env create -f /opt/ml/code/conda.yaml

# Copy project files
ADD ml_ids /opt/ml/code/ml_ids
ADD models/gradient_boost/project /opt/ml/code/models/gradient_boost/project
COPY setup.cfg /opt/ml/code/setup.cfg
COPY setup.py /opt/ml/code/setup.py

# Activate conda env
RUN echo "source activate ml-ids-gradient-boost-catboost" > ~/.bashrc
ENV PATH /opt/conda/envs/ml-ids-gradient-boost-catboost/bin:$PATH

# Copy train script and make it executable
COPY models/gradient_boost/envs/sagemaker/container/train.py /opt/ml/code/train
RUN chmod +x /opt/ml/code/train
ENV PATH="/opt/ml/code:${PATH}"

WORKDIR /opt/ml/code