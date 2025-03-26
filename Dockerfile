FROM debian:12
LABEL authors="Bo Zhang"

WORKDIR /opt
# run as root
# USER root
# change localtime to GMT0
# RUN cp /usr/share/zoneinfo/GMT0 /etc/localtime
RUN cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# switch to tsinghua apt source
RUN sed -i "s/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list.d/debian.sources
#RUN --mount=type=cache,id=apt,target=/var/cache/apt \
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git gcc g++ apt-utils && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install miniconda3
# Miniconda3-latest-Linux-x86_64.sh
# Miniconda3-latest-Linux-aarch64.sh
RUN mkdir -p /opt \
    && mkdir -p /laspec \
    && mkdir -p /slam \
    && wget --quiet https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda.sh \
    && bash /opt/miniconda.sh -b -p /opt/conda \
    && rm /opt/miniconda.sh \
    && /opt/conda/bin/conda clean -ay \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
    && echo "alias ll='ls -alF'" >> ~/.bashrc
# include conda executables to PATH
ENV PATH /opt/conda/bin:$PATH
# ban NumPy multithreading
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

COPY .condarc /root/
COPY laspec /laspec/
COPY setup.py /laspec/
COPY README.md /laspec/
COPY requirements.txt /laspec/
COPY projects/2024-12-22-speczoo/predict.py /slam/
COPY projects/2024-12-22-speczoo/sp.joblib /slam/

# switch to tsinghua pip/conda source
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip install ipython \
    && pip install /laspec

# CMD
CMD [ "/bin/bash" ]

# docker build -t astroslam:latest -f Dockerfile .
# docker run astroslam:latest pip list
# docker run astroslam:latest python /slam/predict.py /nfsdata/share/lamost/dr11-v1.0/fits/20191026/HD213633N331403V01/spec-58783-HD213633N331403V01_sp14-205.fits.gz