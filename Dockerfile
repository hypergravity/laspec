FROM continuumio/miniconda3
LABEL authors="Bo Zhang"

RUN mkdir -p /laspec \
    && mkdir -p /slam \
WORKDIR /slam

# switch to tsinghua apt source
#RUN sed -i "s/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g" /etc/apt/sources.list.d/debian.sources
#RUN --mount=type=cache,id=apt,target=/var/cache/apt \
#RUN apt-get update --fix-missing && \
#    apt-get install -y wget bzip2 ca-certificates curl git gcc g++ apt-utils && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

# ban NumPy multithreading
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

# copy laspec
COPY .condarc /root/
COPY laspec /laspec/
COPY setup.py /laspec/
COPY README.md /laspec/
COPY requirements.txt /laspec/
COPY projects/2024-12-22-speczoo/predict.py /slam/
COPY projects/2024-12-22-speczoo/sp.joblib /slam/
COPY projects/2024-12-22-speczoo/upgrade_astropy.py /slam/

# switch to tsinghua pip/conda source
RUN --mount=type=cache,id=pip,uid=0,gid=0,target=/root/.cache \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip install ipython \
    && pip install /laspec \
    && ipython /slam/upgrade_astropy.py

# CMD
CMD ["ipython", "/slam/predict.py"]

# docker build -t astroslam:latest -f Dockerfile --no-cache .
# docker run astroslam:latest pip list
# docker run -v /nfsdata/share/lamost/dr11-v1.0/fits/20191026/HD213633N331403V01/spec-58783-HD213633N331403V01_sp14-205.fits.gz:/slam/input.fits astroslam:latest python /slam/predict.py
# docker run -it -v /nfsdata/share/lamost/dr11-v1.0/fits/20191026/HD213633N331403V01/spec-58783-HD213633N331403V01_sp14-205.fits.gz:/slam/input.fits astroslam:latest bash

http://data.astropy.org/coordinates/sites.json
https://datacenter.iers.org/data/9/finals2000A.all
