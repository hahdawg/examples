FROM jupyter/base-notebook:lab-2.2.5

USER root
RUN apt-get update && apt-get install -y graphviz
RUN conda update -n base -y conda
RUN conda install -y dask

WORKDIR /home/joyvan
WORKDIR /home/joyvan/mnt
WORKDIR /home/joyvan

COPY dask-docker.ipynb /home/joyvan

EXPOSE 8787

CMD  ["/bin/bash"]
