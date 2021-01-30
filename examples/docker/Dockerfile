FROM ubuntu:18.04

# Adds metadata to the image as a key value pair example LABEL version="1.0"

##Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8



# Set environmental variable within container
WORKDIR /ds
ENV HOME=/ds

# Working directory. Exists only in container. Sets working directory for
# any RUN, CMD, ENTRYPOINT, COPY, and ADD instructions. Use this instead
# of 'RUN cd .."

# When we use the -v param when starting the container, the volume will be 
# mounted to /ds
WORKDIR /ds/mnt
VOLUME /ds/mnt

COPY .vimrc /ds/.vimrc
WORKDIR /ds/.vim
COPY .vim . /ds/.vim/

WORKDIR /ds

# Don't prompt y for apt-get
RUN echo "APT::Get::Assume-Yes \"true\";" >> /etc/apt/apt.conf.d/90forceyes
RUN apt-get update && apt-get install -y curl git software-properties-common

RUN add-apt-repository -y ppa:jonathonf/vim
RUN apt update
RUN apt -y install vim

RUN curl -sL install-node.now.sh/lts | bash -s -- --yes

# RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
#     build-essential \
#     curl \
#     htop \
#     python3-dev \
#     python-pip \
#     python-setuptools \
#     python-virtualenv \
#     && \
# apt-get clean && \
# rm -rf /var/lib/apt/lists/*

# ENV PATH="/root/anaconda3/bin:${PATH}"
# ARG PATH="/root/anaconda3/bin:${PATH}

# RUN wget \
# 	https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh \
#    && mkdir /root/.conda \
#   && bash Anaconda3-2020.11-Linux-x86_64.sh -b \
#  && rm -f Anaconda3-2020.11-Linux-x86_64.sh 

# RUN pip --no-cache-dir install --upgrade \
#         altair \
#         sklearn-pandas

# Open Ports for Jupyter
EXPOSE 8888

#Setup File System
ENV SHELL=/bin/bash

# Run a shell script
CMD  ["/bin/bash"]
