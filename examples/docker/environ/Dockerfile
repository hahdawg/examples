# To build:
# >>> docker build -t <<image name>> -f Dockerfile ./
#
# To run:
# >>> docker run it --name <<container name>> --net=host \
# >>>   -v <<host mnt dir>>:<<container mnt dir>> <<image name>>
FROM ubuntu:18.04

# Don't prompt y for apt-get
RUN echo "APT::Get::Assume-Yes \"true\";" >> /etc/apt/apt.conf.d/90forceyes

# Install ubuntu stuff
RUN apt-get update && apt-get install -y \
    curl \
    htop \
    git \
    python3-dev \
    python-pip \
    python-setuptools \
    python-virtualenv \
    software-properties-common \
    sudo \
    tmux \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists

WORKDIR /ds
VOLUME /ds/mnt

# Copy local vim config
COPY .vimrc /ds/.vimrc
WORKDIR /ds/.vim
COPY .vim . /ds/.vim/

WORKDIR /ds/

# conda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
RUN bash Anaconda3-2020.11-Linux-x86_64.sh -b && \
    echo "export PATH="/root/anaconda3/bin:$PATH"" >> ~/.bashrc && \
    /bin/bash -c "source ~/.bashrc"
RUN rm Anaconda3-2020.11-Linux-x86_64.sh
ENV PATH /root/anaconda3/bin:$PATH
RUN conda install flake8 line_profiler pylint

# vim
RUN add-apt-repository -y ppa:jonathonf/vim && \
    apt update && \
    apt -y install vim && \
    curl -sL install-node.now.sh/lts | bash -s -- --yes

# Open ports for jupyter
EXPOSE 8888

# Setup file system
ENV SHELL=/bin/bash
ENV HOME /ds

# Run a shell script
CMD  ["/bin/bash"]
