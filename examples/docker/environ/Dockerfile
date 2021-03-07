# To build:
#>>> docker build -t <<image name>> -f Dockerfile ./
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
    software-properties-common \
    sudo \
    tmux \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists

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

# yarn
RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - && \
    echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list && \
    apt update && \
    apt install -y yarn && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists

# Copy local vim config
COPY .vimrc /ds/.vimrc
RUN curl -fLo /ds/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

WORKDIR /ds/
RUN echo '{"python.pythonPath": "/root/anaconda3/bin/python"}' >> /ds/.vim/coc-settings.json

# Open ports for jupyter
EXPOSE 8888

# Setup file system
VOLUME /ds/mnt
ENV SHELL=/bin/bash
ENV HOME /ds

# Run a shell script
CMD  ["/bin/bash"]
