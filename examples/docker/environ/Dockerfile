FROM ubuntu:18.04

ARG hm=/home/andrewhah

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
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
RUN rm Miniconda3-latest-Linux-x86_64.sh
RUN echo "PATH=/opt/conda/bin:$PATH" >> /etc/profile
ENV PATH /opt/conda/bin:$PATH
RUN conda install -y python=3.8
RUN conda config --add channels conda-forge
RUN conda config --set channel_priority strict
RUN conda install -y -c conda-forge mamba
RUN mamba install -y flake8 matplotlib jupyter ipython line_profiler pylint pytest pandas
RUN mamba install -y -c conda-forge gcsfs google-api-python-client google-cloud-bigquery gpustat \
    ray-all ray-core ray-dashboard ray-default ray-tune scikit-learn seaborn tqdm jedi

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
WORKDIR ${hm}
COPY .vimrc ${hm}/.vimrc
RUN curl -fLo ${hm}/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
# RUN /usr/bin/vim -E -s -u "${hm}/.vimrc" +PlugInstall +qall

RUN echo '{"python.pythonPath": "/opt/conda/bin/python"}' >> ${hm}/.vim/coc-settings.json

# Open ports for jupyter
EXPOSE 8888

# Setup file system
VOLUME ${hm}/mnt
ENV SHELL=/bin/bash
ENV HOME ${hm}

# Run a shell script
CMD  ["/bin/bash"]
