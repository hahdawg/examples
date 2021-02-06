# In container, core.py reads mounted data dir, makes predictions, and stores them in
# mounted data dir.
# 
# build: 
# >>> docker build -t ml-demo .
# 
# run: 
# >>> docker run it --name <<container name>> --net=host \
# >>>   -v <<absolute host mnt dir>>:<<container mnt dir>> ml-demo

FROM python:3.8
WORKDIR /ds
COPY core.py .
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /ds/data
VOLUME /ds/data

WORKDIR /ds

CMD  ["/bin/bash"]
