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

# RUN apt-get update && apt-get install -y \
#     python3-dev \
#     python-pip \
#     python-setuptools && \
#     rm -rf /var/lib/apt/lists

WORKDIR /ds
COPY core.py /ds/
COPY requirements.txt /ds/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN mkdir /ds/data
VOLUME /ds/data

CMD ["python", "core.py"]
