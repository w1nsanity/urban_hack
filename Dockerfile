FROM ubuntu:22.04

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y \
        build-essential git python3 python3-pip wget \
        ffmpeg libsm6 libxext6 libxrender1 libglib2.0-0

COPY requirements.txt .

RUN pip3 install -U pip
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY /model/ ./model/
COPY /images/ ./images/
COPY inference.py ./inference.py
RUN mkdir /output/

CMD /bin/sh -c "python3 inference.py"