FROM tensorflow/tensorflow:1.14.0-py3

# Dependencies for Pillow, the python image library
RUN apt-get -y update
RUN apt-get -y install wget git libjpeg8-dev zlib1g-dev --fix-missing

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY tnc_detector.py tnc_detector.py

RUN wget -nv -O megadetector_v4_1_0.pb https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb

# Set up MS CameraTraps
RUN git clone https://github.com/microsoft/CameraTraps
RUN cd CameraTraps
RUN git checkout d3123916bdd0304301dd57902f3f56b94bc6dfcc
ENV PATH /CameraTraps:${PATH}

ENTRYPOINT ["python3", "tnc_detector.py", "megadetector_v4_1_0.pb"]