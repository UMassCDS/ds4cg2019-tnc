FROM tensorflow/tensorflow:1.14.0-py3

# TODO Handle mounted file permissions appropriately by creating a user, see https://vsupalov.com/docker-shared-permissions/

# Install fetch tools and dependencies for Pillow, the python image library
RUN apt-get -y update
RUN apt-get -y install wget git libjpeg8-dev zlib1g-dev libfreetype6-dev --fix-missing

# Set up MS CameraTraps and make it the working directory
RUN git clone https://github.com/microsoft/CameraTraps
RUN cd CameraTraps
WORKDIR /CameraTraps
RUN git checkout d3123916bdd0304301dd57902f3f56b94bc6dfcc

RUN wget -nv -O megadetector_v4_1_0.pb https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb

# Copy over our script and set as the container's entry point
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY tnc_detector.py tnc_detector.py

ENTRYPOINT ["python3", "tnc_detector.py", "megadetector_v4_1_0.pb"]