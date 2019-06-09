#!/bin/bash

ROOT=$(pwd)
DATA_PATH=$ROOT/data/wildcam

download() {
    wget https://storage.googleapis.com/iwildcam_2018_us/$1
    tar xvzf $1
    rm $1
    echo "Downloading $1 is completed"
}

# download iWildCam2018 dataset
while test $# -gt 0; do
    case "$1" in
        --download_path)
            shift
            if test $# -gt 0; then
                DATA_PATH="$1"
            else
                echo "Specify a path to download iWildCam2018 dataset"
                exit 0
            fi
            shift
            ;;
        *)
            ;;
    esac
done


echo "Downloading iWildCam2018 dataset to $DATA_PATH..."
mkdir -p $DATA_PATH
cd $DATA_PATH

download train_val.tar.gz
download test.tar.gz
download iwildcam2018_annotations.tar.gz

