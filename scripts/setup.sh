#!/bin/bash

# install dependencies
pip install --user -r requirements.txt

# download Northm America Camera Trap Image (NACTI) dataset
while test $# -gt 0; do
	case "$1" in 
		-d|--download)
			shift
			if test $# -gt 0; then
				download="$1"
			else
				echo "specify whether you want to download NACTI dataset (true/false)"
				exit 0
			fi
			shift
			;;
		*)
			download=false
			;;
	esac
done

if [ "$download" == true ]; then
	echo "Downloading NACTI dataset..."

	ROOT=$(pwd)
	mkdir -p $ROOT/data/nacti
	cd $ROOT/data/nacti

	wget https://lilablobssc.blob.core.windows.net/nacti/nactiPart0.zip
	wget https://lilablobssc.blob.core.windows.net/nacti/nactiPart1.zip
	wget https://lilablobssc.blob.core.windows.net/nacti/nactiPart2.zip
	wget https://lilablobssc.blob.core.windows.net/nacti/nactiPart3.zip
	wget https://lilablobssc.blob.core.windows.net/nacti/nacti_metadata.json.zip

	unzip nactiPart0.zip
	unzip nactiPart1.zip
	unzip nactiPart2.zip
	unzip nactiPart3.zip
	unzip nacti_metadata.json.zip

	rm nactiPart0.zip
	rm nactiPart1.zip
	rm nactiPart2.zip
	rm nactiPart3.zip
	rm nacti_metadata.json.zip
fi