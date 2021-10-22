"""This script runs the specified Megadetector model to detect animals on
a directory of image files and outputs three things:
- Excel file with metadata about the detections in the input images
- A folder containing only images where animals were detected with bounded boxes drawn on the images
- A folder containing all the images renamed to a unique naming schema based on image metadata (camera make and model, timestamp)
"""
import argparse
from datetime import datetime
import os
import time
import shutil
import warnings


import humanfriendly
from tqdm import tqdm
import exif
import pandas


from detection.run_tf_detector import ImagePathUtils,TFDetector
from detection import run_tf_detector_batch
from visualization import visualization_utils


warnings.filterwarnings("ignore", 'ASCII tag', module='exif')

# How to name things if metadata is missing for image
MISSING_DEFAULT = "UNKNOWN"

# Some key constants
FILE_KEY = "file"
ORIGINAL_FILE_KEY = "originalFile"
CONFIDENCE_KEY = "conf" # From Megadetector only
SCORE_KEY = "score" # Confidence column in output CSV
LABEL_KEY = "label" # Column for label in output CSV
MAKE_KEY = "make"
MODEL_KEY = "model"
TIMESTAMP_KEY = "timestamp"
DETECTIONS = "detections"


def prepare_output_directory(output_root, subfolder_name):
    """Creates subfolders within the output root directory to store output images.
    Returns paths to the created directorys: subfolder, subfolder to store annotated images, subfolder to store all renamed images, path to Excel metadata file.
    Args:
        - output_root: The full path to the output parent folder
        - subfolder_name
    """
    subfolder = os.path.join(output_root, subfolder_name)
    # Where metadata will live in that folder
    timestamp = datetime.now().isoformat(timespec='seconds')
    metadata_xlsx = os.path.join(subfolder, f'metadata_{timestamp}.xlsx')
    visualized_images = os.path.join(subfolder, 'annotated_images')
    all_images_out = os.path.join(subfolder, 'all_images')

    for d in [visualized_images, all_images_out]:
        if not os.path.exists(d) or not os.path.isdir(d):
            os.makedirs(d)

    return subfolder, visualized_images, all_images_out, metadata_xlsx


def run_prediction(model_path, image_files, confidence_threshold):
    """Runs detection predictions for images in batch mode and returns results in list of dictionaries.
    Args:
        - model_path: Path to Megadetector model
        - image_files: List of input image paths
        - confidence_threshold: double, confidence threshold setting for the model
    """
    start = time.time()
    prediction_results = run_tf_detector_batch.load_and_run_detector_batch(
                        model_file = model_path,
                        image_file_names = image_files,
                        confidence_threshold = confidence_threshold)

    elapsed = time.time() - start
    print('Finished inference in {}'.format(humanfriendly.format_timespan(elapsed)))
    return prediction_results


def verify_nonempty(input_str, default = MISSING_DEFAULT):
    """Return input_str or the default string if the input_str is empty
    """
    if input_str=="":
        return default
    return input_str


# TODO: Note that we could also extract more metadata here if desired.
# The original code seems to save off GPS info and other exif metadata
def pull_image_metadata_and_unique_file_name(original_file_path):
    """Returns a dictionary of image metadata (Camera make & model and image timestamp).
    Also generates a unique file name for the image based on datetime and camera metadata.
    The resulting image name is formated like MAKE_MODEL_DATETIME_ORIGINALNAME and has
    no spaces.
    """
    original_basename = os.path.basename(original_file_path)
    with open(original_file_path, 'rb') as img:
        image_info = exif.Image(img)
        make = verify_nonempty(image_info.get(MAKE_KEY, MISSING_DEFAULT).replace("_", "-").strip())
        model = verify_nonempty(image_info.get(MODEL_KEY, MISSING_DEFAULT).replace("_", "-").strip())

        date_string = image_info.get("datetime_original", MISSING_DEFAULT)
        parsed_datetime = datetime.strptime(date_string, '%Y:%m:%d %H:%M:%S')

    # This could be a bit annoying if the original filename has underscores in it, but
    # let's just go with this for now.
    # There shouldn't be any reason to map back to the original filename anyway.
    timestamp_str = parsed_datetime.isoformat()
    new_file_name = "_".join([make, model, timestamp_str, original_basename])
    new_file_name = new_file_name.replace(" ", "")
    return {MAKE_KEY: make, MODEL_KEY: model, FILE_KEY: new_file_name, TIMESTAMP_KEY: timestamp_str}


def expand_megadetector_prediction(prediction, metadata_dict=None):
    """Returns a list of results for an individual image's predictions, where
    numerical labels are replaced by category names and each detection is expanded
    to its own entry.
    Args:
        - prediction: A dictionary of detection predictions from megadetector for one file
        - metadata_dict: Optional, specify if you'd like to add additonal metadata
    """
    results = []
    original_file_name = prediction[FILE_KEY]
    detections = prediction[DETECTIONS]

    # This is what the original app code does if there are no detections above threshold
    # I'm not sure if it's the desired behaviour now
    if len(detections) == 0:
        r = {ORIGINAL_FILE_KEY: original_file_name,
            LABEL_KEY: "none",
            SCORE_KEY: 0.0}
        results.append(r)

    # Add a result to the list for each detection, the CSV could have multiple rows for
    # the same file if there are multiple animals
    for d in prediction[DETECTIONS]:
        label = TFDetector.DEFAULT_DETECTOR_LABEL_MAP[str(d['category'])]
        r = {ORIGINAL_FILE_KEY: original_file_name,
            LABEL_KEY: label,
            SCORE_KEY: d[CONFIDENCE_KEY]}
        results.append(r)
    if metadata_dict is not None:
        for r in results:
            r.update(metadata_dict)
    return results

def write_images_and_metadata(prediction_results, image_metadata, all_images_out, visualized_images, metadata_xlsx):
    """Writes annotated output images with bounding boxes and copies all images to an additional folder using a unique naming scheme.
    Also writes image metadata to xlsx.
    Args:
        - prediction_results: List of dicts containing prediction info from Megadetector
        - image_metadata: List of dicts containing metadata about the image, should be same length as prediction_results
        - all_images_out: path to directory for all uniquely renamed images
        - visualized_images: path to directory for images with bounding boxes
        - metadata_xlsx: Path to Excel file to write metadata to
    """
    # Some images might have multiple detections and we want each detection to be its own row in the Excel output
    expanded_predictions = []
    detection_image_count = 0
    print("Writing out image predictions and metadata.")
    for (p, m) in tqdm(zip(prediction_results, image_metadata)):
        expanded_metadata_prediction = expand_megadetector_prediction(p, m)
        expanded_predictions.extend(expanded_metadata_prediction)
        original_path = expanded_metadata_prediction[0][ORIGINAL_FILE_KEY]
        unique_name = expanded_metadata_prediction[0][FILE_KEY]

        # TODO Clarify with Laura - is it more useful to have bounding boxes in all_images_dir or just a uniquely named raw image
        # Make a uniquely named copy of each image
        # copy2 tries to preserve metadata, which seems desirable
        shutil.copy2(original_path, os.path.join(all_images_out, unique_name))

        # Visualize and save bounding box images
        if len(p[DETECTIONS]) > 0:
            detection_image_count+=1
            bounding_box_file = os.path.join(visualized_images, unique_name)
            visualization_utils.draw_bounding_boxes_on_file(original_path, bounding_box_file, p[DETECTIONS])

    print()
    # Use pandas to write the output to Excel without a fuss
    metadata_dataframe = pandas.DataFrame(expanded_predictions)
    metadata_dataframe.to_excel(metadata_xlsx, index=False)
    print("Metadata written to", metadata_xlsx)
    sanity_check_outputs_images(prediction_results, all_images_out, visualized_images, detection_image_count)


def sanity_check_outputs_images(prediction_results, all_images_out, visualized_images, detection_image_count):
    """Validates that the number of images written matches the number of predictions and detections.
    Raises an Assertion error if something went wrong.
    Args:
        - prediction_results: List of dicts containing prediction info from Megadetector
        - image_metadata: List of dicts containing metadata about the image, should be same length as prediction_results
        - all_images_out: path to directory for all uniquely renamed images
        - visualized_images: path to directory for images with bounding boxes
        - detection_image_count, int, number of images that had positive detections
    """
    # Sanity check number of outputs
    num_raw_images_out = len(os.listdir(all_images_out))
    num_visualized_images = len(os.listdir(visualized_images))
    print(num_raw_images_out, "image(s) copied to", all_images_out)
    print(num_visualized_images, "image(s) with bounding boxes written to", visualized_images)
    assert len(prediction_results) == num_raw_images_out, "Different number of input and output images!"
    assert detection_image_count == num_visualized_images, "Not all image detections written out!"


def main(model_path, input_dir, output_dir):
    """Script entry point
    Args:
        - model_path: Path to the Megadetector model to use for detection
        - input_dir: Path to a folder containing image files to run detection on
        - output_dir: Path to the directory where metadata and visualized images will be written
    """
    _, visualized_images_out, all_images_out, metadata_xlsx = prepare_output_directory(output_dir, os.path.basename(input_dir))
    input_file_names = ImagePathUtils.find_images(input_dir, recursive=True)
    confidence_threshold = TFDetector.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD
    print("Model to be used for detection:", model_path)
    print("Confidence threshold to be used for detection:", confidence_threshold)
    print(len(input_file_names), "image(s) will be read in from Google Drive folder:", input_dir)
    print("The results folder is:", output_dir)
    print("Detection metadata will be output to:", metadata_xlsx)
    print("Visualized annotated data will be output to:", visualized_images_out)
    print("Copies of all images will be uniquely renamed and placed in:", all_images_out)

    prediction_results = run_prediction(model_path, input_file_names, confidence_threshold)
    image_metadata = [pull_image_metadata_and_unique_file_name(p['file']) for p in prediction_results]
    write_images_and_metadata(prediction_results, image_metadata, all_images_out, visualized_images_out, metadata_xlsx)


parser = argparse.ArgumentParser(description="Nature Conservancy Image Detector")
parser.add_argument("model", help="The path to the Megadetector model to use for detection")
parser.add_argument("input_dir", help="The path to a folder containing image files to run detection on")
parser.add_argument("output_dir", help="The desired root directory where metadata and visualized images will be written. A new folder will be created here with the same name as input_dir.")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.model, args.input_dir, args.output_dir)