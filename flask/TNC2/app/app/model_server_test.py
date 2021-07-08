import numpy as np
import pandas
import redis
import os, pathlib, uuid, ntpath, random, time, json, sys
import csv
from zipfile import ZipFile
from PIL import Image
from PIL.ExifTags import TAGS
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # remove if using GPU

import tensorflow as tf

# Config parameters
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

IMAGE_HEIGHT = 1920 * 0.75
IMAGE_WIDTH = 1920
IMAGE_DTYPE = "float32"
UPLOAD_PATH = os.path.join(os.getcwd(), "static/uploaded_img/")
DOWNLOAD_PATH = os.path.join(os.getcwd(), "static/detected_img/")

USER_QUEUE = "user_queue"

BATCH_SIZE = 4 # Change only when Tensorflow GPU is used


# Connecting to Redis server
db = redis.StrictRedis(host=REDIS_HOST, 
    port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

# Extract filename from path
def filename_extractor(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# To make hyperlinks to file system paths
def clickable(path):
    f_url = os.path.basename(path)
    return '=HYPERLINK("{}", "{}")'.format(path, f_url)

# Load megadetector_v3.pb
def load_graph():
    sess = tf.compat.v1.Session()
    with tf.io.gfile.GFile(os.path.join(os.getcwd(), "megadetector_v3.pb"), 'rb') as tf_graph:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(tf_graph.read())
        tf.import_graph_def(graph_def, name='')
    # label_lines = [line.rstrip() for line in tf.gfile.GFile(TF_LABELS)]

    input_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
    box_tensor = sess.graph.get_tensor_by_name('detection_boxes:0')
    score_tensor = sess.graph.get_tensor_by_name('detection_scores:0')
    class_tensor = sess.graph.get_tensor_by_name('detection_classes:0')
    output_tensors = [box_tensor, score_tensor, class_tensor]
    return sess, input_tensor, output_tensors # , label_lines

# Function to load model and continuously poll for new images to
# classify from redis as and when available
def tf_classify():
    # loading model
    start = time.time()
    sess, input_tensor, output_tensors = load_graph() 
    end = time.time()
    print("Model loading completed in " + str(end - start))

    # continuosly polling for data
    while True:
        # attempting to grab any user request from Redis databases
        user = db.lrange(USER_QUEUE, 0, 0)
        finalnames = []
        if user:
            # If exists, get ID and email
            userID = json.loads(user[0])["userID"]
            email = json.loads(user[0])["email"]
            queue = db.lrange(userID, 0, BATCH_SIZE - 1)
            predictions = []
            while queue:
                images = []
                imagenames = []
                imtags = []
                for img in queue:
                    # loop over the queue to get all preprocessed images
                    img = json.loads(img)["image"]
                    imagenames.append(img)
                    image = Image.open(os.path.join(UPLOAD_PATH, img))
                    imtags.append(image._getexif())
                    image = image.convert("RGB")
                    image = image.resize((IMAGE_WIDTH, int(IMAGE_HEIGHT)))
                    # image = np.expand_dims(image, axis=0) 
                    images.append(np.array(image)) # Appending instead of expanding since we want batch processing
        
                # checking to see if we should run classifer (redundant)
                if len(images) > 0:
                    print("Running classifier for ", len(images))
                    # Save predictions for all images as a list
                    

                    # Run model and obtain tensor scores on batch of images
                    images = np.asarray(images)
                    start = time.time()
                    boxes, scores, classes = sess.run(output_tensors, {input_tensor: images})
                    end = time.time()
                    print("Model ran in " + str(end - start))
                    # Check each image for passing criteria and construct bounding box if need be
                    for i in range(len(images)):
                        start = time.time()
                        above_threshold = scores[i] > 0.5
                        boxes_above_threshold = boxes[i][above_threshold]
                        classes_above_threshold = classes[i][above_threshold]
                        scores_above_threshold = scores[i][above_threshold]
                        if not np.any(scores_above_threshold):
                            pred = {"image": imagenames[i], "label":"None", "score":"0"}
                            predictions.append(pred)
                        else:
                            is_animal = np.sum(classes_above_threshold == 1) > 0
                            max_score = np.max(scores_above_threshold)
                            label = ["None", "Animal/Human"]
                            pred = {"image":imagenames[i], "label": label[is_animal], "score": str(max_score)}
                            predictions.append(pred)
                        print(predictions)
                        # Adding predictions to userID_PRED queue
                        db.rpush(userID + '_PRED', json.dumps(pred))

                        # Constructing bounding box on image file
                        fig, ax = plt.subplots(1)
                        ax.imshow(images[i].squeeze())
                        if np.any(boxes_above_threshold):
                            # Add imagename to finalnames
                            finalnames.append(imagenames[i])
                            for box, cls in zip(boxes_above_threshold, classes_above_threshold):
                                ymin, xmin, ymax, xmax = box
                                x = xmin * IMAGE_WIDTH
                                y = ymin * IMAGE_HEIGHT
                                w = (xmax - xmin) * IMAGE_WIDTH
                                h = (ymax - ymin) * IMAGE_HEIGHT
                                edgecolor = 'g' if cls == 1 else 'r'
                                rect = patches.Rectangle((x, y), w, h, linewidth=5,
                                                        edgecolor=edgecolor, facecolor='none')
                                ax.add_patch(rect)

                            # Saving file to location on disk
                            plt.axis('off')
                            if imtags[i]:
                                plt.savefig(os.path.join("static/detected_img/", imagenames[i]), pil_kwargs={"exif":imtags[i]}, dpi=100, transparent=True, optimize=True, quality=90)
                            else:
                                plt.savefig(os.path.join("static/detected_img/", imagenames[i]), dpi=100, transparent=True, optimize=True, quality=90)
                        end = time.time()
                        print("Image saved in " + str(end-start))
                    
                    # Remove processed images from user queue
                    db.ltrim(userID, BATCH_SIZE, -1)
                # Update queue for next round
                queue = db.lrange(userID, 0, BATCH_SIZE - 1)

            # Make a dataframe of predictions and convert to excel sheet
            df = pandas.DataFrame(predictions)
            print(df)
            df['image'] = df['image'].apply(lambda x: clickable(x))
            df.to_excel(os.path.join(DOWNLOAD_PATH, userID + ".xlsx"), encoding='utf-8', index=False)

            # with open(DOWNLOAD_PATH + userID + ".csv", "w+") as my_csv:
            #     csvWriter = csv.writer(my_csv, delimiter=',')
            #     csvWriter.writerows(predictions)

            # Write output to zipfile while removing original file
            with ZipFile(DOWNLOAD_PATH + userID + '.zip', 'w') as zipObj:
                for img in finalnames:
                    zipObj.write(os.path.join(DOWNLOAD_PATH, img), img)
                    os.remove(os.path.join(DOWNLOAD_PATH, img))
                zipObj.write(os.path.join(DOWNLOAD_PATH, userID + ".xlsx"), userID + ".xlsx")

            # Remove csv file
            os.remove(os.path.join(DOWNLOAD_PATH, userID + ".xlsx"))

            # Notify user of completion
            msg = MIMEMultipart()
            msg['From'] = "foxboundtnc@gmail.com"
            msg['To'] = email
            msg['Subject'] = "Download Link"
            message = '''Your images have been processed. Please download the predictions by visiting the website https://127.0.0.1:5000/results/{}'''.format(userID)
            msg.attach(MIMEText(message, 'plain'))
            s = smtplib.SMTP('smtp.gmail.com', 587)
            s.ehlo()
            s.starttls()
            s.login("foxboundtnc@gmail.com", "ASnowyDayInAmherst")
            s.sendmail(msg['From'], email, msg.as_string())
            s.quit()

            # Remove userID from user queue and add to COMPLETE_QUEUE
            db.ltrim(USER_QUEUE, 1, -1)

                # Updating progress
                # Generate key for progress linked to userID
                # progressID = userIDs[i] + "progress"
                # # Calculate percentage value
                # progress = []
                # percent = i/len(images) * 100.00
                # progress.append(percent)
                # msg = str(i) + "of" + str(len(images)) + "completed."
                # progress.append(msg)
                # db.set(progressID, percent)

            # Storing completed images under userID
            # db.set(userID, json.dumps(predictions))
        
        # Sleep for a second before polling again
        time.sleep(1)

if __name__ == "__main__":
    tf_classify()
