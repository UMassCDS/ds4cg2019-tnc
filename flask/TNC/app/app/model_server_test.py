# from app import app
import tensorflow as tf
import numpy as np
import redis
import os, pathlib, uuid, ntpath, random, time, json
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

# Config parameters
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

IMAGE_QUEUE = "image_queue"
IMAGE_HEIGHT = 1920 * 0.75
IMAGE_WIDTH = 1920
IMAGE_DTYPE = "float32"

BATCH_SIZE = 1 # Only for when Tensorflow GPU is used

# Connecting to Redis server
db = redis.StrictRedis(host=REDIS_HOST, 
    port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

# Extract filename from path
def filename_extractor(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

# Load megadetector_v3.pb
def load_graph():
    sess = tf.Session()
    with tf.gfile.GFile(os.path.join(os.getcwd(), "megadetector_v3.pb"), 'rb') as tf_graph:
        graph_def = tf.GraphDef()
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
    sess, input_tensor, output_tensors = load_graph() 
    predictions = []

    # continuosly polling for data
    while True:
        # attempting to grab userIDs from Redis databases
        queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
        # UUID list to store and retrieve images
        imageIDs = [] 
        for q in queue:
            q = json.loads(q.decode("utf-8")) 
            # loop over the queue to get all images
            for img in queue["images"]:
                imageIDs.append(img)
            userID = queue["user_id"]
        
        # checking too see if there are images on redis to classify 
        if len(imageIDs) > 0:
            print("Running classifier")
            # Save predictions for all images as a list
            predictions = []
            for i in range(len(imageIDs)):
                # filepath = os.path.join(os.getcwd(), "app/static/uploaded_img/", imageIDs[i])

                # Can preprocess image during upload process but for now
                # let's make the model run the preprocessing
                image = Image.open(imageID[i]).convert("RGB")
                image = image.resize((IMAGE_WIDTH, int(IMAGE_HEIGHT)))
                image = np.expand_dims(image, axis=0)

                # Run model and obtain tensor scores
                boxes, scores, classes = sess.run(output_tensors, {input_tensor: image})

                # Check for passing criteria 
                above_threshold = scores > 0.35
                boxes_above_threshold = boxes[above_threshold]
                classes_above_threshold = classes[above_threshold]
                scores_above_threshold = scores[above_threshold]

                is_animal = np.sum(classes_above_threshold == 1) > 0
                max_score = np.max(scores_above_threshold)
                predictions.append([str(imageIDs[i]), "Yes", max_score]) # ([LABELS[int(is_animal)], max_score])
                # print(predictions)

                # Constructing bounding box on image file
                fig, ax = plt.subplots(1)
                ax.imshow(image.squeeze())
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
                plt.savefig(os.path.join("app/static/detected_img/", filename), dpi=100, transparent=True, optimize=True, quality=90)
                progressID = str(userID) + "progress"
                progress = str(i) + "of" + str(len(imageIDs)) + "completed."
                db.set(progressId, progress)

            # Storing completed images under userID
            db.set(userID, json.dumps(predictions))
            
            # Deleting userID from the processing queue
            db.ltrim(IMAGE_QUEUE, 1, -1)
        
        # Sleep for half a second before polling again
        time.sleep(0.5)

if __name__ == "__main__":
    tf_classify()
