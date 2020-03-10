from app import app
from flask import render_template, request, make_response, session, url_for, redirect
from flask import safe_join, send_from_directory, abort, flash, copy_current_request_context
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_executor import Executor
import os, pathlib, uuid
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import ntpath
import random

import threading
import time


executor = Executor(app=app)
model_threads = {}
thread = None
# Dropzone Settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_MAX_FILE_SIZE'] = 100 * 1024 * 1024
app.config['DROPZONE_REDIRECT_VIEW'] = 'results'
app.config['SECRET_KEY'] = 'SnowyDayInAmherst'
app.config['EXECUTOR_TYPE'] = 'thread'
app.config['EXECUTOR_MAX_WORKERS'] = 20

app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(os.getcwd(), "app/static/uploaded_img/")
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)

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

sess, input_tensor, output_tensors = load_graph()

def filename_extractor(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def tf_classify(image_file):
    width, height = 1920, 1920 * 0.75
    predictions = []

    image = Image.open(image_file).convert("RGB")
    image = image.resize((width, int(height)))
    image = np.expand_dims(image, axis=0)

    boxes, scores, classes = sess.run(output_tensors, {input_tensor: image})

    above_threshold = scores > 0.35
    boxes_above_threshold = boxes[above_threshold]
    classes_above_threshold = classes[above_threshold]
    scores_above_threshold = scores[above_threshold]

    is_animal = np.sum(classes_above_threshold == 1) > 0
    max_score = np.max(scores_above_threshold)
    predictions.append(["Yes", max_score]) # ([LABELS[int(is_animal)], max_score])
    # print(predictions)
    fig, ax = plt.subplots(1)
    ax.imshow(image.squeeze())
    for box, cls in zip(boxes_above_threshold, classes_above_threshold):
        ymin, xmin, ymax, xmax = box
        x = xmin * width
        y = ymin * height
        w = (xmax - xmin) * width
        h = (ymax - ymin) * height
        edgecolor = 'g' if cls == 1 else 'r'
        rect = patches.Rectangle((x, y), w, h, linewidth=5,
                                 edgecolor=edgecolor, facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    filename = filename_extractor(image_file)
    # filepath = os.path.join(os.getcwd(), "app/static/detected_img/", filename)
    # print(filepath)
    # print(filename)
    # print(os.path.join("app/static/detected_img/", filename))
    plt.savefig(os.path.join("app/static/detected_img/", filename), dpi=100, transparent=True, optimize=True, quality=90)
    return predictions

def process_images(image_urls):
    for image in image_urls:
        print(session["image_urls"])
        fname = filename_extractor(image)
        fpath = os.path.join(os.getcwd(), "app/static/detected_img/", fname)
        # print(fpath)
        pred = tf_classify(image)
        pred.append(fpath)
        session["detected_urls"].append(pred)
    
    return pred

class ModelThread(threading.Thread):
    def __init__(self, image_urls):
        self.image_urls = image_urls
        self.progress = 0
        self.total = len(image_urls)
        super().__init__()
    
    def run(self):
        for image in self.image_urls:
            fname = filename_extractor(image)
            fpath = os.path.join(os.getcwd(), "app/static/detected_img/", fname)
            pred = tf_classify(image)
            pred.append(fpath)
            session["detected_urls"].append(pred)
            time.sleep(1)
            self.progress += 1

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=['GET', 'POST'])
def upload():

    # set session for image uploads
    if "image_urls" not in session:
        session['image_urls'] = []
    # list to hold our uploaded image urls
    image_urls = session['image_urls']



    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            filename = photos.save(file, name=str(uuid.uuid4()) + ".")
            print(filename)
            image_urls.append(photos.path(filename))
            # print(photos.path(filename))
        session['image_urls'] = image_urls
        # print(image_urls)
        return "uploading..."
    return render_template("upload.html")

@app.route("/results")
def results():

    @copy_current_request_context
    def spawn_threads(image_urls):
        for image in image_urls:
            print(session["image_urls"])
            fname = filename_extractor(image)
            fpath = os.path.join(os.getcwd(), "app/static/detected_img/", fname)
            # print(fpath)
            pred = tf_classify(image)
            pred.append(fpath)
            session["detected_urls"].append(pred)

    # redirect to home if no images to display
    if "image_urls" not in session or session['image_urls'] == []:
        return redirect(url_for('upload'))
        
    # set the file_urls and remove the session variable
    image_urls = session['image_urls']
    # session.pop('image_urls', None) ## TO ADD LATER FOR SURE

    # set session for image results
    if "detected_urls" not in session:
        session['detected_urls'] = []
    # list to hold our detected image urls
    # predictions = session['detected_urls']

    flash("Process has begun")
    ################
    global thread
    global model_threads

    if "thread_id" not in session:
        session["thread_id"] = str(uuid.uuid4())
        thread = ModelThread(target=spawn_threads, args=(image_urls))
        thread.start()
        return thread.is_alive()

    return thread.is_alive()


    ################


    # session["thread_id"] = str(uuid.uuid4())
    # futures = executor.submit_stored(session["thread_id"], process_images, image_urls)
    # if not executor.futures.done(session["thread_id"]):
    #     future_status = executor.futures._state(session["thread_id"])
    #     return future_status


    # flash("Save job id to track progress: ")

    # for image in image_urls:
        
    #     fname = filename_extractor(image)
    #     fpath = os.path.join(os.getcwd(), "app/static/detected_img/", fname)
    #     print(fpath)
    #     pred = tf_classify(image)
    #     pred.append(fpath)
    
    
        
    
    return render_template('results.html') #, image_urls=image_urls)