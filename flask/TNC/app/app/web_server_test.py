from app import app
from flask import render_template, request, make_response, session, url_for, redirect
from flask import safe_join, send_from_directory, abort, flash, copy_current_request_context
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_executor import Executor
import redis
import os, uuid, pathlib, time, json, ntpath, random



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

# Config parameters
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

IMAGE_QUEUE = "image_queue"
IMAGE_HEIGHT = 1920 * 0.75
IMAGE_WIDTH = 1920
IMAGE_DTYPE = "float32"

BATCH_SIZE = 64 # Only for when Tensorflow GPU is used

# Connecting to redis
db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, 
    db=REDIS_DB, decode_responses=True)

# Home page. Pretty much static
@app.route("/")
def index():
    return render_template("index.html")

# Upload page. Contains dropzone code. Saves images to uploaded_img directory
@app.route("/upload", methods=['GET', 'POST'])
def upload():

    # Note: Session may be avoided altogether but for now, leave as is
    # Generate user ID for user to track progress and retrieve download link
    # Set session for userID
    if "userID" not in session:
        session['userID'] = str(uuid.uuid4())
        session['images'] = []
    # List to hold our uploaded image IDs
    imageIDs = session['images']

    # If images are uploaded, save to upload directory and store in session
    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            filename = photos.save(file, name=str(uuid.uuid4()) + ".")

            # Note: Can save path or just filename. Let's try filename
            # and build with that logic
            imageIDs.append(photos.path(filename))
            session['images'].append(photos.path(filename))
        userID = session['userID']
        # Flashing userID to user so that they can save it
        flash("PLEASE SAVE:" + userID)
        # Store userID and imageIDs as a dictionary and push to redis
        d = {"user_id": userID, "images": imageIDs, "progress": 0}
        db.rpush(IMAGE_QUEUE, json.dumps(d))
        return "Uploading to disk and Redis"
    return render_template("upload.html") 

@app.route("/progress")
def results():
    # Check if userID in session. If not, redirect to upload page.
    if userID not in session:
        return redirect(url_for('upload'))
    # Get userID from session
    userID = session
    return "Well something is happening."