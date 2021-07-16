from app import app
from flask import render_template, request, make_response, session, url_for, redirect, Response
from flask import safe_join, send_file, send_from_directory, abort, flash, copy_current_request_context, stream_with_context
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, ARCHIVES, patch_request_class
from flask_executor import Executor
from redis import *
from threading import Thread
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os, uuid, pathlib, time, json, ntpath, random, sys
import zipfile

# Dropzone Settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*, application/zip, .zip, application/x-zip-compressed, multipart/x-zip'
app.config['DROPZONE_MAX_FILE_SIZE'] = 10240 * 1024 * 1024 # 10GB for zip
app.config['DROPZONE_PARALLEL_UPLOADS'] = 10000
app.config['SECRET_KEY'] = 'SnowyDayInAmherst'
app.config['EXECUTOR_TYPE'] = 'thread'
app.config['EXECUTOR_MAX_WORKERS'] = 20
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(os.getcwd(), "app/static/uploaded_img/")
app.config['UPLOADS_DEFAULT_DEST'] = os.path.join(os.getcwd(), "app/static/uploaded_img/")
app.config['UPLOAD_PATH'] = os.path.join(os.getcwd(), "app/static/uploaded_img/")
photos = UploadSet('photos', IMAGES)
zips = UploadSet('zips', ARCHIVES)
configure_uploads(app, photos)
# patch_request_class(app) # to limit upload size. default 16MB

# Config parameters
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

IMAGE_HEIGHT = 1920 * 0.75
IMAGE_WIDTH = 1920
IMAGE_DTYPE = "float32"

USER_QUEUE = "user_queue"
USER_TOTAL = "user_total"
USER_REMAINING = ""
UPLOAD_PATH = os.path.join(os.getcwd(), "app/static/uploaded_img/")
DOWNLOAD_PATH = os.path.join(os.getcwd(), "app/static/detected_img/")

def mail(userID, email):
    # Mail user saying that the images have been processed
    msg = MIMEMultipart()
    msg['From'] = "foxboundtnc@gmail.com"
    msg['To'] = email
    msg['Subject'] = "Submitted for Processing"
    message = '''Your images have been submitted for processing. Please track and download the predictions by visiting the website https://127.0.0.1:5000/results/{}'''.format(userID)
    msg.attach(MIMEText(message, 'plain'))
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.ehlo()
    s.starttls()
    s.login("foxboundtnc@gmail.com", "ASnowyDayInAmherst")
    s.sendmail(msg['From'], email, msg.as_string())
    s.quit()

# Connecting to redis
db = StrictRedis(host=REDIS_HOST, port=REDIS_PORT, 
    db=REDIS_DB, decode_responses=True)
# print(db.ping(), file=sys.stderr)

# Home page. Pretty much static
@app.route("/")
def index():
    return render_template("index.html")

# Upload page. Contains dropzone code. Saves images to uploaded_img directory
@app.route("/upload")
def upload():
    # Note: Session may be avoided altogether but for now, leave as is
    # Generate user ID for user to track progress and retrieve download link
    # Set session for userID
    # print("before userid and sess")
    
    session.clear()
    # print(session)
    if "userID" not in session:
        session.clear() # clearing flash messages from previous run if any
        session['userID'] = str(uuid.uuid4())
        session['images'] = False
    return render_template("upload.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_images():
    # If images are uploaded, save to upload directory and push to user session in redis
    if request.method == 'POST':
        for key, f in request.files.items():
            # print(key, f, file=sys.stderr)
            if key.startswith('file'):
                print(f.filename)
                if f.filename.rsplit(".", 1)[1] == "zip":
                    with zipfile.ZipFile(f) as z:
                        # Checking for jpg only to extract and push to redis
                        for name in z.namelist():
                            if name.rsplit(".", 1)[1].lower() == "jpg":
                                print(name)
                                session["images"] = True
                                try:
                                    z.extract(name, os.path.join(UPLOAD_PATH))
                                    d = {"image":name}
                                    db.rpush(session["userID"], json.dumps(d))
                                except:
                                    pass # todo: return relevant info back to user maybe
                else:
                    session["images"] = True
                    filename = photos.save(f) # Ensure strong naming convention
                    # filename = photos.save(f, name=str(uuid.uuid4()) + ".")
                    d = {"image":filename}
                    db.rpush(session["userID"], json.dumps(d))
    return render_template("upload.html")

@app.route("/process_redis", methods=['GET', 'POST'])
def process_redis():
    # Check if any images have been submitted
    if session["images"]:
        userID = session["userID"]
        email = request.form['email']

        thread = Thread(target = mail, args = (userID, email, ))
        thread.start()

        # Flashing download link to user
        flash("/results/" + userID)
        # Push user request to USER_QUEUE to be polled by model server
        d = {"userID": userID, "email": email}
        db.rpush(USER_QUEUE, json.dumps(d))
    else:
        flash("Please upload images before starting!")
    # Clear session for next run
    [session.pop(key) for key in list(session.keys()) if key != '_flashes']
    return render_template("process_redis.html")

@app.route("/results/<userID>")
def result(userID):
    # Check if download file present in local system. If not send to tracking page
    # print("yup")
    if os.path.isfile(os.path.join(DOWNLOAD_PATH, userID + ".zip")):
        return send_from_directory(DOWNLOAD_PATH, filename=userID + ".zip")
    else:
        session["tempID"] = userID
        users = db.lrange(USER_QUEUE, 0, -1)
        print(users)
        cu, ci = 0, 0
        for user in users:
            user = json.loads(user)["userID"]
            ci += len(db.lrange(user, 0, -1))
            print(ci)
            if user == userID:
                et = ci * 5
                return render_template("results.html", users = cu, images=ci, et=et)
                break
            cu += 1
        # abort(404)
        return Response("Invalid request. Your link expired or your submission wasn't accepted. Try again!", 500);

# Remove comment section if you wanna add progress bar instead of countdown timer
# Progress is continiously streamed to client so it can get data heavy if there are many users. Unlikely to happen with FoxBound
# @app.route('/progress')
# def progress():
#     def generate():
#         user = session["tempID"]
#         ti = len(db.lrange(user, 0, -1))
#         ci = ti
#         x = 0.0
        
#         while ci != 0:
#             et, cu, tti = 0, 0, 0
#             users = db.lrange(USER_QUEUE, 0, -1)
#             for u in users:
#                 u = json.loads(u)["userID"]
#                 tti += len(db.lrange(u, 0, -1))
#                 print(ci)
#                 if user == u:
#                     et = tti * 10 / 60

#             ci = len(db.lrange(user, 0, -1))
#             print(ci)
#             try:
#                 prog = (ti - ci) / ti * 100
#             except:
#                 prog = 0
#             ret = "data:{\nprog:" +str(prog) + ",\nusers:" + str(cu) + ",\nimages:" +str(images) + ",\net:" + str(et) + "}\n\n"
#             yield ret

#             time.sleep(10)
#     return Response(stream_with_context(generate()), mimetype='text/event-stream')
