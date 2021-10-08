from flask import Flask, render_template


app = Flask(__name__)

@app.route("/")
def landing():
	#if logged in:
	#	send the main interface
	#if not:
	#	send the login interface

	return "it works"


#ajax path - runs on user interactions
#def login
#	recieve login attempt
#		if successfull, register login and redirect browser
#	else
#		send error back.


#ajax path
#def upload - runs only on user interaction
#	validate login
#	validate the upload.
#	throw the images in S3
# 	throw the upload record into ddb
#	possibly, send whatever api call is needed to wake up the model server.

#ajax path - runs when user loads the main interface and on a cycle after that
#def poll_db
#	validate login
#	fetch all the records for this user


#def download_results
#   validate login
#	grabs the files from the given s3 location- either the images themselves or the result csv