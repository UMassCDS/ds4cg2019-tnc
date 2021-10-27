from flask import Flask, render_template, url_for, request
import json, uuid, time, hashids
import boto3

settings = json.load(open("app_settings.json"))
ddb_client = boto3.client("dynamodb", "us-east-2")
s3_client = boto3.client("s3", "us-east-2")

app = Flask(__name__)

@app.route("/")
def main():
	return render_template("main.html", 
		style_link = url_for("static", filename="style.css"),
		js_link = url_for("static", filename="main.js")
	)

#an ajax route
@app.route("/poll_ddb")
def poll_ddb():
	
	response = ddb_client.query(
		TableName = settings["JOB_TABLE"],
		IndexName = settings["USR_INDEX"],
		Select = "ALL_ATTRIBUTES",
		KeyConditionExpression = "user_id = :user_id_val",
		ExpressionAttributeValues = {":user_id_val": {"S": "test_id"}}
	)
	
	#just get all the documents currently in the ddb and return them raw.
	#hypothetically this will query by user_id once that's implemented

	return {"items": response["Items"]}

#ajax route
@app.route("/get_s3_upload_url")
def get_s3_upload_url():
	fname = request.args.get("filename").split(".")
	hasher = hashids.Hashids()
	timehash = hasher.encode(int(time.time()*10))
	#s3 key is the original file name plus a hash of the current time.
	#just to prevent collisions. 
	full_key = f'{fname[0]}-{timehash}.{fname[1]}'
	post_url = s3_client.generate_presigned_url("put_object",
			{"Bucket":settings["S3_BUCKET"], "Key":full_key})
	return {"uploadURL":post_url}

@app.route("/put_job_record_ddb")
def put_job_record_ddb():
	location = request.args.get("location")
	item = {
		'job_id':{"S": uuid.uuid1().hex},
		'user_id':{"S":"test_id"},
		'upload_location':{"S":location},
		'timestamp':{"N":f'{time.time()}'}
	}
	ddb_client.put_item(
		TableName = settings["JOB_TABLE"],
		Item = item
	)
	return {"Status":"OK"}

#ajax path - runs on user interactions
#def login
#	recieve login attempt
#		if successfull, register login and redirect browser
#	else
#		send error back.



if __name__ == "__main__":
	app.run()	