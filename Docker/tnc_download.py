import json, time, zipfile, os, shutil
import boto3

ZIP_PATH = "./zips"
TASK_PATH = "./tasks"

settings = json.load(open("app_settings.json"))
ddb_client = boto3.client("dynamodb", "us-east-2")
s3_client = boto3.client("s3", "us-east-2")


def reset_env():
	if os.path.isdir(ZIP_PATH):
		shutil.rmtree(ZIP_PATH)

	if os.path.isdir(TASK_PATH):
		shutil.rmtree(TASK_PATH)


#Queries the dynamodb for jobs which are in step 0 (ie, just uploaded, this might change)
#Downloads them to the current filesystem
#and unzips them
def get_queued_jobs():
	queued_jobs = ddb_client.query(
			TableName = settings["JOB_TABLE"],
			IndexName = settings["STEP_INDEX"],
			Select = "ALL_ATTRIBUTES",
			KeyConditionExpression = "step = :step_val",
			ExpressionAttributeValues = {":step_val": {"N": "0"}}
		)

	if not os.path.isdir(ZIP_PATH):
		os.mkdir(ZIP_PATH)

	if not os.path.isdir(TASK_PATH):
		os.mkdir(TASK_PATH)

	for i in queued_jobs['Items']:
		loc = i["upload_location"]["S"]
		fname = loc.split("/")[-1]
		task_name = fname.split(".")[0]
		s3_client.download_file(settings["S3_BUCKET"], fname, f"{ZIP_PATH}/{fname}")
		print(f"{fname} downloaded")
		f = open(f"{ZIP_PATH}/{fname}", "rb")
		zipfile.ZipFile(f).extractall(f"{TASK_PATH}/{task_name}")
		print(f"{fname} unzipped")
		f.close()
		


if __name__ == "__main__":
	reset_env()
	get_queued_jobs()