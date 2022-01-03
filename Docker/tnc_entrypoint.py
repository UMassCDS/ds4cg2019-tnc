import tnc_detector
import json, time, zipfile, os, shutil
import boto3
import argparse

from simple_scheduler.recurring import recurring_scheduler

ZIP_PATH = "./zips"
TASK_PATH = "./tasks"
OUT_PATH = "./output"

settings = json.load(open("app_settings.json"))
ddb_client = boto3.client("dynamodb", "us-east-2")
s3_client = boto3.client("s3", "us-east-2")



def reset_env():
	if os.path.isdir(ZIP_PATH):
		shutil.rmtree(ZIP_PATH)

	if os.path.isdir(TASK_PATH):
		shutil.rmtree(TASK_PATH)

	if os.path.isdir(OUT_PATH):
		shutil.rmtree(OUT_PATH)

def setup_env():
	if not os.path.isdir(ZIP_PATH):
		os.mkdir(ZIP_PATH)

	if not os.path.isdir(TASK_PATH):
		os.mkdir(TASK_PATH)

	if not os.path.isdir(OUT_PATH):
		os.mkdir(OUT_PATH)


class detector_job_manager():

	#One class which manages downloading, detecting, and uploading.
	#largely just for the convenience of keeping the paths organized between the tasks.

	def __init__(self, model, ddb_result):
		self.model = model

		self.job_id = ddb_result["job_id"] #the primary key for the job table
		self.location = ddb_result["upload_location"]["S"]
		#we only need the key to download from s3- so the filename 
		self.fname = self.location.split("/")[-1] 
		#the task id is the original upload name plus a hash
		self.task_id = self.fname.split(".")[0]
		#the task name is just the original upload name, no hash:
		#it's still present in the directory structure of the zipfile.
		self.task_name = self.task_id.split("--")[0]
		
		#where we will dl the raw zip to
		self.zip_loc = f"{ZIP_PATH}/{self.fname}"
		
		#where we will unzip it to
		self.unzip_loc = f"{TASK_PATH}/{self.task_id}-raw"
		

		self.task_loc = f"{TASK_PATH}/{self.task_id}"
		
		#where we will output results to
		self.output_loc = f"{OUT_PATH}/{self.task_id}"
		self.output_zip_name = f'{ZIP_PATH}/{self.task_id}-output'
		self.output_zip = self.output_zip_name+".zip"
		self.output_fname = self.output_zip.split("/")[-1]

		self.error = False

	def download(self):
		s3_client.download_file(settings["S3_BUCKET"], self.fname, self.zip_loc)

		f = open(self.zip_loc, "rb")
		z = zipfile.ZipFile(f)

		z.extractall(self.unzip_loc)
		f.close()

		ddb_resp = ddb_client.update_item(
			TableName = settings["JOB_TABLE"],
			Key = {
				"job_id":self.job_id
			},
			UpdateExpression= "SET step = :step_val",
			ExpressionAttributeValues = {
				":step_val":{"N":"1"},
				

			}

		)
		
		os.mkdir(self.task_loc)
		for dpath, dname, fnames in os.walk(self.unzip_loc):
			for f in fnames:
				if(".jpg" in f or ".jpeg" in f):
					os.rename(os.path.join(dpath, f), os.path.join(self.task_loc, f))
				else:
					self.error = True
		
		if self.error:
			ddb_resp = ddb_client.update_item(
			TableName = settings["JOB_TABLE"],
			Key = {
				"job_id":self.job_id
			},
			UpdateExpression= "SET step = :step_val, error_msg=:error_msg",
			ExpressionAttributeValues = {
				":step_val":{"N":"3"},
				":error_msg":{"S": "Bad Zipfile"}
			}

		)
		#Now would be the time to validate the contents of the zipfile, also. Only jpgs please!



	def do_detection_task(self):
		tnc_detector.main(self.model, self.task_loc, OUT_PATH)

	def put_results(self):
		#zip the results
		zip_name = shutil.make_archive(self.output_zip_name,'zip', self.output_loc)
		#upload to s3
		s3_resp = s3_client.put_object(
			Bucket=settings["S3_BUCKET"],
			Body=open(f'{self.output_zip}', 'rb'),
			Key=self.output_fname
		)

		#update the ddb table
		ddb_resp = ddb_client.update_item(
			TableName = settings["JOB_TABLE"],
			Key = {
				"job_id":self.job_id
			},
			UpdateExpression= "SET step = :step_val, output_location=:output_val",
			ExpressionAttributeValues = {
				":step_val":{"N":"2"},
				":output_val":{"S":self.output_fname}

			}

		)
		print(s3_resp)
		print(ddb_resp)

	def run_job(self):
		setup_env()
		self.download()
		if(not self.error):
			self.do_detection_task()
			self.put_results()
			reset_env()


#Queries the dynamodb for jobs which are in step 0 (ie, just uploaded, this might change)
#Downloads them to the current filesystem
#and unzips them
def do_queued_jobs(model):
	queued_jobs = ddb_client.query(
			TableName = settings["JOB_TABLE"],
			IndexName = settings["STEP_INDEX"],
			Select = "ALL_ATTRIBUTES",
			KeyConditionExpression = "step = :step_val",
			ExpressionAttributeValues = {":step_val": {"N": "0"}}
		)
	print(f'Found {len(queued_jobs["Items"])} jobs in ddb')
	for job in queued_jobs['Items']:
		manager = detector_job_manager(model, ddb_result=job)
		manager.run_job()



def do_task(model = "megadetector_v4_1_0.pb"):
	print("Task is starting- will query ddb") 
	#Give the model by default here for scheduler's sake, while modularity isn't a concern.
	setup_env()
	do_queued_jobs(args.model)
	


parser = argparse.ArgumentParser(description="Nature Conservancy Image Detector")
parser.add_argument("model", help="The path to the Megadetector model to use for detection")



if __name__ == "__main__":
	args = parser.parse_args()
	do_task(args.model)
        #run it every ten minutes, for now.
	print("starting scheduler")
	#recurring_scheduler.add_job(do_task, 600, job_name="megadetector task")
	#recurring_scheduler.run()
