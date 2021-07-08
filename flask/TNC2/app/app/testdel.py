from threading import Thread
import requests
import time

UPLOAD_URL = "http://127.0.0.1:5000/upload"
IMAGE_PATH = "stress.jpg"
NUM_REQUESTS = 1000
SLEEP_COUNT = 0.05

def upload(n):
    image = open(IMAGE_PATH, "rb").read()
    payload = {"image": image}
    r = requests.post(UPLOAD_URL, files=payload).json()
    if r["success"]:
        print("[INFO] thread {} OK".format(n))
	# otherwise, the request failed
    else:
        print("[INFO] thread {} FAILED".format(n))

for i in range(0, NUM_REQUESTS):
    	# start a new thread to call the API
	t = Thread(target=upload, args=(i,))
	t.daemon = True
	t.start()
	time.sleep(SLEEP_COUNT)

# time.sleep()

    
