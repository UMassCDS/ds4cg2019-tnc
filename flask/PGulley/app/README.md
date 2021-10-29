# TNC Flask App
Serving a simple web interface which will host user profiles and interface with ddb and s3 to host and track classification jobs. The app is built to run on aws lambda.

# Running the app
for development mode, just run 'python tnc_app.py'

# Deploying the app
I'm using Zappa to manage the deployment to lambda- refer to their documentation for more information
In breif, to deploy or update:
''' zappa deploy basic_deployment_test '''
''' zappa update basic_deployment_test'''

'''zappa_settings.json''' contains all of the different deployment settings. Right now we only have the development stage,
but once things are somewhat more settled I'll transition into a new production stage. 