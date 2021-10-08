## TNC Flask App
Serving a simple web interface which hosts user profiles and sends user images (or zips containing images) to a separate model server.

TODO:

Templates:
	Landing Page (+/- login)
	Main Interface Page.

Routes:
	Main
		-serves the main page- landing if not logged in, main if logged in.
	Login
		-updates the session if authentication is correct
	Logout
		-updates the session regardless
	Upload_Files
		-recieves the uploaded image or zip, sends them on to the model server
		 
