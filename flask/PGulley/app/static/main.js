

//Just a rendering handle placeholder while I'm assembling the functionality
job_line = function(job){
	if(job.output_location == undefined){
		return `<tr><td>${job.upload_location.S}</td><td>${new Date(job.timestamp.N*1000).toString()}</td><td>Not Ready</td></tr>`
	}
	else{
		return `<tr><td>${job.upload_location.S}</td><td>${new Date(job.timestamp.N*1000).toString()}</td>
		<td><button value='${job.job_id.S}' class="download">Download</button></td></tr>`
	}
	
}

//Calls the poll route on the lambda monolith, returns all of the relevant documents
poll_ddb = function(){
	$.ajax({
		url:window.location.href+"/poll_ddb"
	}).done(function(resp){
		table = $("#ddb_poll_table")
		table.empty()
		for( const job_record of resp["items"]){
			table.append(job_line(job_record))
		}
	})
}

poll_ddb() //call pollddb on load
setInterval(poll_ddb, 1000*60*10) //and also then every 10 minutes

///S3 upload concerns:
//- user uploads the file to the browser
//- user presses 'upload'- the browser asks the server for a signed aws url
//- the browesr recieves the signed url, then PUTs to it
//- the browser recieves status 200- tells the server to create a ddb entry
file_to_upload = ""
fname = ""

$("#file_upload").change(function(){
	reader = new FileReader()
	reader.onload = function(e){
		//do validation here
		file_to_upload = e.target.result
	}
	reader.readAsDataURL($(this)[0].files[0])
	fname = $(this)[0].files[0].name
})

$("#do_upload").click(function(){
	if(file_to_upload != ""){
		$.ajax({
			url:window.location.href+"/get_s3_upload_url",
			data:{filename:fname}
		}).done(function(resp){
			
			///this is adapted from an official amazon example.
			// it reads the data in, peels of the header,
			// and transforms it into an array of b64 bytes
			
			binary = atob(file_to_upload.split(",")[1]) 
			array = []
			for (var i = 0; i < binary.length; i++) {
	            array.push(binary.charCodeAt(i))
	        }
	       	blobData = new Blob([new Uint8Array(array)], {type: 'zip'})

			fetch(resp.uploadURL,{
				 
				method:"PUT",
				body:blobData
			
			}).then(function(resp){
				if(resp.status == 200){
					$.ajax({
						url:window.location.href+"/put_job_record_ddb",
						data:{location:resp.url.split("?")[0]}
					}).done(function(resp){
						console.log("whole loop complete")
						poll_ddb()
					})
				}
			})

		})
	}
})


$('body').on('click', 'button.download', function(){
	console.log("got attached")
	download_job_id = $(this)[0].value
	$.ajax({
		url:window.location.href+"/get_s3_download_url",
		data:{job_id:download_job_id}
	}).done(function(resp){
		link = document.createElement("a")
		link.href = resp.get_url
		link.click()
	})
})
