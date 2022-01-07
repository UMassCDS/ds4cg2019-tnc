
var opts = {
  lines: 13, // The number of lines to draw
  length: 38, // The length of each line
  width: 17, // The line thickness
  radius: 45, // The radius of the inner circle
  scale: 0.1, // Scales overall size of the spinner
  corners: 1, // Corner roundness (0..1)
  speed: 0.5, // Rounds per second
  rotate: 0, // The rotation offset
  animation: 'spinner-line-fade-more', // The CSS animation name for the lines
  direction: 1, // 1: clockwise, -1: counterclockwise
  color: '#ffffff', // CSS color or array of colors
  fadeColor: 'dark-grey', // CSS color or array of colors
  top: 0, // Top position relative to parent
  left: '50%', // Left position relative to parent
  shadow: '0 0 1px transsparent', // Box-shadow for the lines
  zIndex: 2000000000, // The z-index (defaults to 2e9)
  className: 'spinner', // The CSS class to assign to the spinner
  position: 'relative', // Element positioning
};

default_color ="#DCDCDC"
hover_color = "#c2c2ea"
error_color = "#f6ffca"
success_color = "#ffcdca"


upload_spinner = new Spin.Spinner(opts)
opts.scale = 0.5
dz_spinner = new Spin.Spinner(opts)

//Just a rendering handle placeholder while I'm assembling the functionality
job_line = function(job){
	state_message = ""
	if(job.step.N == 0){
		state_message = "Waiting"
		
	}
	if(job.step.N == 1){
		state_message = "Working"
	}
	if(job.step.N == 2){
		state_message = `Complete: <button value='${job.job_id.S}' class="download">Download</button>`
	}
	if(job.step.N == 3){
		state_message = `Error: ${job.error_msg.S }`
	}

	filename = job.upload_location.S.split("/").pop()
	return `<tr><td>${filename}</td><td sorttable_customkey="${job.timestamp.N}">${new Date(job.timestamp.N*1000).toString().split("GMT")[0]}</td><td>${state_message}</td></tr>`
	
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
		//sorttable.makeSortable(table);
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


function do_upload_loop_drag_drop(fname, file){
	dz_spinner.spin($("#dropzone_spinner")[0])
	$.ajax({
			url:window.location.href+"/get_s3_upload_url",
			data:{filename:fname}
	}).done(function(resp){
		console.log(resp)
		binary = atob(file.split(",")[1]) 
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
						
						dz_spinner.stop()
						$(".dropzone").addClass("success", {duration:250})
						$("#dz_success").fadeIn(250)
						setTimeout(function(){
							$(".dropzone").removeClass("success", {duration:500})
							$("#dz_success").fadeOut(500)
							setTimeout(function(){
								$("#dz_default").fadeIn(500)
							},500)
						},2000)
						poll_ddb()
					})
				}
			})

	})
}

$("#do_upload").click(function(){
	if(file_to_upload != ""){
		upload_spinner.spin($("#upload_spinner")[0])
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
						upload_spinner.stop()
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

input_reader = new FileReader()
input_reader.onloadend = function(e){
	console.log("finished read load")
	
	if(fname.split(".")[1] != "zip"){
		console.log("badfilenouplode")
	
		$(".dropzone").addClass("error",{duration:0})
		$("#dz_error").fadeIn(0)
		
		setTimeout(function(){
			$(".dropzone").removeClass("error", {duration:500})
			$("#dz_error").fadeOut(500)
			setTimeout(function(){
				$("#dz_default").fadeIn(250)
			},500)
		}, 2000)
	}
	else{
		file = e.target.result
		
		do_upload_loop_drag_drop(fname, file)
	}
	
}

function dropHandler(ev) {
 
  // Prevent default behavior (Prevent file from being opened)
  ev.preventDefault();
  $(".dropzone").removeClass("hover", {duration:0})
  $("#dz_default").fadeOut(0)
  if (ev.dataTransfer.items) {
    // Use DataTransferItemList interface to access the file(s)
    for (var i = 0; i < ev.dataTransfer.items.length; i++) {
      // If dropped items aren't files, reject them
      if (ev.dataTransfer.items[i].kind === 'file') {
        var file = ev.dataTransfer.items[i].getAsFile();
        fname = file.name
        input_reader.readAsDataURL(file)
        
        
      }
    }
  } else {
    // Use DataTransfer interface to access the file(s)
    for (var i = 0; i < ev.dataTransfer.files.length; i++) {

      fname = ev.dataTransfer.files[0].name
      input_reader.readAsDataURL(ev.dataTransfer.files[0])
    }
  }
}

function dragenter(ev){
	$(".dropzone").addClass("hover", {duration:250})
}

function dragexit(ev){
	$(".dropzone").removeClass("hover", {duration: 250})
}

function dragOverHandler(ev) {
  // Prevent default behavior (Prevent file from being opened)
  ev.preventDefault();
}
