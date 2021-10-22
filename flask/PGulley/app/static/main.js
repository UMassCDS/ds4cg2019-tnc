
	$("#test_ajax").click(function(){
		$.ajax({
			url:"/ajax_test",
			data:{name:"nothin"}
		}).done(function(resp){
			
			$("#sample_body").append(`<tr><td>${resp.status}</td><td>${resp.data}</td></tr>`);
		});
	})



