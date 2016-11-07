<?php
	$name="none";
	$file_name="none";
	$pathUpload="/var/www/html/upload/";
	if(isset($_POST['name']))
		$name=$_POST['name'];
	if(isset($_FILES['image']) and $_FILES['image']['error']==0){
		$timeStart=microtime($get_as_float=True);
		do{
			$pathUpload=$pathUpload.microtime(True)."/";
		}while(mkdir($pathUpload,0777)==False);
		// upload de l'image
		move_uploaded_file($_FILES['image']['tmp_name'],$pathUpload.basename($_FILES['image']['name']));
		$file_name=$_FILES['image']['name'];
	}
	
	if(isset($_POST['name'])or isset($_FILES['image'])){
		$monfichier = fopen('log.txt', 'a');
		if($file_name!="none")
			$file_name=$pathUpload.$file_name;
		
		fputs($monfichier,$file_name.", ".$name."\r\n");
		fclose($monfichier);
	}
?>


<!doctype html>


<html lang="en">
<head>
  <meta charset="utf-8">

  <title></title>
  <meta name="description" content="">
  <meta name="author" content="">

</head>

<body>
  <form action="" method="post" enctype="multipart/form-data">
	<input type="file" name="image" capture>
	<input name="name">
  </form>
</body>
</html>