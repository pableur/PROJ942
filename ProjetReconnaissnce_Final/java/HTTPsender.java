package com.example.enzoboully.projetreconnaissance;

import android.app.Activity;
import android.app.ProgressDialog;
import android.util.Log;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;

/**
 * Created by adrien on 31/10/2016.
 */
public class HTTPsender extends Activity {

    ProgressDialog dialog = null;
    private String TAG="HTTPsender";
    private String url="";
    private String filePath="";
    private String parametre[];
    private int serverResponseCode = 0;
    private String serveurAnswer = "";

    public HTTPsender(String url, String filePath, String parametre[]){
        this.filePath=filePath;
        this.url=url;
        this.parametre=parametre;


        Log.i(TAG,"Uploading file....");
        // uploadFile();

        new Thread(new Runnable() {
            public void run() {

                runOnUiThread(new Runnable() {
                    public void run() {

                    }
                });

                uploadFile();

            }
        }).start();

    }

    public int uploadFile() {

        String fileName = this.filePath;

        HttpURLConnection conn = null;
        DataOutputStream dos = null;
        BufferedReader reader = null;
        String lineEnd = "\r\n";
        String twoHyphens = "--";
        String boundary = "*****";
        int bytesRead, bytesAvailable, bufferSize;
        byte[] buffer;
        int maxBufferSize = 1 * 1024 * 1024;
        File sourceFile = new File(fileName);

        if (!sourceFile.isFile()) {
            Log.e(TAG, "Source File not exist :"+fileName);
            return 0;
        }
        else
        {
            try {

                // open a URL connection to the Servlet
                FileInputStream fileInputStream = new FileInputStream(sourceFile);
                final URL url = new URL(this.url);

                // Open a HTTP  connection to  the URL
                conn = (HttpURLConnection) url.openConnection();
                conn.setDoInput(true); // Allow Inputs
                conn.setDoOutput(true); // Allow Outputs
                conn.setUseCaches(false); // Don't use a Cached Copy
                conn.setRequestMethod("POST");
                conn.setRequestProperty("Connection", "Keep-Alive");
                conn.setRequestProperty("ENCTYPE", "multipart/form-data");
                conn.setRequestProperty("Content-Type", "multipart/form-data;boundary=" + boundary);
                conn.setRequestProperty("uploaded_file", fileName);

                dos = new DataOutputStream(conn.getOutputStream());

                dos.writeBytes(twoHyphens + boundary + lineEnd);
                for(int i=0; i<parametre.length;i=i+2) {
                    dos.writeBytes("Content-Disposition: form-data; name=\""+parametre[i]+"\"\"\r\n\n"+parametre[i+1]+"\r\n");
                    Log.i(TAG,"Content-Disposition: form-data; name=\""+parametre[i]+"\"\"\r\n\n"+parametre[i+1]+"\r\n");
                    dos.writeBytes(twoHyphens + boundary + lineEnd);
                }
                dos.writeBytes(twoHyphens + boundary + lineEnd);
                dos.writeBytes("Content-Disposition: form-data; name=\"image\";filename="+'"'+ fileName +'"' + lineEnd);
                dos.writeBytes(lineEnd);

                // create a buffer of  maximum size
                bytesAvailable = fileInputStream.available();

                bufferSize = Math.min(bytesAvailable, maxBufferSize);
                buffer = new byte[bufferSize];

                // read file and write it into form...
                bytesRead = fileInputStream.read(buffer, 0, bufferSize);

                while (bytesRead > 0) {

                    dos.write(buffer, 0, bufferSize);
                    bytesAvailable = fileInputStream.available();
                    bufferSize = Math.min(bytesAvailable, maxBufferSize);
                    bytesRead = fileInputStream.read(buffer, 0, bufferSize);
                }
                // send multipart form data necesssary after file data...

                dos.writeBytes(lineEnd);
                dos.writeBytes(twoHyphens + boundary + twoHyphens + lineEnd);

                // Responses from the server (code and message)
                serverResponseCode = conn.getResponseCode();
                //String serverResponseMessage = conn.getResponseMessage();
                reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                String ligne;
                String a="";
                while ((ligne = reader.readLine()) != null) {
                    a += ligne;
                }
                reader.close();
                String serverResponseMessage=a.toString();
                serveurAnswer=serverResponseMessage;
                Log.i(TAG, "HTTP Response is : " + serverResponseMessage + ": " + serverResponseCode);

                if(serverResponseCode == 200){
                    String msg = "File Upload Completed. See uploaded file here : "+this.url;
                    Log.i(TAG,msg);
                    Log.i(TAG,"Answer "+serverResponseMessage);
                }

                dos.flush();
                dos.close();
                fileInputStream.close();


            } catch (MalformedURLException ex) {
                ex.printStackTrace();
                Log.i(TAG,"MalformedURLException Exception : check script url.");
                Log.e("Upload file to server", "error: " + ex.getMessage(), ex);
            } catch (Exception e) {
                e.printStackTrace();
                Log.e(TAG, "Exception : "+ e.getMessage(), e);
            }
            return serverResponseCode;

        } // End else block
    }
    public int getServerResponseCode(){
        return serverResponseCode;
    }
    public String getServeurAnswer(){
        return serveurAnswer;
    }
}
