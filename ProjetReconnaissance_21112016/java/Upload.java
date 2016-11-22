package com.example.enzoboully.projetreconnaissance;

import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

public class Upload extends AppCompatActivity {

    private String TAG="MainActivity";
    private static final int MY_INTENT_CLICK=302;

    //ProgressDialog dialog = null;
    Button uploadButton;
    Button pickImageButton;
    EditText editTextAddresse;
    EditText editTextPath;
    EditText editTextParam1;
    EditText editTextParam2;
    TextView textAnswer;
    HTTPsender request;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.content_upload);
        final Context context = Upload.this;

        uploadButton = (Button)findViewById(R.id.uploadButton);
        editTextAddresse    = (EditText)findViewById(R.id.editTextAddresse);
        editTextPath        = (EditText)findViewById(R.id.editTextPath);
        editTextParam1      = (EditText)findViewById(R.id.editTextParam1);
        editTextParam2  = (EditText)findViewById(R.id.editTextParametre2);
        textAnswer      = (TextView)findViewById(R.id.idReponse);
        pickImageButton = (Button)findViewById(R.id.pickImageButton);

        uploadButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                Log.i(TAG,"Adresse : "+editTextAddresse.getText().toString());
                Log.i(TAG,"Path : "+editTextPath.getText().toString());
                request = new HTTPsender(
                        editTextAddresse.getText().toString(),
                        editTextPath.getText().toString(),
                        new String[]{"nameImage", editTextParam1.getText().toString(), "param2", editTextParam2.getText().toString()});
                new Thread(new Runnable() {
                    public void run() {

                        runOnUiThread(new Runnable() {
                            public void run() {
                                while(request.getServeurAnswer()==""){}
                                textAnswer.setText("PERSONNE IDENTIFIÉE: "+request.getServeurAnswer());
                            }
                        });
                    }
                }).start();
            }
        });

        pickImageButton.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v)
            {
                Intent intent = new Intent();
                intent.setType("*/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent, "Select File"),MY_INTENT_CLICK);
            }
        });

    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        if (resultCode == RESULT_OK)
        {
            if (requestCode == MY_INTENT_CLICK)
            {
                if (null == data) return;

                String selectedImagePath;
                Uri selectedImageUri = data.getData();

                //MEDIA GALLERY
                selectedImagePath = ImageFilePath.getPath(getApplicationContext(), selectedImageUri);
                Log.i("Image File Path", ""+selectedImagePath);
                editTextPath.setText(selectedImagePath);
            }
        }
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event)
    {
        if ((keyCode == KeyEvent.KEYCODE_BACK))
        {
            Intent intent = new Intent(Upload.this, MainActivity.class);
            startActivity(intent);
            finish();
        }
        return super.onKeyDown(keyCode, event);
    }
}
