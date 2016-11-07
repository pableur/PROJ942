package com.adriendeconto.exemplehttp;

import android.app.ProgressDialog;
import android.content.Context;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

public class MainActivity extends AppCompatActivity {

    private String TAG="MainActivity";

    //ProgressDialog dialog = null;
    Button uploadButton;
    EditText editTextAddresse;
    EditText editTextPath;
    EditText editTextParam1;
    EditText editTextParam2;
    TextView textAnswer;
    HTTPsender request;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        final Context context = MainActivity.this;

        uploadButton = (Button)findViewById(R.id.uploadButton);
        editTextAddresse    = (EditText)findViewById(R.id.editTextAddresse);
        editTextPath        = (EditText)findViewById(R.id.editTextPath);
        editTextParam1      = (EditText)findViewById(R.id.editTextParam1);
        editTextParam2  = (EditText)findViewById(R.id.editTextParametre2);
        textAnswer      = (TextView)findViewById(R.id.idReponse);
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
                                textAnswer.setText(request.getServeurAnswer());
                            }
                        });
                    }
                }).start();
            }
        });
    }
}
