package com.example.enzoboully.projetreconnaissance;

import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.graphics.PixelFormat;
import android.hardware.Camera;
import android.media.Image;
import android.net.Uri;
import android.provider.MediaStore;
import android.os.Bundle;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class MainActivity extends Activity{
    ImageButton ImageButton;
    ImageButton ImageButton2;
    Boolean isPreview;
    FileOutputStream stream;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Nous mettons l'application en plein écran et sans barre de titre
        getWindow().setFormat(PixelFormat.TRANSLUCENT);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        isPreview = false;

        // Nous appliquons notre layout
        setContentView(R.layout.page_accueil);

        ImageButton = (ImageButton)findViewById(R.id.ImageButton);
        ImageButton2 = (ImageButton)findViewById(R.id.ImageButton2);

        ImageButton2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, Upload.class);
                startActivity(intent);
                finish();
            }
        });

        ImageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MainActivity.this, Photo.class);
                startActivity(intent);
                finish();
            }
        });
    }

    // Retour sur l'application
    @Override
    public void onResume() {
        super.onResume();
    }

    // Mise en pause de l'application
    @Override
    public void onPause() {
        super.onPause();
    }
}
