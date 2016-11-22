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
import android.provider.Settings;
import android.view.KeyEvent;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

import static android.os.SystemClock.sleep;

public class Photo extends Activity implements SurfaceHolder.Callback {
    Camera camera;
    SurfaceView surfaceCamera;
    Boolean isPreview;
    FileOutputStream stream;
    TextView Text1,Text2;
    Button OKButton;
    String s;

    HTTPsender request2;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Nous mettons l'application en plein écran et sans barre de titre
       getWindow().setFormat(PixelFormat.TRANSLUCENT);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        isPreview = false;

        setContentView(R.layout.content_photo);

        // Nous récupérons notre surface pour le preview
        surfaceCamera = (SurfaceView) findViewById(R.id.surfaceViewCamera);
        Text1 = (TextView) findViewById(R.id.Titre);
        Text2 = (TextView) findViewById(R.id.personne);
        OKButton = (Button) findViewById(R.id.button);

        // Méthode d'initialisation de la caméra

        InitializeCamera();

        // Quand nous cliquons sur notre surface
        surfaceCamera.setOnClickListener(new View.OnClickListener() {

            public void onClick(View v) {
                // Nous prenons une photo
                if (camera != null) {
                    SavePicture();
                    sleep(2000);
                    new Thread(new Runnable() {
                        public void run() {
                            runOnUiThread(new Runnable() {
                                public void run() {
                                    DisplayResult();
                                }
                            });
                        }
                    }).start();
            }

            }
        });

        OKButton.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Intent intent = new Intent(Photo.this, MainActivity.class);
                startActivity(intent);
                finish();
            }
        });
    }

    // Retour sur l'application
    @Override
    public void onResume() {
        super.onResume();
        camera = Camera.open();
    }

    // Mise en pause de l'application
    @Override
    public void onPause() {
        super.onPause();

        if (camera != null) {
            camera.release();
            camera = null;
        }
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event)
    {
        if ((keyCode == KeyEvent.KEYCODE_BACK))
        {
            Intent intent = new Intent(Photo.this, MainActivity.class);
            startActivity(intent);
            finish();
        }
        return super.onKeyDown(keyCode, event);
    }

    private void SavePicture() {
        try {
            SimpleDateFormat timeStampFormat = new SimpleDateFormat(
                    "yyyy-MM-dd-HH.mm.ss");
            String fileName = "photo_" + timeStampFormat.format(new Date())
                    + ".jpg";

            // Metadata pour la photo
            ContentValues values = new ContentValues();
            values.put(MediaStore.Images.Media.TITLE, fileName);
            System.out.println("Titre:"+fileName);
            values.put(MediaStore.Images.Media.DISPLAY_NAME, fileName);
            values.put(MediaStore.Images.Media.DESCRIPTION, "Image prise par Photo");
            values.put(MediaStore.Images.Media.DATE_TAKEN, new Date().getTime());
            values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");

            // Support de stockage
            Uri taken = getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                    values);
            System.out.println(taken.getPath());
            File myFile = new File(taken.getPath());
            System.out.println(myFile.getAbsolutePath());
            // Ouverture du flux pour la sauvegarde
            stream = (FileOutputStream) getContentResolver().openOutputStream(taken);
            s="/storage/emulated/0/DCIM/Camera/"+System.currentTimeMillis()+".jpg";
            camera.takePicture(null, pictureCallback, pictureCallback);

        } catch (Exception e) {
            // TODO: handle exception
            System.out.println("Erreur SavePicture: "+e);
        }

    }

    // Callback pour la prise de photo
    Camera.PictureCallback pictureCallback = new Camera.PictureCallback() {

        public void onPictureTaken(byte[] data, Camera camera) {
            if (data != null) {
                // Enregistrement de votre image
                try {
                    if (stream != null) {
                        stream.write(data);
                        stream.flush();
                        stream.close();
                    }
                } catch (Exception e) {
                    // TODO: handle exception
                    System.out.println("Erreur onPictureTaken: "+e);
                }

                // Nous redémarrons la prévisualisation
                camera.startPreview();
            }
        }
    };

    // Quand la surface change
    public void surfaceChanged(SurfaceHolder holder, int format, int width,
                               int height) {

        // Si le mode preview est lancé alors nous le stoppons
        if (isPreview) {
            camera.stopPreview();
        }
        Camera.Parameters p = camera.getParameters();
        p.setRotation(90);
        camera.setParameters(p);
        camera.setDisplayOrientation(90);;

        try {
            // Nous attachons notre prévisualisation de la caméra au holder de la surface
            camera.setPreviewDisplay(surfaceCamera.getHolder());
        } catch (IOException e) {
            System.out.println("Erreur SetpreviewDisplay: "+e);
        }

        // Nous lançons la preview
        camera.startPreview();

        isPreview = true;
    }
    public void surfaceCreated(SurfaceHolder holder) {
        // Nous prenons le contrôle de la camera
        if (camera == null)
            camera = Camera.open();
    }
    public void surfaceDestroyed(SurfaceHolder holder) {
        // Nous arrêtons la camera et nous rendons la main
        if (camera != null) {
            camera.stopPreview();
            isPreview = false;
            camera.release();
        }
    }

    public void InitializeCamera() {
        // Nous attachons nos retours du holder à notre activité
        surfaceCamera.getHolder().addCallback(this);
        // Nous spécifiions le type du holder en mode SURFACE_TYPE_PUSH_BUFFERS
        surfaceCamera.getHolder().setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
    }

    public void DisplayResult(){
        File f = new File("/storage/emulated/0/DCIM/Camera/");
        System.out.println( f.listFiles()[f.listFiles().length-1].getAbsolutePath());
        s= f.listFiles()[f.listFiles().length-1].getAbsolutePath();
        String tab[]={};
        request2=new HTTPsender("http://192.168.118.103/form.php",s,tab);

        new Thread(new Runnable() {
            public void run() {

                runOnUiThread(new Runnable() {
                    public void run() {
                        while(request2.getServeurAnswer()==""){}
                        Text2.setText("PERSONNE IDENTIFIÉE:\n"+request2.getServeurAnswer()+"       ");
                    }
                });
            }
        }).start();
        Text1.setVisibility(View.VISIBLE);
        Text2.setVisibility(View.VISIBLE);
        OKButton.setVisibility(View.VISIBLE);
        surfaceCamera.setVisibility(View.INVISIBLE);
    }
}
