/*
* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

package pp.imagesegmenter;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.TextView;



import com.google.android.material.snackbar.Snackbar;

import pp.imagesegmenter.env.BorderedText;
import pp.imagesegmenter.env.ImageUtils;
import pp.imagesegmenter.env.Logger;
import pp.imagesegmenter.tracking.MultiBoxTracker;

import java.util.Collections;
import java.util.Dictionary;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.List;
import java.util.Set;
import java.util.Vector;

/**
* An activity that uses a Deeplab and ObjectTracker to segment and then track objects.
*/
public class MainActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    private static final int CROP_SIZE = 257;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;

    private Integer sensorOrientation;

    private Deeplab deeplab;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private byte[] luminanceCopy;

    private BorderedText borderedText;

    private Snackbar initSnackbar;

    private boolean initialized = false;
    Dictionary dict = new Hashtable();




    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        FrameLayout container = findViewById(R.id.container);
        initSnackbar = Snackbar.make(container, "Initializing...", Snackbar.LENGTH_INDEFINITE);

        init();

        final float textSizePx =
        TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(CROP_SIZE, CROP_SIZE, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        CROP_SIZE, CROP_SIZE,
                        sensorOrientation, false);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                canvas -> {
                    tracker.draw(canvas);
                    if (isDebug()) {
                        tracker.drawDebug(canvas);
                    }
                });

        addCallback(
                canvas -> {
                    if (!isDebug()) {
                        return;
                    }
                    final Bitmap copy = cropCopyBitmap;
                    if (copy == null) {
                        return;
                    }

                    final int backgroundColor = Color.argb(100, 0, 0, 0);
                    canvas.drawColor(backgroundColor);

                    final Matrix matrix = new Matrix();
                    final float scaleFactor = 2;
                    matrix.postScale(scaleFactor, scaleFactor);
                    matrix.postTranslate(
                            canvas.getWidth() - copy.getWidth() * scaleFactor,
                            canvas.getHeight() - copy.getHeight() * scaleFactor);
                    canvas.drawBitmap(copy, matrix, new Paint());

                    final Vector<String> lines = new Vector<String>();
                    lines.add("Frame: " + previewWidth + "x" + previewHeight);
                    lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
                    lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
                    lines.add("Rotation: " + sensorOrientation);
                    lines.add("Inference time: " + lastProcessingTimeMs + "ms");

                    borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
                });
    }

    OverlayView trackingOverlay;

    void init() {
        runInBackground(() -> {
            runOnUiThread(()->initSnackbar.show());
            try {
                deeplab = Deeplab.create(getAssets(), CROP_SIZE, CROP_SIZE, sensorOrientation);
            } catch (Exception e) {
                LOGGER.e("Exception initializing classifier!", e);
                finish();
            }
            runOnUiThread(()->initSnackbar.dismiss());
            initialized = true;
        });
    }

    @Override
    protected void processImage() {
        ++timestamp;
        setdict();
        //setContentView(R.layout.camera_connection_fragment_tracking);
        TextView text= (TextView)findViewById(R.id.outpclasses),textinf=(TextView)findViewById(R.id.infview);
        //String res ;
        final long currTimestamp = timestamp;
        byte[] originalLuminance = getLuminance();
        tracker.onFrame(
                previewWidth,
                previewHeight,
                getLuminanceStride(),
                sensorOrientation,
                originalLuminance,
                timestamp);
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection || !initialized) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        if (luminanceCopy == null) {
            luminanceCopy = new byte[originalLuminance.length];
        }
        System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                () -> {
                    LOGGER.i("Running detection on image " + currTimestamp);
                    final long startTime = SystemClock.uptimeMillis();

                    cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                    List<Deeplab.Recognition> mappedRecognitions =
                            deeplab.segment(croppedBitmap,cropToFrameTransform);
                    /*for(int i=0;i<mappedRecognitions.size();i++){
                        res+=mappedRecognitions.get(i+1).getId();
                    }*/
                    final String res=displayres(mappedRecognitions);
                    lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                    text.setText("Objects:"+res);
                    textinf.setText("Inference time: "+lastProcessingTimeMs+"ms");
                    tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
                    trackingOverlay.postInvalidate();

                    requestRender();
                    computingDetection = false;
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment_tracking;
    }
    protected  void setdict(){
        dict.put("1","aeroplane");
        dict.put("2","bicycle");
        dict.put("3","bird");
        dict.put("4","boat");
        dict.put("5","bottle");
        dict.put("6","bus");
        dict.put("7","car");
        dict.put("8","cat");
        dict.put("9","chair");
        dict.put("10","cow");
        dict.put("11","diningtable");
        dict.put("12","dog");
        dict.put("13","horse");
        dict.put("14","motorbike");
        dict.put("15","person");
        dict.put("16","pottedplant");
        dict.put("17","sheep");
        dict.put("18","sofa");
        dict.put("19","train");
        dict.put("20","TV");
    }

    protected String displayres(List<Deeplab.Recognition> recognitions) {
        String res="";
        Set<String> ids=new HashSet<String>();
        for(int i=0;i<recognitions.size();i++){
            ids.add(recognitions.get(i).getId());
        }
        for(String id:ids){
            res+=dict.get(id)+" ";
        }


        return res;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }
}
