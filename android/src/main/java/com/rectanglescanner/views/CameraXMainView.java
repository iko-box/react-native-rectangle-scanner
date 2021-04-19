package com.rectanglescanner.views;

import android.app.Activity;
import android.content.Context;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.DisplayMetrics;
import android.view.LayoutInflater;

import androidx.camera.core.AspectRatio;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraControl;
import androidx.camera.core.CameraInfo;
import androidx.camera.core.CameraSelector;
import androidx.constraintlayout.widget.ConstraintLayout;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.WritableNativeMap;
import com.facebook.react.bridge.ReactContext;
import com.facebook.react.uimanager.events.RCTEventEmitter;
import com.facebook.react.uimanager.ThemedReactContext;

import com.google.common.util.concurrent.ListenableFuture;
import com.rectanglescanner.R;

import com.rectanglescanner.helpers.Quadrilateral;
import com.rectanglescanner.helpers.CustomOpenCVLoader;
import com.rectanglescanner.helpers.CapturedImage;

import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.view.PreviewView;
import androidx.camera.lifecycle.ProcessCameraProvider;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Rational;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.MatOfInt;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.UUID;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import com.facebook.react.bridge.Arguments;

import static java.lang.Math.abs;

public class CameraXMainView extends ConstraintLayout {
  protected static final String TAG = "CameraXMainView";
  public static CameraXMainView instance = null;
  PreviewView viewFinder;
  ProcessCameraProvider cameraProvider;
  ImageCapture imageCapture;
  ImageAnalysis imageAnalyzer;
  Preview preview;
  ExecutorService cameraExecutor;
  Activity currentActivity;
  Boolean processorBusy = false;
  CameraControl cameraControl;
  CameraInfo cameraInfo;

  private int filterId = 1;
  private Quadrilateral lastDetectedRectangle = null;
  private String cacheFolderName = "RNRectangleScanner";
  private double capturedQuality = 0.7;
  protected Context mContext;

  public static CameraXMainView getInstance() {
    return instance;
  }

  public static void createInstance(ThemedReactContext context, Activity activity) {
    instance = new CameraXMainView(context, activity);
  }

  private CameraXMainView(ThemedReactContext context, Activity activity) {
    super(context);

    this.currentActivity = activity;
    this.mContext = context;

    LayoutInflater lf = (LayoutInflater) context.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
    ConstraintLayout layout = (ConstraintLayout) lf.inflate(R.layout.activity_rectangle_scanner_camerax, null);

    addViewInLayout(layout, 0, new FrameLayout.LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT));

    viewFinder = layout.findViewById(R.id.view_finder);
    initializeOpenCV(context);

    this.cameraExecutor = Executors.newSingleThreadExecutor();

    viewFinder.setOnHierarchyChangeListener(new OnHierarchyChangeListener() {
      @Override
      public void onChildViewAdded(View parent, View child) {
        parent.measure(
                 MeasureSpec.makeMeasureSpec(getMeasuredWidth(), MeasureSpec.EXACTLY),
                 MeasureSpec.makeMeasureSpec(getMeasuredHeight(), MeasureSpec.EXACTLY)
        );
        parent.layout(0, 0, parent.getMeasuredWidth(), parent.getMeasuredHeight());
      }

      @Override
      public void onChildViewRemoved(View parent, View child) {

      }
    });
  }

  /**
   * Sets up the image processor. It uses OpenCV so it needs to load that first
   */
  private void initializeOpenCV(Context context) {
    BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(context) {
      @Override
      public void onManagerConnected(int status) {
        if (status == LoaderCallbackInterface.SUCCESS) {
          Log.d(TAG, "SUCCESS init OpenCV: " + status);
        } else {
          Log.d(TAG, "ERROR init OpenCV: " + status);
          super.onManagerConnected(status);
        }
      }
    };

    if (!OpenCVLoader.initDebug()) {
      CustomOpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, context, mLoaderCallback);
    } else {
      mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
    }
  }

  public void startCamera() {
    Log.i(TAG, "START CAMERA");

    viewFinder.post(new Runnable() {
      public void run() {
        final ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider
            .getInstance(mContext);

        cameraProviderFuture.addListener(new Runnable() {
          @Override
          public void run() {
            try {
              cameraProvider = cameraProviderFuture.get();
              cameraProvider.unbindAll();

              DisplayMetrics metrics = new DisplayMetrics();
              viewFinder.getDisplay().getRealMetrics(metrics);
              Log.d(TAG, "Screen metrics: " + metrics.widthPixels + " x " + metrics.heightPixels);

              int screenAspectRatio = aspectRatio(metrics.widthPixels, metrics.heightPixels);
              Log.d(TAG, "Preview aspect ratio: " + screenAspectRatio);

              CameraSelector cameraSelector = new CameraSelector.Builder()
                  .requireLensFacing(CameraSelector.LENS_FACING_BACK).build();

              int rotation = viewFinder.getDisplay().getRotation();
              preview = new Preview.Builder().setTargetAspectRatio(screenAspectRatio).setTargetRotation(rotation)
                  .build();

              imageCapture = new ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                  .setTargetAspectRatio(screenAspectRatio).setTargetRotation(rotation).build();

              imageAnalyzer = new ImageAnalysis.Builder().setTargetAspectRatio(screenAspectRatio)
                  .setTargetRotation(rotation).build();
              setImageAnalysis();

              viewFinder.setPreferredImplementationMode(PreviewView.ImplementationMode.SURFACE_VIEW);

              Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner) currentActivity, cameraSelector, preview,
                  imageCapture, imageAnalyzer);

              preview.setSurfaceProvider(viewFinder.createSurfaceProvider());

              cameraControl = camera.getCameraControl();
              cameraInfo = camera.getCameraInfo();


            } catch (ExecutionException | InterruptedException ex) {
              Log.d(TAG, ex.getLocalizedMessage());
            }
          }
        }, ContextCompat.getMainExecutor(mContext));
      }
    });
  }

  private int aspectRatio(int width, int height) {
    double previewRatio = 0;

    double RATIO_4_3_VALUE = 4.0 / 3.0;
    double RATIO_16_9_VALUE = 16.0 / 9.0;

    if (height > width) {
      previewRatio = height / width;
    } else {
      previewRatio = width / height;
    }

    if (abs(previewRatio - RATIO_4_3_VALUE) <= abs(previewRatio - RATIO_16_9_VALUE)) {
      return AspectRatio.RATIO_4_3;
    }
    return AspectRatio.RATIO_16_9;
  }

  private Bitmap toBitmap(Image image, int rotation) {
    Image.Plane[] planes = image.getPlanes();
    ByteBuffer buffer = planes[0].getBuffer();
    buffer.rewind();
    byte[] bytes = new byte[buffer.capacity()];
    buffer.get(bytes);
    Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);
    Matrix matrix = new Matrix();
    matrix.postRotate(rotation);
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
  }

  public void capture() {
    if (imageCapture != null) {
      imageCapture.takePicture(cameraExecutor, new ImageCapture.OnImageCapturedCallback() {

        @Override
        public void onCaptureSuccess(ImageProxy imageProxy) {
          int rotation = imageProxy.getImageInfo().getRotationDegrees();

          Bitmap bitmap = toBitmap(imageProxy.getImage(), rotation);
          if (bitmap == null) {
            WritableMap bitmapError = new WritableNativeMap();
            bitmapError.putString("message", "Null Captured Image");
            pictureDidFailToProcess(bitmapError);
            return;
          }

          Mat mat = new Mat();
          Utils.bitmapToMat(bitmap, mat);

          CapturedImage doc = cropImageToLatestQuadrilateral(mat);
          processedCapturedImage(doc);
          doc.release();
        }
      });
    }
  }

  private void setImageAnalysis() {
    imageAnalyzer.setAnalyzer(cameraExecutor, new ImageAnalysis.Analyzer() {
      @Override
      public void analyze(ImageProxy image) {
        // Analyzing live camera feed begins.
        if (processorBusy) {
          Log.d(TAG, "Busy");
          return;
        }
        Log.d(TAG, "Processed");
        processorBusy = true;

        final Bitmap bitmap = viewFinder.getBitmap();

        if (bitmap == null)
          return;

        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap, mat);

        detectRectangleInFrame(mat);

        processorBusy = false;

        image.close();
      }
    });
  }

  /**
   * Detects a rectangle from the image and sets the last detected rectangle
   */
  private void detectRectangleInFrame(Mat inputRgba) {
    ArrayList<MatOfPoint> contours = findContours(inputRgba);
    if (contours.isEmpty())
      return;

    org.opencv.core.Size srcSize = inputRgba.size();

    Quadrilateral detectedRectangle = getQuadrilateral(contours, srcSize);
    Bundle data = new Bundle();
    if (detectedRectangle != null) {
      Bundle quadMap = detectedRectangle.toBundle();
      data.putBundle("detectedRectangle", quadMap);
      Log.i(TAG, detectedRectangle.toString());
      Log.i(TAG, quadMap.toString());
      lastDetectedRectangle = detectedRectangle;
    } else {
      data.putBoolean("detectedRectangle", false);
    }

    rectangleWasDetected(Arguments.fromBundle(data));
  }

  /**
   * Crops the image to the latest detected rectangle and fixes perspective
   */
  private CapturedImage cropImageToLatestQuadrilateral(Mat capturedImage) {
    applyFilters(capturedImage);

    Mat doc;
    if (this.lastDetectedRectangle != null) {
      Mat croppedCapturedImage = this.lastDetectedRectangle.cropImageToRectangleSize(capturedImage);
      doc = fourPointTransform(croppedCapturedImage,
          this.lastDetectedRectangle.getPointsForSize(croppedCapturedImage.size()));
      croppedCapturedImage.release();
    } else {
      doc = new Mat(capturedImage.size(), CvType.CV_8UC4);
      capturedImage.copyTo(doc);
    }

    Core.flip(doc.t(), doc, 0);
    Core.flip(capturedImage.t(), capturedImage, 0);
    CapturedImage sd = new CapturedImage(capturedImage);

    sd.originalSize = capturedImage.size();
    sd.heightWithRatio = Double.valueOf(sd.originalSize.width).intValue();
    sd.widthWithRatio = Double.valueOf(sd.originalSize.height).intValue();
    return sd.setProcessed(doc);
  }

  private ArrayList<MatOfPoint> findContours(Mat src) {

    Mat grayImage;
    Mat cannedImage;
    Mat resizedImage;

    int height = Double.valueOf(src.size().height).intValue();
    int width = Double.valueOf(src.size().width).intValue();
    org.opencv.core.Size size = new org.opencv.core.Size(width, height);

    resizedImage = new Mat(size, CvType.CV_8UC4);
    grayImage = new Mat(size, CvType.CV_8UC4);
    cannedImage = new Mat(size, CvType.CV_8UC1);

    Imgproc.resize(src, resizedImage, size);
    Imgproc.cvtColor(resizedImage, grayImage, Imgproc.COLOR_RGBA2GRAY, 4);
    Imgproc.GaussianBlur(grayImage, grayImage, new org.opencv.core.Size(5, 5), 0);
    Imgproc.Canny(grayImage, cannedImage, 80, 100, 3, false);

    ArrayList<MatOfPoint> contours = new ArrayList<>();
    Mat hierarchy = new Mat();

    Imgproc.findContours(cannedImage, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

    hierarchy.release();

    Collections.sort(contours, new Comparator<MatOfPoint>() {

      @Override
      public int compare(MatOfPoint lhs, MatOfPoint rhs) {
        return Double.compare(Imgproc.contourArea(rhs), Imgproc.contourArea(lhs));
      }
    });

    resizedImage.release();
    grayImage.release();
    cannedImage.release();

    return contours;
  }

  private Quadrilateral getQuadrilateral(ArrayList<MatOfPoint> contours, org.opencv.core.Size srcSize) {

    int height = Double.valueOf(srcSize.height).intValue();
    int width = Double.valueOf(srcSize.width).intValue();
    org.opencv.core.Size size = new org.opencv.core.Size(width, height);

    for (MatOfPoint c : contours) {
      MatOfPoint2f c2f = new MatOfPoint2f(c.toArray());
      double peri = Imgproc.arcLength(c2f, true);
      MatOfPoint2f approx = new MatOfPoint2f();
      Imgproc.approxPolyDP(c2f, approx, 0.02 * peri, true);

      Point[] points = approx.toArray();

      // select biggest 4 angles polygon
      // if (points.length == 4) {
      Point[] foundPoints = sortPoints(points);

      if (insideArea(foundPoints, size)) {

        return new Quadrilateral(c, foundPoints, new org.opencv.core.Size(srcSize.width, srcSize.height));
      }
      // }
    }

    return null;
  }

  private Point[] sortPoints(Point[] src) {

    ArrayList<Point> srcPoints = new ArrayList<>(Arrays.asList(src));

    Point[] result = { null, null, null, null };

    Comparator<Point> sumComparator = new Comparator<Point>() {
      @Override
      public int compare(Point lhs, Point rhs) {
        return Double.compare(lhs.y + lhs.x, rhs.y + rhs.x);
      }
    };

    Comparator<Point> diffComparator = new Comparator<Point>() {

      @Override
      public int compare(Point lhs, Point rhs) {
        return Double.compare(lhs.y - lhs.x, rhs.y - rhs.x);
      }
    };

    // top-left corner = minimal sum
    result[0] = Collections.min(srcPoints, sumComparator);

    // bottom-right corner = maximal sum
    result[2] = Collections.max(srcPoints, sumComparator);

    // top-right corner = minimal difference
    result[1] = Collections.min(srcPoints, diffComparator);

    // bottom-left corner = maximal difference
    result[3] = Collections.max(srcPoints, diffComparator);

    return result;
  }

  private boolean insideArea(Point[] rp, org.opencv.core.Size size) {

    int width = Double.valueOf(size.width).intValue();
    int height = Double.valueOf(size.height).intValue();

    int minimumSize = width / 10;

    boolean isANormalShape = rp[0].x != rp[1].x && rp[1].y != rp[0].y && rp[2].y != rp[3].y && rp[3].x != rp[2].x;
    boolean isBigEnough = ((rp[1].x - rp[0].x >= minimumSize) && (rp[2].x - rp[3].x >= minimumSize)
        && (rp[3].y - rp[0].y >= minimumSize) && (rp[2].y - rp[1].y >= minimumSize));

    double leftOffset = rp[0].x - rp[3].x;
    double rightOffset = rp[1].x - rp[2].x;
    double bottomOffset = rp[0].y - rp[1].y;
    double topOffset = rp[2].y - rp[3].y;

    boolean isAnActualRectangle = ((leftOffset <= minimumSize && leftOffset >= -minimumSize)
        && (rightOffset <= minimumSize && rightOffset >= -minimumSize)
        && (bottomOffset <= minimumSize && bottomOffset >= -minimumSize)
        && (topOffset <= minimumSize && topOffset >= -minimumSize));

    return isANormalShape && isAnActualRectangle && isBigEnough;
  }

  /**
   * After an image is captured and cropped, this method is called
   */
  private void processedCapturedImage(CapturedImage capturedImage) {
    WritableMap pictureWasTakenConfig = new WritableNativeMap();
    WritableMap pictureWasProcessedConfig = new WritableNativeMap();
    String croppedImageFileName = null;
    String originalImageFileName = null;
    boolean hasCroppedImage = (capturedImage.processed != null);
    try {
      originalImageFileName = generateStoredFileName("O");
      if (hasCroppedImage) {
        croppedImageFileName = generateStoredFileName("C");
      } else {
        croppedImageFileName = originalImageFileName;
      }
    } catch (Exception e) {
      WritableMap folderError = new WritableNativeMap();
      folderError.putString("message", "Failed to create the cache directory");
      pictureDidFailToProcess(folderError);
      return;
    }

    pictureWasTakenConfig.putString("croppedImage", "file://" + croppedImageFileName);
    pictureWasTakenConfig.putString("initialImage", "file://" + originalImageFileName);
    pictureWasProcessedConfig.putString("croppedImage", "file://" + croppedImageFileName);
    pictureWasProcessedConfig.putString("initialImage", "file://" + originalImageFileName);
    pictureWasTaken(pictureWasTakenConfig);

    if (hasCroppedImage && !this.saveToDirectory(capturedImage.processed, croppedImageFileName)) {
      WritableMap fileError = new WritableNativeMap();
      fileError.putString("message", "Failed to write cropped image to cache");
      fileError.putString("filePath", croppedImageFileName);
      pictureDidFailToProcess(fileError);
      return;
    }
    if (!this.saveToDirectory(capturedImage.original, originalImageFileName)) {
      WritableMap fileError = new WritableNativeMap();
      fileError.putString("message", "Failed to write original image to cache");
      fileError.putString("filePath", originalImageFileName);
      pictureDidFailToProcess(fileError);
      return;
    }

    pictureWasProcessed(pictureWasProcessedConfig);
    capturedImage.release();
    Log.d(TAG, "Captured Images");
  }

  private String generateStoredFileName(String name) throws Exception {
    String folderDir = this.mContext.getCacheDir().toString();
    File folder = new File(folderDir + "/" + this.cacheFolderName);
    if (!folder.exists()) {
      boolean result = folder.mkdirs();
      if (result) {
        Log.d(TAG, "wrote: created folder " + folder.getPath());
      } else {
        Log.d(TAG, "Not possible to create folder");
        throw new Exception("Failed to create the cache directory");
      }
    }
    return folderDir + "/" + this.cacheFolderName + "/" + name + UUID.randomUUID() + ".png";
  }

  /**
   * Saves a file to a folder
   */
  private boolean saveToDirectory(Mat doc, String fileName) {
    Mat endDoc = new Mat(doc.size(), CvType.CV_8UC4);
    doc.copyTo(endDoc);
    Core.flip(doc.t(), endDoc, 1);

    int height = Double.valueOf(endDoc.size().height).intValue();
    int width = Double.valueOf(endDoc.size().width).intValue();
    int nbPixel = height * width;
    if (nbPixel > 3000000) {
      // RESIZE BIG PICTURE
      double r = Math.sqrt(((double) nbPixel) / 3000000);
      org.opencv.core.Size size = new org.opencv.core.Size(width / r, height / r);
      Mat resizedDoc = new Mat(size, CvType.CV_8UC4);
      Imgproc.resize(endDoc, resizedDoc, size);
      endDoc.release();
      endDoc = resizedDoc;
    }

    Mat bgrDoc = new Mat(endDoc.size(), CvType.CV_8UC4);
    // Preview getBitmap gives RGB & Imgcodecs.imwrite needs BGR
    Imgproc.cvtColor(endDoc, bgrDoc, Imgproc.COLOR_RGBA2BGRA, 4);
    ArrayList<Integer> parameters = new ArrayList();
    parameters.add(Imgcodecs.CV_IMWRITE_JPEG_QUALITY);
    parameters.add((int) (this.capturedQuality * 100));
    MatOfInt par = new MatOfInt();
    par.fromList(parameters);
    boolean success = Imgcodecs.imwrite(fileName, bgrDoc, par);

    endDoc.release();
    bgrDoc.release();

    return success;
  }

  public void setEnableTorch(boolean enable) {
    if (cameraControl != null)
      cameraControl.enableTorch(enable);
    torchWasChanged(enable);
  }

  /*
   * ! Applies filters to the image based on the set filter
   */
  public void applyFilters(Mat image) {
    switch (filterId) {
      case 1: {
        // original image
        break;
      }
      case 2: {
        // applyGreyscaleFilterToImage(image);
        break;
      }
      case 3: {
        // applyColorFilterToImage(image);
        break;
      }
      case 4: {
        // applyBlackAndWhiteFilterToImage(image);
        break;
      }
      default:
        // original image
    }
  }

  private Mat fourPointTransform(Mat src, Point[] pts) {
    Point tl = pts[0];
    Point tr = pts[1];
    Point br = pts[2];
    Point bl = pts[3];

    double widthA = Math.sqrt(Math.pow(br.x - bl.x, 2) + Math.pow(br.y - bl.y, 2));
    double widthB = Math.sqrt(Math.pow(tr.x - tl.x, 2) + Math.pow(tr.y - tl.y, 2));

    double dw = Math.max(widthA, widthB);
    int maxWidth = Double.valueOf(dw).intValue();

    double heightA = Math.sqrt(Math.pow(tr.x - br.x, 2) + Math.pow(tr.y - br.y, 2));
    double heightB = Math.sqrt(Math.pow(tl.x - bl.x, 2) + Math.pow(tl.y - bl.y, 2));

    double dh = Math.max(heightA, heightB);
    int maxHeight = Double.valueOf(dh).intValue();

    Mat doc = new Mat(maxHeight, maxWidth, CvType.CV_8UC4);

    Mat src_mat = new Mat(4, 1, CvType.CV_32FC2);
    Mat dst_mat = new Mat(4, 1, CvType.CV_32FC2);

    src_mat.put(0, 0, tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y);
    dst_mat.put(0, 0, 0.0, 0.0, dw, 0.0, dw, dh, 0.0, dh);

    Mat m = Imgproc.getPerspectiveTransform(src_mat, dst_mat);

    Imgproc.warpPerspective(src, doc, m, doc.size());

    return doc;
  }

  public void setCapturedQuality(double quality) {
    this.capturedQuality = quality;
  }

  public void setFilterId(int filterId) {
    this.filterId = filterId;
  }

  public void stopCamera() {
    if (cameraProvider != null)
      cameraProvider.unbindAll();
  }

  public void cleanupCamera() {
    // view.cleanupCamera();
  }

  public void focusCamera() {
    // view.focusCamera();
  }

  public void deviceWasSetup(WritableMap config) {
    final ReactContext context = (ReactContext) getContext();
    context.getJSModule(RCTEventEmitter.class).receiveEvent(getId(), "onDeviceSetup", config);
  }

  public void torchWasChanged(boolean torchEnabled) {
    WritableMap map = new WritableNativeMap();
    map.putBoolean("enabled", torchEnabled);
    final ReactContext context = (ReactContext) getContext();
    context.getJSModule(RCTEventEmitter.class).receiveEvent(getId(), "onTorchChanged", map);
  }

  public void rectangleWasDetected(WritableMap detection) {
    final ReactContext context = (ReactContext) getContext();
    context.getJSModule(RCTEventEmitter.class).receiveEvent(getId(), "onRectangleDetected", detection);
  }

  public void pictureWasTaken(WritableMap pictureDetails) {
    final ReactContext context = (ReactContext) getContext();
    context.getJSModule(RCTEventEmitter.class).receiveEvent(getId(), "onPictureTaken", pictureDetails);
  }

  public void pictureWasProcessed(WritableMap pictureDetails) {
    final ReactContext context = (ReactContext) getContext();
    context.getJSModule(RCTEventEmitter.class).receiveEvent(getId(), "onPictureProcessed", pictureDetails);
  }

  public void pictureDidFailToProcess(WritableMap errorDetails) {
    final ReactContext context = (ReactContext) getContext();
    context.getJSModule(RCTEventEmitter.class).receiveEvent(getId(), "onErrorProcessingImage", errorDetails);
  }
}
