package com.rectanglescanner;

import com.rectanglescanner.views.CameraXMainView;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;

/**
 * Created by Jake on Jan 6, 2020.
 */

public class RNRectangleScannerModule extends ReactContextBaseJavaModule {

    public RNRectangleScannerModule(ReactApplicationContext reactContext) {
        super(reactContext);
    }

    @Override
    public String getName() {
        return "RNRectangleScannerManager";
    }

    @ReactMethod
    public void start() {
        CameraXMainView view = CameraXMainView.getInstance();
        view.startCamera();
    }

    @ReactMethod
    public void stop() {
        CameraXMainView view = CameraXMainView.getInstance();
        view.stopCamera();
    }

    @ReactMethod
    public void cleanup() {
        CameraXMainView view = CameraXMainView.getInstance();
        view.cleanupCamera();
    }

    @ReactMethod
    public void refresh() {
        CameraXMainView view = CameraXMainView.getInstance();
        view.stopCamera();
        view.startCamera();
    }

    @ReactMethod
    public void capture() {
        CameraXMainView view = CameraXMainView.getInstance();
        view.capture();
    }

    @ReactMethod
    public void focus() {
        CameraXMainView view = CameraXMainView.getInstance();
        view.focusCamera();
    }
}
