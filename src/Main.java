import org.opencv.core.Core;

// Driver class for calibration / pose estimation
public class Main {
    static { System.loadLibrary(Core.NATIVE_LIBRARY_NAME); }

    public static void main(String[] args) {
        System.out.println(Core.VERSION);
        ChessBoardCalibration.calibrate();
    }

}
