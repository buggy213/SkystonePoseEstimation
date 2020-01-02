import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.Feature2D;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;

// Inspired by https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
public class ChessBoardCalibration {

    private static final String CALIBRATION_IMAGE_PATH = "";
    private static final Size CHESSBOARD_SIZE = new Size(8, 8);
    private static final float SQUARE_SIZE = 0; // (mm)
    private static final TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 30, 0.001);

    // reuse mats (to avoid constant allocation / deallocation
    private static Mat gray;
    private static MatOfPoint2f imagePoints = new MatOfPoint2f();

    private static Mat cameraMatrix, distCoeffs = new Mat();
    private static List<Mat> rvecs, tvecs = new ArrayList<>();

    private static List<Mat> loadCalibrationImages() {
        // Load all images as mats from calibrationImagePath directory

        Stream<Path> imagePaths;
        try {
            imagePaths = Files.list(Paths.get(CALIBRATION_IMAGE_PATH));
        }
        catch (IOException e) {
            System.err.println("Failed to load calibration images from: " + CALIBRATION_IMAGE_PATH);
            return new ArrayList<>();
        }
        ArrayList<Mat> images = new ArrayList<>();
        Iterator<Path> i = imagePaths.iterator();
        while (i.hasNext()) {
            Path imagePath = i.next();
            images.add(Imgcodecs.imread(imagePath.toString()));
        }

        return images;
    }

    public static class CalibrationResults {
        Mat cameraMatrix;
        Mat distortionCoefficients;

        List<Mat> rotationVectors;
        List<Mat> translationVectors;

        public CalibrationResults(Mat cameraMatrix, Mat distortionCoefficients,
                                  List<Mat> rotationVectors, List<Mat> translationVectors) {
            this.cameraMatrix = cameraMatrix;
            this.distortionCoefficients = distortionCoefficients;
            this.rotationVectors = rotationVectors;
            this.translationVectors = translationVectors;
        }

        @Override
        public String toString() {
            return cameraMatrix.dump() + System.lineSeparator()
                    + distortionCoefficients.dump() + System.lineSeparator();
        }
    }

    public static CalibrationResults calibrate(boolean saveToFile) {
        List<Mat> calibrationImages = loadCalibrationImages();
        Mat objectPoints = new Mat(CHESSBOARD_SIZE, CvType.CV_32FC3);
        for (int i = 0; i < (int) CHESSBOARD_SIZE.area(); i++) {
            int y = (int) Math.round(i / CHESSBOARD_SIZE.width);
            int x = (int) Math.round(i % CHESSBOARD_SIZE.width);

            objectPoints.put(y, x, new float[] { x * SQUARE_SIZE, y * SQUARE_SIZE, 0 });
        }

        List<Mat> imagePointsList = new ArrayList<>();
        Size imageSize = new Size();


        for (Mat image : calibrationImages) {
            if (gray == null) {
                gray = new Mat(image.rows(), image.cols(), CvType.CV_8UC1);
                imageSize = image.size();
            }

            // Convert to B+W (requirement for findChessboardCorners)
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);

            // Checks if calibration was successful, then refines image points and before adding
            // them to the calibration data
            if (Calib3d.findChessboardCorners(image, CHESSBOARD_SIZE, imagePoints)) {
                Imgproc.cornerSubPix(gray, imagePoints, new Size(11, 11), new Size(-1, -1), criteria);

                // reference concern -- could be all pointing to same object?
                imagePointsList.add(imagePoints);

                Calib3d.drawChessboardCorners(image, CHESSBOARD_SIZE, imagePoints, true);
                HighGui.imshow("Chessboard Calibration", image);
                HighGui.waitKey(500);
            }

            gray.release();
        }

        HighGui.destroyAllWindows();

        List<Mat> objectPointsList = Collections.nCopies(imagePointsList.size(), objectPoints);
        Calib3d.calibrateCamera(objectPointsList, imagePointsList, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);

        CalibrationResults results = new CalibrationResults(cameraMatrix, distCoeffs, rvecs, tvecs);

        System.out.println(results.toString());

        if (!saveToFile)
            return results;
        try {
            Files.writeString(Paths.get("calibration_output"), results.toString(), StandardOpenOption.CREATE);
        }
        catch (IOException e) {
            System.err.println("Failed to write calibration output to file");
        }

        return results;
    }
}
