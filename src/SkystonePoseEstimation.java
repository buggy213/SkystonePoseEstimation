import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class SkystonePoseEstimation {

    private static Mat morphKernel;
    private static Size morphKernelSize = new Size(5, 5);
    private static List<MatOfPoint> contours = new ArrayList<>();

    // Unused
    private static Mat hierarchy = new Mat();

    public static void init() {
        morphKernel = new Mat(morphKernelSize, CvType.CV_8UC1, Scalar.all(1));
    }

    public static void detect(Mat img) {
        // First, find approximate locations of skystones through contour / thresholding

        Mat workingImage = img.clone();
        Mat blurredImage = img.clone();

        // Do a bilateral filter to hopefully reduce noise while maintaining sharp edges
        // TODO: tune the sigma values
        Imgproc.bilateralFilter(workingImage, blurredImage, 5, 50, 50);

        Imgproc.cvtColor(blurredImage, blurredImage, Imgproc.COLOR_BGR2HSV);

        // Threshold to gold areas (mask is grayscale image containing these areas)
        Mat mask = new Mat(blurredImage.size(), CvType.CV_8UC1);
        Threshold gold = Threshold.gold;
        gold.threshold(blurredImage, mask);

        // 5x5 kernel for opening (erode -> dilate, helps reduce noise)

        Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, morphKernel);

        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);

        // Filter off contours below a minimum size
        

        mask.release();
        workingImage.release();
        blurredImage.release();
    }
}
