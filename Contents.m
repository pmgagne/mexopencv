% mexopencv
% Version 3.4.12 (R2020b) 23-October-2020
%
%% opencv:
%
% calib3d:
%   cv.RQDecomp3x3                                       - Computes an RQ decomposition of 3x3 matrices
%   cv.Rodrigues                                         - Converts a rotation matrix to a rotation vector or vice versa
%   cv.StereoBM                                          - Class for computing stereo correspondence using the block matching algorithm
%   cv.StereoSGBM                                        - Class for computing stereo correspondence using the semi-global block matching algorithm
%   cv.calibrateCamera                                   - Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern
%   cv.calibrationMatrixValues                           - Computes useful camera characteristics from the camera matrix
%   cv.composeRT                                         - Combines two rotation-and-shift transformations
%   cv.computeCorrespondEpilines                         - For points in an image of a stereo pair, computes the corresponding epilines in the other image
%   cv.convertPointsFromHomogeneous                      - Converts points from homogeneous to Euclidean space
%   cv.convertPointsToHomogeneous                        - Converts points from Euclidean to homogeneous space
%   cv.correctMatches                                    - Refines coordinates of corresponding points
%   cv.decomposeEssentialMat                             - Decompose an essential matrix to possible rotations and translation
%   cv.decomposeHomographyMat                            - Decompose a homography matrix to rotation(s), translation(s) and plane normal(s)
%   cv.decomposeProjectionMatrix                         - Decomposes a projection matrix into a rotation matrix and a camera matrix
%   cv.drawChessboardCorners                             - Renders the detected chessboard corners
%   cv.estimateAffine2D                                  - Computes an optimal affine transformation between two 2D point sets
%   cv.estimateAffine3D                                  - Computes an optimal affine transformation between two 3D point sets
%   cv.estimateAffinePartial2D                           - Computes an optimal limited affine transformation with 4 degrees of freedom between two 2D point sets
%   cv.filterSpeckles                                    - Filters off small noise blobs (speckles) in the disparity map
%   cv.find4QuadCornerSubpix                             - Finds subpixel-accurate positions of the chessboard corners
%   cv.findChessboardCorners                             - Finds the positions of internal corners of the chessboard
%   cv.findCirclesGrid                                   - Finds the centers in the grid of circles
%   cv.findEssentialMat                                  - Calculates an essential matrix from the corresponding points in two images
%   cv.findFundamentalMat                                - Calculates a fundamental matrix from the corresponding points in two images
%   cv.findHomography                                    - Finds a perspective transformation between two planes
%   cv.fisheyeCalibrate                                  - Performs camera calibration (fisheye)
%   cv.fisheyeDistortPoints                              - Distorts 2D points using fisheye model
%   cv.fisheyeEstimateNewCameraMatrixForUndistortRectify - Estimates new camera matrix for undistortion or rectification (fisheye)
%   cv.fisheyeInitUndistortRectifyMap                    - Computes undistortion and rectification maps (fisheye)
%   cv.fisheyeProjectPoints                              - Projects points using fisheye model
%   cv.fisheyeStereoCalibrate                            - Performs stereo calibration (fisheye)
%   cv.fisheyeStereoRectify                              - Stereo rectification for fisheye camera model
%   cv.fisheyeUndistortImage                             - Transforms an image to compensate for fisheye lens distortion
%   cv.fisheyeUndistortPoints                            - Undistorts 2D points using fisheye model
%   cv.getOptimalNewCameraMatrix                         - Returns the new camera matrix based on the free scaling parameter
%   cv.getValidDisparityROI                              - Computes valid disparity ROI from the valid ROIs of the rectified images
%   cv.initCameraMatrix2D                                - Finds an initial camera matrix from 3D-2D point correspondences
%   cv.matMulDeriv                                       - Computes partial derivatives of the matrix product for each multiplied matrix
%   cv.projectPoints                                     - Projects 3D points to an image plane
%   cv.recoverPose                                       - Recover relative camera rotation and translation from an estimated essential matrix and the corresponding points in two images, using cheirality check
%   cv.rectify3Collinear                                 - Computes the rectification transformations for 3-head camera, where all the heads are on the same line
%   cv.reprojectImageTo3D                                - Reprojects a disparity image to 3D space
%   cv.sampsonDistance                                   - Calculates the Sampson Distance between two points
%   cv.solveP3P                                          - Finds an object pose from 3 3D-2D point correspondences
%   cv.solvePnP                                          - Finds an object pose from 3D-2D point correspondences
%   cv.solvePnPRansac                                    - Finds an object pose from 3D-2D point correspondences using the RANSAC scheme
%   cv.stereoCalibrate                                   - Calibrates the stereo camera
%   cv.stereoRectify                                     - Computes rectification transforms for each head of a calibrated stereo camera
%   cv.stereoRectifyUncalibrated                         - Computes a rectification transform for an uncalibrated stereo camera
%   cv.triangulatePoints                                 - Reconstructs points by triangulation
%   cv.validateDisparity                                 - Validates disparity using the left-right check
%
% core:
%   cv.ConjGradSolver                                    - Non-linear non-constrained minimization of a function with known gradient
%   cv.DownhillSolver                                    - Non-linear non-constrained minimization of a function
%   cv.FileStorage                                       - Reading from or writing to a XML/YAML/JSON file storage
%   cv.LDA                                               - Linear Discriminant Analysis
%   cv.LUT                                               - Performs a look-up table transform of an array
%   cv.Mahalanobis                                       - Calculates the Mahalanobis distance between two vectors
%   cv.PCA                                               - Principal Component Analysis class
%   cv.PSNR                                              - Computes the Peak Signal-to-Noise Ratio (PSNR) image quality metric
%   cv.Rect                                              - Class for 2D rectangles
%   cv.RotatedRect                                       - The class represents rotated (i.e. not up-right) rectangles on a plane
%   cv.SVD                                               - Singular Value Decomposition
%   cv.TickMeter                                         - A class to measure passing time
%   cv.Utils                                             - Utility and system information functions
%   cv.absdiff                                           - Calculates the per-element absolute difference between two arrays or between an array and a scalar
%   cv.add                                               - Calculates the per-element sum of two arrays or an array and a scalar
%   cv.addWeighted                                       - Calculates the weighted sum of two arrays
%   cv.batchDistance                                     - Naive nearest neighbor finder
%   cv.bitwise_and                                       - Calculates the per-element bit-wise conjunction of two arrays or an array and a scalar
%   cv.bitwise_not                                       - Inverts every bit of an array
%   cv.bitwise_or                                        - Calculates the per-element bit-wise disjunction of two arrays or an array and a scalar
%   cv.bitwise_xor                                       - Calculates the per-element bit-wise "exclusive or" operation on two arrays or an array and a scalar
%   cv.borderInterpolate                                 - Computes the source location of an extrapolated pixel
%   cv.calcCovarMatrix                                   - Calculates the covariance matrix of a set of vectors
%   cv.cartToPolar                                       - Calculates the magnitude and angle of 2D vectors
%   cv.compare                                           - Performs the per-element comparison of two arrays or an array and scalar value
%   cv.convertFp16                                       - Converts an array to half precision floating number
%   cv.convertScaleAbs                                   - Scales, calculates absolute values, and converts the result to 8-bit
%   cv.convertTo                                         - Converts an array to another data type with optional scaling
%   cv.copyMakeBorder                                    - Forms a border around an image
%   cv.copyTo                                            - Copies the matrix to another one
%   cv.dct                                               - Performs a forward or inverse discrete Cosine transform of 1D or 2D array
%   cv.dft                                               - Performs a forward or inverse Discrete Fourier transform of a 1D or 2D floating-point array
%   cv.divide                                            - Performs per-element division of two arrays or a scalar by an array
%   cv.eigen                                             - Calculates eigenvalues and eigenvectors of a symmetric matrix
%   cv.eigenNonSymmetric                                 - Calculates eigenvalues and eigenvectors of a non-symmetric matrix (real eigenvalues only)
%   cv.flip                                              - Flips a 2D array around vertical, horizontal, or both axes
%   cv.getBuildInformation                               - Returns OpenCV build information
%   cv.getOptimalDFTSize                                 - Returns the optimal DFT size for a given vector size
%   cv.glob                                              - Find all pathnames matching a specified pattern
%   cv.inRange                                           - Checks if array elements lie between the elements of two other arrays
%   cv.invert                                            - Finds the inverse or pseudo-inverse of a matrix
%   cv.kmeans                                            - Finds centers of clusters and groups input samples around the clusters
%   cv.magnitude                                         - Calculates the magnitude of 2D vectors
%   cv.mulSpectrums                                      - Performs the per-element multiplication of two Fourier spectrums
%   cv.multiply                                          - Calculates the per-element scaled product of two arrays
%   cv.norm                                              - Calculates absolute array norm, absolute difference norm, or relative difference norm
%   cv.normalize                                         - Normalizes the norm or value range of an array
%   cv.perspectiveTransform                              - Performs the perspective matrix transformation of vectors
%   cv.phase                                             - Calculates the rotation angle of 2D vectors
%   cv.polarToCart                                       - Calculates x and y coordinates of 2D vectors from their magnitude and angle
%   cv.rotate                                            - Rotates a 2D array in multiples of 90 degrees
%   cv.setRNGSeed                                        - Sets state of default random number generator
%   cv.solve                                             - Solves one or more linear systems or least-squares problems
%   cv.solveLP                                           - Solve given (non-integer) linear programming problem using the Simplex Algorithm
%   cv.subtract                                          - Calculates the per-element difference between two arrays or array and a scalar
%   cv.tempfile                                          - Return name of a temporary file
%   cv.transform                                         - Performs the matrix transformation of every array element
%
% dnn:
%   cv.Net                                               - Create and manipulate comprehensive artificial neural networks
%
% features2d:
%   cv.AGAST                                             - Detects corners using the AGAST algorithm
%   cv.AKAZE                                             - Class implementing the AKAZE keypoint detector and descriptor extractor
%   cv.AgastFeatureDetector                              - Wrapping class for feature detection using the AGAST method
%   cv.BOWImgDescriptorExtractor                         - Class to compute an image descriptor using the bag of visual words
%   cv.BOWKMeansTrainer                                  - KMeans-based class to train visual vocabulary using the bag of visual words approach
%   cv.BRISK                                             - Class implementing the BRISK keypoint detector and descriptor extractor
%   cv.DescriptorExtractor                               - Common interface of 2D image Descriptor Extractors
%   cv.DescriptorMatcher                                 - Common interface for matching keypoint descriptors
%   cv.FAST                                              - Detects corners using the FAST algorithm
%   cv.FastFeatureDetector                               - Wrapping class for feature detection using the FAST method
%   cv.FeatureDetector                                   - Common interface of 2D image Feature Detectors
%   cv.GFTTDetector                                      - Wrapping class for feature detection using the goodFeaturesToTrack function
%   cv.KAZE                                              - Class implementing the KAZE keypoint detector and descriptor extractor
%   cv.KeyPointsFilter                                   - Methods to filter a vector of keypoints
%   cv.MSER                                              - Maximally Stable Extremal Region extractor
%   cv.ORB                                               - Class implementing the ORB (oriented BRIEF) keypoint detector and descriptor extractor
%   cv.SimpleBlobDetector                                - Class for extracting blobs from an image
%   cv.computeRecallPrecisionCurve                       - Evaluate a descriptor extractor by computing precision/recall curve
%   cv.drawKeypoints                                     - Draws keypoints
%   cv.drawMatches                                       - Draws the found matches of keypoints from two images
%   cv.evaluateFeatureDetector                           - Evaluates a feature detector
%
% imgcodecs:
%   cv.imdecode                                          - Reads an image from a buffer in memory
%   cv.imencode                                          - Encodes an image into a memory buffer
%   cv.imread                                            - Loads an image from a file
%   cv.imreadmulti                                       - Loads a multi-page image from a file
%   cv.imwrite                                           - Saves an image to a specified file
%
% imgproc:
%   cv.CLAHE                                             - Contrast Limited Adaptive Histogram Equalization
%   cv.Canny                                             - Finds edges in an image using the Canny algorithm
%   cv.Canny2                                            - Finds edges in an image using the Canny algorithm with custom image gradient
%   cv.EMD                                               - Computes the "minimal work" distance between two weighted point configurations
%   cv.GaussianBlur                                      - Smooths an image using a Gaussian filter
%   cv.GeneralizedHoughBallard                           - Generalized Hough transform
%   cv.GeneralizedHoughGuil                              - Generalized Hough transform
%   cv.HoughCircles                                      - Finds circles in a grayscale image using the Hough transform
%   cv.HoughLines                                        - Finds lines in a binary image using the standard Hough transform
%   cv.HoughLinesP                                       - Finds line segments in a binary image using the probabilistic Hough transform
%   cv.HoughLinesPointSet                                - Finds lines in a set of points using the standard Hough transform
%   cv.HuMoments                                         - Calculates seven Hu invariants
%   cv.Laplacian                                         - Calculates the Laplacian of an image
%   cv.LineIterator                                      - Raster line iterator
%   cv.LineSegmentDetector                               - Line segment detector class
%   cv.Scharr                                            - Calculates the first x- or y- image derivative using Scharr operator
%   cv.Sobel                                             - Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator
%   cv.Subdiv2D                                          - Delaunay triangulation and Voronoi tessellation
%   cv.accumulate                                        - Adds an image to the accumulator image
%   cv.accumulateProduct                                 - Adds the per-element product of two input images to the accumulator
%   cv.accumulateSquare                                  - Adds the square of a source image to the accumulator image
%   cv.accumulateWeighted                                - Updates a running average
%   cv.adaptiveThreshold                                 - Applies an adaptive threshold to an array
%   cv.applyColorMap                                     - Applies a GNU Octave/MATLAB equivalent colormap on a given image
%   cv.approxPolyDP                                      - Approximates a polygonal curve(s) with the specified precision
%   cv.arcLength                                         - Calculates a contour perimeter or a curve length
%   cv.arrowedLine                                       - Draws an arrow segment pointing from the first point to the second one
%   cv.bilateralFilter                                   - Applies the bilateral filter to an image
%   cv.blendLinear                                       - Performs linear blending of two images
%   cv.blur                                              - Smooths an image using the normalized box filter
%   cv.boundingRect                                      - Calculates the up-right bounding rectangle of a point set
%   cv.boxFilter                                         - Blurs an image using the box filter
%   cv.boxPoints                                         - Finds the four vertices of a rotated rectangle
%   cv.buildPyramid                                      - Constructs the Gaussian pyramid for an image
%   cv.calcBackProject                                   - Calculates the back projection of a histogram
%   cv.calcHist                                          - Calculates a histogram of a set of arrays
%   cv.circle                                            - Draws a circle
%   cv.clipLine                                          - Clips the line against the image rectangle
%   cv.compareHist                                       - Compares two histograms
%   cv.connectedComponents                               - Computes the connected components labeled image of boolean image
%   cv.contourArea                                       - Calculates a contour area
%   cv.convertMaps                                       - Converts image transformation maps from one representation to another
%   cv.convexHull                                        - Finds the convex hull of a point set
%   cv.convexityDefects                                  - Finds the convexity defects of a contour
%   cv.cornerEigenValsAndVecs                            - Calculates eigenvalues and eigenvectors of image blocks for corner detection
%   cv.cornerHarris                                      - Harris corner detector
%   cv.cornerMinEigenVal                                 - Calculates the minimal eigenvalue of gradient matrices for corner detection
%   cv.cornerSubPix                                      - Refines the corner locations
%   cv.createHanningWindow                               - Computes a Hanning window coefficients in two dimensions
%   cv.cvtColor                                          - Converts an image from one color space to another
%   cv.cvtColorTwoPlane                                  - Dual-plane color conversion modes
%   cv.demosaicing                                       - Demosaicing algorithm
%   cv.dilate                                            - Dilates an image by using a specific structuring element
%   cv.distanceTransform                                 - Calculates the distance to the closest zero pixel for each pixel of the source image
%   cv.drawContours                                      - Draws contours outlines or filled contours
%   cv.drawMarker                                        - Draws a marker on a predefined position in an image
%   cv.ellipse                                           - Draws a simple or thick elliptic arc or fills an ellipse sector
%   cv.ellipse2Poly                                      - Approximates an elliptic arc with a polyline
%   cv.equalizeHist                                      - Equalizes the histogram of a grayscale image
%   cv.erode                                             - Erodes an image by using a specific structuring element
%   cv.fillConvexPoly                                    - Fills a convex polygon
%   cv.fillPoly                                          - Fills the area bounded by one or more polygons
%   cv.filter2D                                          - Convolves an image with the kernel
%   cv.findContours                                      - Finds contours in a binary image
%   cv.fitEllipse                                        - Fits an ellipse around a set of 2D points
%   cv.fitLine                                           - Fits a line to a 2D or 3D point set
%   cv.floodFill                                         - Fills a connected component with the given color
%   cv.getAffineTransform                                - Calculates an affine transform from three pairs of corresponding points
%   cv.getDefaultNewCameraMatrix                         - Returns the default new camera matrix
%   cv.getDerivKernels                                   - Returns filter coefficients for computing spatial image derivatives
%   cv.getFontScaleFromHeight                            - Calculates the font-specific size to use to achieve a given height in pixels
%   cv.getGaborKernel                                    - Returns Gabor filter coefficients
%   cv.getGaussianKernel                                 - Returns Gaussian filter coefficients
%   cv.getPerspectiveTransform                           - Calculates a perspective transform from four pairs of the corresponding points
%   cv.getRectSubPix                                     - Retrieves a pixel rectangle from an image with sub-pixel accuracy
%   cv.getRotationMatrix2D                               - Calculates an affine matrix of 2D rotation
%   cv.getStructuringElement                             - Returns a structuring element of the specified size and shape for morphological operations
%   cv.getTextSize                                       - Calculates the width and height of a text string
%   cv.goodFeaturesToTrack                               - Determines strong corners on an image
%   cv.grabCut                                           - Runs the GrabCut algorithm
%   cv.initUndistortRectifyMap                           - Computes the undistortion and rectification transformation map
%   cv.initWideAngleProjMap                              - Initializes maps for cv.remap for wide-angle
%   cv.integral                                          - Calculates the integral of an image
%   cv.intersectConvexConvex                             - Finds intersection of two convex polygons
%   cv.invertAffineTransform                             - Inverts an affine transformation
%   cv.isContourConvex                                   - Tests a contour convexity
%   cv.line                                              - Draws a line segment connecting two points
%   cv.linearPolar                                       - Remaps an image to polar coordinates space
%   cv.logPolar                                          - Remaps an image to semilog-polar coordinates space
%   cv.matchShapes                                       - Compares two shapes
%   cv.matchTemplate                                     - Compares a template against overlapped image regions
%   cv.medianBlur                                        - Blurs an image using the median filter
%   cv.minAreaRect                                       - Finds a rotated rectangle of the minimum area enclosing the input 2D point set
%   cv.minEnclosingCircle                                - Finds a circle of the minimum area enclosing a 2D point set
%   cv.minEnclosingTriangle                              - Finds a triangle of minimum area enclosing a 2D point set and returns its area
%   cv.moments                                           - Calculates all of the moments up to the third order of a polygon or rasterized shape
%   cv.morphologyEx                                      - Performs advanced morphological transformations
%   cv.phaseCorrelate                                    - Detect translational shifts that occur between two images
%   cv.pointPolygonTest                                  - Performs a point-in-contour test
%   cv.polylines                                         - Draws several polygonal curves
%   cv.preCornerDetect                                   - Calculates a feature map for corner detection
%   cv.putText                                           - Draws a text string
%   cv.pyrDown                                           - Blurs an image and downsamples it
%   cv.pyrMeanShiftFiltering                             - Performs initial step of meanshift segmentation of an image
%   cv.pyrUp                                             - Upsamples an image and then blurs it
%   cv.rectangle                                         - Draws a simple, thick, or filled up-right rectangle
%   cv.remap                                             - Applies a generic geometrical transformation to an image
%   cv.resize                                            - Resizes an image
%   cv.rotatedRectangleIntersection                      - Finds out if there is any intersection between two rotated rectangles
%   cv.sepFilter2D                                       - Applies a separable linear filter to an image
%   cv.spatialGradient                                   - Calculates the first order image derivative in both x and y using a Sobel operator
%   cv.sqrBoxFilter                                      - Calculates the normalized sum of squares of the pixel values overlapping the filter
%   cv.threshold                                         - Applies a fixed-level threshold to each array element
%   cv.undistort                                         - Transforms an image to compensate for lens distortion
%   cv.undistortPoints                                   - Computes the ideal point coordinates from the observed point coordinates
%   cv.warpAffine                                        - Applies an affine transformation to an image
%   cv.warpPerspective                                   - Applies a perspective transformation to an image
%   cv.watershed                                         - Performs a marker-based image segmentation using the watershed algorithm
%
% ml:
%   cv.ANN_MLP                                           - Artificial Neural Networks - Multi-Layer Perceptrons
%   cv.Boost                                             - Boosted tree classifier derived from cv.DTrees
%   cv.DTrees                                            - Decision Trees
%   cv.EM                                                - Expectation Maximization Algorithm
%   cv.KNearest                                          - The class implements K-Nearest Neighbors model
%   cv.LogisticRegression                                - Logistic Regression classifier
%   cv.NormalBayesClassifier                             - Bayes classifier for normally distributed data
%   cv.RTrees                                            - Random Trees
%   cv.SVM                                               - Support Vector Machines
%   cv.SVMSGD                                            - Stochastic Gradient Descent SVM classifier
%   cv.createConcentricSpheresTestSet                    - Creates test set
%   cv.randMVNormal                                      - Generates sample from multivariate normal distribution
%
% objdetect:
%   cv.CascadeClassifier                                 - Haar Feature-based Cascade Classifier for Object Detection
%   cv.DetectionBasedTracker                             - Detection-based tracker
%   cv.HOGDescriptor                                     - Histogram of Oriented Gaussian (HOG) descriptor and object detector
%   cv.SimilarRects                                      - Class for grouping object candidates, detected by Cascade Classifier, HOG etc.
%   cv.groupRectangles                                   - Groups the object candidate rectangles
%   cv.groupRectangles_meanshift                         - Groups the object candidate rectangles using meanshift
%
% photo:
%   cv.AlignMTB                                          - Aligns images of the same scene with different exposures
%   cv.CalibrateDebevec                                  - Camera Response Calibration algorithm
%   cv.CalibrateRobertson                                - Camera Response Calibration algorithm
%   cv.MergeDebevec                                      - Merge exposure sequence to a single image
%   cv.MergeMertens                                      - Merge exposure sequence to a single image
%   cv.MergeRobertson                                    - Merge exposure sequence to a single image
%   cv.Tonemap                                           - Tonemapping algorithm used to map HDR image to 8-bit range
%   cv.TonemapDrago                                      - Tonemapping algorithm used to map HDR image to 8-bit range
%   cv.TonemapDurand                                     - Tonemapping algorithm used to map HDR image to 8-bit range
%   cv.TonemapMantiuk                                    - Tonemapping algorithm used to map HDR image to 8-bit range
%   cv.TonemapReinhard                                   - Tonemapping algorithm used to map HDR image to 8-bit range
%   cv.colorChange                                       - Color Change
%   cv.decolor                                           - Transforms a color image to a grayscale image
%   cv.denoise_TVL1                                      - Primal-Dual algorithm to perform image denoising
%   cv.detailEnhance                                     - This filter enhances the details of a particular image
%   cv.edgePreservingFilter                              - Edge-preserving smoothing filter
%   cv.fastNlMeansDenoising                              - Image denoising using Non-local Means Denoising algorithm
%   cv.fastNlMeansDenoisingColored                       - Modification of fastNlMeansDenoising function for colored images
%   cv.fastNlMeansDenoisingColoredMulti                  - Modification of fastNlMeansDenoisingMulti function for colored images sequences
%   cv.fastNlMeansDenoisingMulti                         - Modification of fastNlMeansDenoising function for colored images sequences
%   cv.illuminationChange                                - Illumination Change
%   cv.inpaint                                           - Restores the selected region in an image using the region neighborhood
%   cv.pencilSketch                                      - Pencil-like non-photorealistic line drawing
%   cv.seamlessClone                                     - Seamless Cloning
%   cv.stylization                                       - Stylization filter
%   cv.textureFlattening                                 - Texture Flattening
%
% shape:
%   cv.EMDL1                                             - Computes the "minimal work" distance between two weighted point configurations
%   cv.HausdorffDistanceExtractor                        - A simple Hausdorff distance measure between shapes defined by contours
%   cv.ShapeContextDistanceExtractor                     - Implementation of the Shape Context descriptor and matching algorithm
%   cv.ShapeTransformer                                  - Base class for shape transformation algorithms
%
% stitching:
%   cv.Blender                                           - Class for all image blenders
%   cv.BundleAdjuster                                    - Class for all camera parameters refinement methods
%   cv.Estimator                                         - Rotation estimator base class
%   cv.ExposureCompensator                               - Class for all exposure compensators
%   cv.FeaturesFinder                                    - Feature finders class
%   cv.FeaturesMatcher                                   - Feature matchers class
%   cv.RotationWarper                                    - Rotation-only model image warper
%   cv.SeamFinder                                        - Class for all seam estimators
%   cv.Stitcher                                          - High level image stitcher
%   cv.Timelapser                                        - Timelapser class
%
% superres:
%   cv.SuperResolution                                   - Class for a whole family of Super Resolution algorithms
%
% video:
%   cv.BackgroundSubtractorKNN                           - K-nearest neighbours based Background/Foreground Segmentation Algorithm
%   cv.BackgroundSubtractorMOG2                          - Gaussian Mixture-based Background/Foreground Segmentation Algorithm
%   cv.CamShift                                          - Finds an object center, size, and orientation
%   cv.DualTVL1OpticalFlow                               - "Dual TV L1" Optical Flow Algorithm
%   cv.FarnebackOpticalFlow                              - Dense optical flow using the Gunnar Farneback's algorithm
%   cv.KalmanFilter                                      - Kalman filter class
%   cv.SparsePyrLKOpticalFlow                            - Class used for calculating a sparse optical flow
%   cv.buildOpticalFlowPyramid                           - Constructs the image pyramid which can be passed to cv.calcOpticalFlowPyrLK
%   cv.calcOpticalFlowFarneback                          - Computes a dense optical flow using the Gunnar Farneback's algorithm
%   cv.calcOpticalFlowPyrLK                              - Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids
%   cv.estimateRigidTransform                            - Computes an optimal affine transformation between two 2D point sets
%   cv.findTransformECC                                  - Finds the geometric transform (warp) between two images in terms of the ECC criterion
%   cv.meanShift                                         - Finds an object on a back projection image
%
% videoio:
%   cv.VideoCapture                                      - Class for video capturing from video files or cameras
%   cv.VideoWriter                                       - Video Writer class
%
% videostab:
%   cv.OnePassStabilizer                                 - A one-pass video stabilizer
%   cv.TwoPassStabilizer                                 - A two-pass video stabilizer
%   cv.calcBlurriness                                    - Calculate image blurriness
%   cv.estimateGlobalMotionLeastSquares                  - Estimates best global motion between two 2D point clouds in the least-squares sense
%   cv.estimateGlobalMotionRansac                        - Estimates best global motion between two 2D point clouds robustly (using RANSAC method)
%
%% opencv_contrib:
%
% aruco:
%   cv.boardDump                                         - Dump board (aruco)
%   cv.calibrateCameraAruco                              - Calibrate a camera using aruco markers
%   cv.calibrateCameraCharuco                            - Calibrate a camera using ChArUco corners
%   cv.detectCharucoDiamond                              - Detect ChArUco Diamond markers
%   cv.detectMarkers                                     - Basic ArUco marker detection
%   cv.dictionaryDump                                    - Dump dictionary (aruco)
%   cv.drawAxis                                          - Draw coordinate system axis from pose estimation
%   cv.drawCharucoBoard                                  - Draw a ChArUco board
%   cv.drawCharucoDiamond                                - Draw a ChArUco Diamond marker
%   cv.drawDetectedCornersCharuco                        - Draws a set of ChArUco corners
%   cv.drawDetectedDiamonds                              - Draw a set of detected ChArUco Diamond markers
%   cv.drawDetectedMarkers                               - Draw detected markers in image
%   cv.drawMarkerAruco                                   - Draw a canonical marker image
%   cv.drawPlanarBoard                                   - Draw a planar board
%   cv.estimatePoseBoard                                 - Pose estimation for a board of markers
%   cv.estimatePoseCharucoBoard                          - Pose estimation for a ChArUco board given some of their corners
%   cv.estimatePoseSingleMarkers                         - Pose estimation for single markers
%   cv.getBoardObjectAndImagePoints                      - Given a board configuration and a set of detected markers, returns the corresponding image points and object points to call solvePnP
%   cv.interpolateCornersCharuco                         - Interpolate position of ChArUco board corners
%   cv.refineDetectedMarkers                             - Refind not detected markers based on the already detected and the board layout
%
% bgsegm:
%   cv.BackgroundSubtractorCNT                           - Background subtraction based on counting
%   cv.BackgroundSubtractorGMG                           - Background Subtractor module
%   cv.BackgroundSubtractorGSOC                          - Background Subtraction implemented during GSOC
%   cv.BackgroundSubtractorLSBP                          - Background Subtraction using Local SVD Binary Pattern
%   cv.BackgroundSubtractorMOG                           - Gaussian Mixture-based Background/Foreground Segmentation Algorithm
%   cv.SyntheticSequenceGenerator                        - Synthetic frame sequence generator for testing background subtraction algorithms
%
% bioinspired:
%   cv.Retina                                            - A biological retina model for image spatio-temporal noise and luminance changes enhancement
%   cv.RetinaFastToneMapping                             - Class with tone mapping algorithm of Meylan et al. (2007)
%   cv.TransientAreasSegmentationModule                  - Class which provides a transient/moving areas segmentation module
%
% datasets:
%   cv.Dataset                                           - Class for working with different datasets
%
% dnn_objdetect:
%   cv.InferBbox                                         - Post-process DNN object detection model predictions
%
% dpm:
%   cv.DPMDetector                                       - Deformable Part-based Models (DPM) detector
%
% face:
%   cv.BIF                                               - Implementation of bio-inspired features (BIF)
%   cv.BasicFaceRecognizer                               - Face Recognition based on Eigen-/Fisher-faces
%   cv.Facemark                                          - Base class for all facemark models
%   cv.FacemarkKazemi                                    - Face Alignment
%   cv.LBPHFaceRecognizer                                - Face Recognition based on Local Binary Patterns
%
% hfs:
%   cv.HfsSegment                                        - Hierarchical Feature Selection for Efficient Image Segmentation
%
% img_hash:
%   cv.ImgHash                                           - Base class for Image Hashing algorithms
%
% line_descriptor:
%   cv.BinaryDescriptor                                  - Class implements both functionalities for detection of lines and computation of their binary descriptor
%   cv.BinaryDescriptorMatcher                           - BinaryDescriptor matcher class
%   cv.LSDDetector                                       - Line Segment Detector
%   cv.drawKeylines                                      - Draws keylines
%   cv.drawLineMatches                                   - Draws the found matches of keylines from two images
%
% optflow:
%   cv.DISOpticalFlow                                    - DIS optical flow algorithm
%   cv.GPCForest                                         - Implementation of the Global Patch Collider algorithm
%   cv.OpticalFlowPCAFlow                                - PCAFlow algorithm
%   cv.VariationalRefinement                             - Variational optical flow refinement
%   cv.calcGlobalOrientation                             - Calculates a global motion orientation in a selected region
%   cv.calcMotionGradient                                - Calculates a gradient orientation of a motion history image
%   cv.calcOpticalFlowDF                                 - DeepFlow optical flow algorithm implementation
%   cv.calcOpticalFlowSF                                 - Calculate an optical flow using "SimpleFlow" algorithm
%   cv.calcOpticalFlowSparseToDense                      - Fast dense optical flow based on PyrLK sparse matches interpolation
%   cv.readOpticalFlow                                   - Read a .flo file
%   cv.segmentMotion                                     - Splits a motion history image into a few parts corresponding to separate independent motions (for example, left hand, right hand)
%   cv.updateMotionHistory                               - Updates the motion history image by a moving silhouette
%   cv.writeOpticalFlow                                  - Write a .flo to disk
%
% plot:
%   cv.Plot2d                                            - Class to plot 2D data
%
% saliency:
%   cv.MotionSaliencyBinWangApr2014                      - A Fast Self-tuning Background Subtraction Algorithm for Motion Saliency
%   cv.ObjectnessBING                                    - The Binarized normed gradients algorithm for Objectness
%   cv.StaticSaliencyFineGrained                         - The Fine Grained Saliency approach for Static Saliency
%   cv.StaticSaliencySpectralResidual                    - The Spectral Residual approach for Static Saliency
%
% text:
%   cv.TextDetectorCNN                                   - Class providing functionality of text detection
%
% xfeatures2d:
%   cv.AffineFeature2D                                   - Class implementing affine adaptation for key points
%   cv.BoostDesc                                         - Class implementing BoostDesc (Learning Image Descriptors with Boosting)
%   cv.BriefDescriptorExtractor                          - Class for computing BRIEF descriptors
%   cv.DAISY                                             - Class implementing DAISY descriptor
%   cv.FASTForPointSet                                   - Estimates cornerness for pre-specified KeyPoints using the FAST algorithm
%   cv.FREAK                                             - Class implementing the FREAK (Fast Retina Keypoint) keypoint descriptor
%   cv.HarrisLaplaceFeatureDetector                      - Class implementing the Harris-Laplace feature detector
%   cv.LATCH                                             - Class for computing the LATCH descriptor
%   cv.LUCID                                             - Class implementing the Locally Uniform Comparison Image Descriptor
%   cv.MSDDetector                                       - Class implementing the MSD (Maximal Self-Dissimilarity) keypoint detector
%   cv.PCTSignatures                                     - Class implementing PCT (Position-Color-Texture) signature extraction
%   cv.PCTSignaturesSQFD                                 - Class implementing Signature Quadratic Form Distance (SQFD)
%   cv.SIFT                                              - Class for extracting keypoints and computing descriptors using the Scale Invariant Feature Transform (SIFT)
%   cv.SURF                                              - Class for extracting Speeded Up Robust Features from an image
%   cv.StarDetector                                      - The class implements the Star keypoint detector
%   cv.VGG                                               - Class implementing VGG (Oxford Visual Geometry Group) descriptor
%   cv.matchGMS                                          - GMS (Grid-based Motion Statistics) feature matching strategy
%
% ximgproc:
%   cv.AdaptiveManifoldFilter                            - Interface for Adaptive Manifold Filter realizations
%   cv.BrightEdges                                       - Bright edges detector
%   cv.ContourFitting                                    - Contour Fitting algorithm using Fourier descriptors
%   cv.DTFilter                                          - Interface for realizations of Domain Transform filter
%   cv.DisparityWLSFilter                                - Disparity map filter based on Weighted Least Squares filter
%   cv.EdgeAwareInterpolator                             - Sparse match interpolation algorithm
%   cv.EdgeBoxes                                         - Class implementing Edge Boxes algorithm
%   cv.FastGlobalSmootherFilter                          - Interface for implementations of Fast Global Smoother filter
%   cv.FastHoughTransform                                - Calculates 2D Fast Hough transform of an image
%   cv.FastLineDetector                                  - Class implementing the FLD (Fast Line Detector) algorithm
%   cv.GradientDeriche                                   - Applies Deriche filter to an image
%   cv.GradientPaillou                                   - Applies Paillou filter to an image
%   cv.GraphSegmentation                                 - Graph Based Segmentation algorithm
%   cv.GuidedFilter                                      - Interface for realizations of Guided Filter
%   cv.HoughPoint2Line                                   - Calculates coordinates of line segment corresponded by point in Hough space
%   cv.PeiLinNormalization                               - Calculates an affine transformation that normalize given image using Pei/Lin Normalization
%   cv.RidgeDetectionFilter                              - Ridge Detection Filter
%   cv.SelectiveSearchSegmentation                       - Selective search segmentation algorithm
%   cv.StructuredEdgeDetection                           - Class implementing edge detection algorithm
%   cv.SuperpixelLSC                                     - Class implementing the LSC (Linear Spectral Clustering) superpixels algorithm
%   cv.SuperpixelSEEDS                                   - Class implementing the SEEDS (Superpixels Extracted via Energy-Driven Sampling) superpixels algorithm
%   cv.SuperpixelSLIC                                    - Class implementing the SLIC (Simple Linear Iterative Clustering) superpixels algorithm
%   cv.anisotropicDiffusion                              - Performs anisotropic diffusion on an image
%   cv.bilateralTextureFilter                            - Applies the bilateral texture filter to an image
%   cv.covarianceEstimation                              - Computes the estimated covariance matrix of an image using the sliding window formulation
%   cv.jointBilateralFilter                              - Applies the joint bilateral filter to an image
%   cv.l0Smooth                                          - Global image smoothing via L0 gradient minimization
%   cv.niBlackThreshold                                  - Performs thresholding on input images using Niblack's technique or some of the popular variations it inspired
%   cv.rollingGuidanceFilter                             - Applies the rolling guidance filter to an image
%   cv.thinning                                          - Applies a binary blob thinning operation, to achieve a skeletization of the input image
%   cv.weightedMedianFilter                              - Applies weighted median filter to an image
%
% xobjdetect:
%   cv.WBDetector                                        - WaldBoost detector - Object Detection using Boosted Features
%
% xphoto:
%   cv.GrayworldWB                                       - Gray-world white balance algorithm
%   cv.LearningBasedWB                                   - More sophisticated learning-based automatic white balance algorithm
%   cv.SimpleWB                                          - Simple white balance algorithm
%   cv.applyChannelGains                                 - Implements an efficient fixed-point approximation for applying channel gains, which is the last step of multiple white balance algorithms
%   cv.bm3dDenoising                                     - Performs image denoising using the Block-Matching and 3D-filtering algorithm
%   cv.dctDenoising                                      - The function implements simple dct-based denoising
%   cv.inpaint2                                          - The function implements different single-image inpainting algorithms
%
