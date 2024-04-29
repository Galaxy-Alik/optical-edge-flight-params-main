import cv2
import numpy as np
from glob import glob
from pose_est import *
fx, fy, cx, cy = [718.8560, 718.8560, 607.1928, 185.2157]
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

class FeatureLocalization:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2

    def feature_detection(self, detector = "SIFT"):

        # Using SIFT to find the keypoints and decriptors in the images.
        if detector == "SIFT":
            sift = cv2.SIFT_create()
            baseImage_kp, baseImage_des = sift.detectAndCompute(cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY), None)
            secImage_kp, secImage_des = sift.detectAndCompute(cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY), None)
            return baseImage_kp, baseImage_des, secImage_kp, secImage_des

    def feature_match(self, des1, des2, matcher = "BF", threshold = 0.75):        
        
        # Using Brute Force matcher to find matches.
        BF_Matcher = cv2.BFMatcher()
        InitialMatches = BF_Matcher.knnMatch(des1, des2, k=2)

        # Applying ratio test and filtering out the good matches.
        GoodMatches = []
        for m, n in InitialMatches:
            if m.distance < threshold * n.distance:
                GoodMatches.append([m])
        return GoodMatches
    
    def estimatePose(self, matches, kp1, kp2):
        baseImage_pts, secImage_pts = [], []
        for match in matches:
            baseImage_pts.append(kp1[match[0].queryIdx].pt)
            secImage_pts.append(kp2[match[0].trainIdx].pt)
        E, mask = cv2.findEssentialMat(np.array(baseImage_pts), np.array(secImage_pts))
        pts1 = baseImage_pts
        pts2 = secImage_pts
        _, R, t, mask = cv2.recoverPose(E, np.array(baseImage_pts), np.array(secImage_pts))
        
        # get camera motion
        R = R.transpose()
        t = np.matmul(R, t)

        return R, t


class Stitcher:
    def __init__(self, img1, img2, detector, matcher, threshold):
        self.img1 = img1
        self.img2 = img2
        self.detector = detector
        self.matcher = matcher
        self.threshold = threshold

    def findHomography(self, matches, kp1, kp2):
        
        #if match count < 4, cannot compute homography, exit code
        if len(matches) < 4:
            print("Not enough matches found between the images to compute homography")
            exit(0)

        # Storing coordinates of points corresponding to the matches found in both the images
        baseImage_pts = []
        secImage_pts = []
        for match in matches:
            baseImage_pts.append(kp1[match[0].queryIdx].pt)
            secImage_pts.append(kp2[match[0].trainIdx].pt)

        # Changing the datatype to "float32" for finding homography
        baseImage_pts = np.float32(baseImage_pts)
        secImage_pts = np.float32(secImage_pts)

        # Finding the homography matrix(transformation matrix) for selected keypoints
        (homographyMatrix, status) = cv2.findHomography(secImage_pts, baseImage_pts, cv2.RANSAC, 4.0)
        return homographyMatrix, status
    

    def getNewFrameSizeAndMatrix(self, homographymatrix, shape_img2, shape_img1):
        (h, w) = shape_img2

        # Taking the matrix of initial coordinates of the corners of the secondary image
        # Stored in the following format: [[x1, x2, x3, x4], [y1, y2, y3, y4], [1, 1, 1, 1]] clockwise rot
        # Where (xi, yi) is the coordinate of the i th corner of the image. 
        initialmatrix = np.array([[0, w - 1, w - 1, 0],
                              [0, 0, h - 1, h - 1],
                              [1, 1, 1, 1]])

        # Finding the final coordinates of the corners of the image after transformation.
        # NOTE: Here, the coordinates of the corners of the frame may go out of the 
        # frame(negative values). We will correct this afterwards by updating the 
        # homography matrix accordingly.
        finalmatrix = np.dot(homographymatrix, initialmatrix)
        [x, y, c] = finalmatrix
        x = np.divide(x, c)
        y = np.divide(y, c)

        # Finding the dimentions of the stitched image frame and the "Correction" factor
        min_x, max_x = int(round(min(x))), int(round(max(x)))
        min_y, max_y = int(round(min(y))), int(round(max(y)))

        modW, modH = max_x, max_y
        correction = [0,0]
        if min_x < 0:
            modW -= min_x
            correction[0] = abs(min_x)

        if min_y < 0:
            modH -= min_y
            correction[1] = abs(min_y)

        # Correction of dimensions, Helpful when img2 image is overlaped on the left hand side of the img1.
        if modW < shape_img1[1] + correction[0]:
            modW = shape_img1[1] + correction[0]
        if modH < shape_img1[0] + correction[1]:
            modH = shape_img1[0] + correction[1]

        # Finding the coordinates of the corners of the image if they all were within the frame.
        x = np.add(x, correction[0])
        y = np.add(y, correction[1])
        
        initialPts= np.float32([[0, 0],
                                    [w - 1, 0],
                                    [w - 1, h - 1],
                                    [0, h - 1]])
        finalPts = np.float32(np.array([x, y]).transpose())
        # Updating the homography matrix. Done so that now the secondary image completely lies inside the frame
        homographyMatrix = cv2.getPerspectiveTransform(initialPts, finalPts)  
        return [modH, modW], correction, homographyMatrix

    
    def stitch_frames(self, prev_R, prev_t, fx, cx, cy):
        feat = FeatureLocalization(self.img1, self.img2)
        baseImage_kp, baseImage_des, secImage_kp, secImage_des = feat.feature_detection(detector=self.detector)
        matches = feat.feature_match(baseImage_des, secImage_des, matcher = self.matcher, threshold=self.threshold)
        curr_R, curr_t = feat.estimatePose(matches, baseImage_kp, secImage_kp)

        #Find Homography
        homographyMatrix, status = self.findHomography(matches, baseImage_kp, secImage_kp)

        # Finding size of new frame of stitched images and updating the homography matrix 
        newFrameSize, correction, homographyMatrix = self.getNewFrameSizeAndMatrix(homographyMatrix, self.img2.shape[:2], self.img1.shape[:2])

        # Warping the images
        stitchedFrame = cv2.warpPerspective(self.img2, homographyMatrix, (newFrameSize[1], newFrameSize[0]))
        stitchedFrame[correction[1]:correction[1]+self.img1.shape[0], correction[0]:correction[0]+self.img1.shape[1]] = self.img1
        return stitchedFrame, curr_R, curr_t

    
   
if __name__ == "__main__":

    img_data_dir = "./test"
    img_list = glob(img_data_dir + '/*.jpg')
    img_list.sort()
    num_frames = len(img_list)
    detector = "SIFT"
    matcher = "BF" 
    threshold = 0.75
    trajMap = np.zeros((100, 100, 3), dtype=np.uint8)
    out_pose_file = 'traj_est.txt'

    
    for i in range(num_frames):
        if i == 0:
            result_img = cv2.imread(img_list[i])
            curr_R = np.eye(3)
            curr_t = np.array([0, 0, 0]).astype('float64')

        else:
            Image1 = result_img
            Image2 = cv2.imread(img_list[i])

            # Checking if images read
            if Image1 is None or Image2 is None:
                print("\nImages not read properly or does not exist.\n")
                exit(0)

            # Calling function for stitching images.
            stitcher = Stitcher(Image2, Image1, detector, matcher, threshold)
            result_img, R, t = stitcher.stitch_frames(curr_R, curr_t, fx, cx, cy)

            if i == 1:
                curr_R = R
                curr_t = t
            else:
                curr_R = np.matmul(prev_R, R)   
                curr_t = (np.matmul(prev_R, t)) + (prev_t)

            # save current pose
            [tx, ty, tz] = [curr_t[0], curr_t[1], curr_t[2]]
            rz, ry, rx, qw, qx, qy, qz = rot2quat(curr_R)
            
            with open(out_pose_file, 'a') as f:
                f.write('%f %f %f %f %f %f %f %f\n' % (0.0, tx, ty, tz, qx, qy, qz, qw))

            prev_R = curr_R
            prev_t = abs(curr_t)
            print(prev_t)
            # draw estimated trajectory (blue) and gt trajectory (red)
            offset_draw = (int(100/2))
            scale = 10
            cv2.circle(trajMap, (int(curr_t[0]) * scale + offset_draw, int(curr_t[2]) * scale + offset_draw), 1, (0,255,0), -1)
            cv2.imshow('Trajectory', cv2.resize(trajMap, (1000,1000)))
            cv2.waitKey(1)

            # Displaying the stitched images.
            cv2.imshow('stitched', cv2.resize(result_img ,(600, 600)))
            cv2.waitKey(1)

    cv2.imwrite('mosaic.png', result_img )      
    
