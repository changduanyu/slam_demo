#include <iostream>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <pangolin/pangolin.h>

cv::Mat DrawEpilines(const cv::Mat &img1, const std::vector<cv::Point2f> &pt1, const cv::Mat &F, 
                    cv::Mat &img2, const std::vector<cv::Point2f> &pt2) {

    cv::Mat output_img(img1.rows, img1.cols*2, CV_8UC3); 
    std::vector<cv::Vec3f> lines1, lines2;
        
    cv::computeCorrespondEpilines(pt1, 1, F, lines1);
    cv::computeCorrespondEpilines(pt2, 2, F, lines2);
    int rows = img1.rows, cols = img1.cols;
    cv::Rect rect1(0, 0, cols, rows);
    cv::Rect rect2(cols, 0, cols, rows);
    img1.copyTo(output_img(rect1));
    img2.copyTo(output_img(rect2));
    
    cv::RNG rng(0);
    for (int i=0; i<pt1.size(); i++) {
        cv::Scalar color(rng(256), rng(256), rng(256));

        cv::circle(output_img(rect1), pt1[i], 5, color, 3);
        cv::Point left_line_pa(0, -lines2[i][2]/lines2[i][1]);
        cv::Point left_line_pb(cols,-(lines2[i][2] + lines2[i][0]*cols)/lines2[i][1]);
        // std::cout << "left_line_pa" << left_line_pa << std::endl;
        // std::cout << "left_line_pb" << left_line_pb << std::endl;            
        cv::line(output_img(rect1), left_line_pa, left_line_pb, color, 1);

        cv::Point right_line_pa(0, -lines1[i][2]/lines1[i][1]);
        cv::Point right_line_pb(cols,-(lines1[i][2] + lines1[i][0]*cols)/lines1[i][1]);
        // std::cout << "right_line_pa" << right_line_pa << std::endl;
        // std::cout << "right_line_pb" << right_line_pb << std::endl;

        cv::circle(output_img(rect2), pt2[i], 5, color, 3);
        cv::line(output_img(rect2), right_line_pa, right_line_pb, color, 1);            
    }

    return output_img.clone();
}

void Triangulate(const cv::Point2f &pt1, const cv::Point2f &pt2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &X) {
    cv::Mat A(4, 4, CV_64F);
    A.row(0) = pt1.x * P1.row(2) - P1.row(0);
    A.row(1) = pt1.y * P1.row(2) - P1.row(1);
    A.row(2) = pt2.x * P2.row(2) - P2.row(0);
    A.row(3) = pt2.y * P2.row(2) - P2.row(1);

    cv::Mat u, w, vt;
    cv::SVD::compute(A, w, u, vt);
    X = vt.row(3).t();
    X = X.rowRange(0,3) / X.at<double>(3);
    X = X / cv::norm(X);
}

int CheckGoodPoint(const std::vector<cv::Point2f> &pt1, const std::vector<cv::Point2f> &pt2, const cv::Mat &K,
                    const cv::Mat &P1, const cv::Mat &P2) {
    int num_goods=0;
    cv::Mat P1_norm = K * P1;
    cv::Mat P2_norm = K * P2;
    for (int i=0; i<pt1.size(); i++) {
        cv::Mat X, X_prime;
        // std::cout << "round: " << i << std::endl;
        // std::cout << "pt1" << pt1[i] << std::endl;
        // std::cout << "pt2" << pt2[i] << std::endl;
        Triangulate(pt1[i], pt2[i], P1_norm, P2_norm, X);
        // std::cout << "triangulated X" << X << std::endl;

        if (X.at<double>(0) > 100 || X.at<double>(1) > 100 || X.at<double>(2) > 100)
            continue;
        if (X.at<double>(2) <= 0)
            continue;
        X_prime = P2.colRange(0,3) * X + P2.col(3);
        // std::cout << "X " << X << std::endl;
        // std::cout << "X prime " << X_prime << std::endl;

        if (X_prime.at<double>(0) > 100 || X_prime.at<double>(1) > 100 || X_prime.at<double>(2) > 100)
            continue;
        if (X_prime.at<double>(2) <= 0)
            continue;
        num_goods++;
    }
    return num_goods;
}

void DecomposeEssentialMatrix(const std::vector<cv::Point2f> &pt1, 
                        const std::vector<cv::Point2f> &pt2, 
                        const cv::Mat &E, 
                        const cv::Mat &K, 
                        cv::Mat &R,
                        cv::Mat &t) {
    // svd
    cv::Mat u,w,vt;
    cv::SVD::compute(E, w, u, vt);
    // std::cout << "u: " << u*u.t() << std::endl;
    // std::cout << "u determinant: " << cv::determinant(u) << std::endl;
    // std::cout << "vt: " << vt.t() * vt << std::endl;
    // std::cout << "vt determinant: " << cv::determinant(vt) << std::endl;
    // std::cout << "w: " << w << std::endl;

    // check equality
    double ratio = fabs(w.at<double>(0) / w.at<double>(1));
    if (ratio > 1.0) ratio = 1.0 / ratio;
    if (ratio <0.7) {
        std::cout << "sigular values are too far away: " << ratio << std::endl;
        return;
    } 
    // std::cout << "ratio: " << ratio << std::endl;
    cv::Mat W = (cv::Mat_<double>(3,3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
    cv::Mat R0 = u * W * vt;
    if (cv::determinant(R0) < 0)
        R0 = -R0;
    cv::Mat R1 = u * W.t() * vt;
    if (cv::determinant(R1) < 0)
        R1 = -R1;
    cv::Mat t0 = u.col(2);
    cv::Mat t1 = -u.col(2);

    // test four solutions
    cv::Mat P0 = cv::Mat::eye(3,4,R0.type());
    std::vector<cv::Mat> candidates;
    for (int i=0; i<4; i++)
        candidates.push_back(cv::Mat(3,4,R0.type()));
    R0.copyTo(candidates[0].colRange(0,3));t0.copyTo(candidates[0].col(3));
    R0.copyTo(candidates[1].colRange(0,3));t1.copyTo(candidates[1].col(3));
    R1.copyTo(candidates[2].colRange(0,3));t0.copyTo(candidates[2].col(3));
    R1.copyTo(candidates[3].colRange(0,3));t1.copyTo(candidates[3].col(3));
    
    // cv::Mat P1(3,4,R0.type()), P2(3,4,R0.type()), P3(3,4,R0.type()), P4(3,4,R0.type());    
    // std::cout << "R0: " << R0 << std::endl;
    // std::cout << "R1: " << R1 << std::endl;
    // std::cout << "t: " << t0 << std::endl;
    // std::cout << "P1: " << candidates[0] << std::endl;
    // std::cout << "P2: " << candidates[1] << std::endl;
    // std::cout << "P3: " << candidates[2] << std::endl;
    // std::cout << "P4: " << candidates[3] << std::endl;
    
    std::vector<int> num_good(4,0);
    for (int i=0; i<4; i++) {            
        num_good[i] = CheckGoodPoint(pt1, pt2, K, P0, candidates[i]);
        // std::cout << "num_good" << i << ": " << num_good[i] << std::endl;
    }

    std::vector<int>::iterator result = std::max_element(num_good.begin(), num_good.end());
    int max_position = std::distance(num_good.begin(), result);
    R = candidates[max_position].colRange(0,3).clone();
    t = candidates[max_position].col(3).clone();

    // cv::Mat R, trans;
    // double focal = 517.3;
    // cv::Point2d pp(318.6, 255.3); 
    // cv::recoverPose(E, pt1, pt2, R, trans, focal, pp);
    // cv::Mat gt(3, 4, R.type());
    // gt.colRange(0,3) = R * 1.0;
    // gt.col(3) = trans * 1.0;
    // std::cout << "estimated: " << camera_matrix << std::endl;
    // std::cout << "gt: " << gt << std::endl;
}

void TriangulateUnseenPoints(const std::vector<cv::Point2f> &pt1,
                            const std::vector<cv::Point2f> &pt2,
                            std::vector<cv::Point3f> &point_cloud, 
                            const cv::Mat &last_pose,
                            const cv::Mat &current_pose,
                            const cv::Mat &K,
                            int widht,
                            int height) {
    // reproject
    std::vector<cv::Point2f> reproject_points;

    for (int i=0; i<point_cloud.size(); i++) {
        cv::Mat homogeneous_point = (cv::Mat_<double>(4, 1) << point_cloud[i].x, point_cloud[i].y, point_cloud[i].z, 1.0);
        std::cout << "homogeneous_point : " << point_cloud[i] << std::endl;
        std::cout << "homogeneous_point : " << homogeneous_point << std::endl;
        cv::Mat reproject_point = K * current_pose.rowRange(0,3) * homogeneous_point;
        reproject_point = reproject_point.rowRange(0,3) / reproject_point.at<double>(3);    
        reproject_points.push_back(cv::Point2f(reproject_point.at<double>(0), reproject_point.at<double>(1)));  
        std::cout << "reproject point: " << reproject_point << std::endl;          
    }
       
    std::vector<bool> unseen(pt2.size(), true);
    for (int i=0; i<pt2.size(); i++) {
        for (int j=0; j<reproject_points.size(); j++) {    
            double dist = cv::norm(pt2[i] - reproject_points[j]);
            if (dist < 2) {
                unseen[i] = false;
                break;
            }                
        }                
    }

    for (int i=0; i<pt2.size(); i++) {
        if (!unseen[i])
            continue;
        cv::Mat X;
        Triangulate(pt1[i], pt2[i], last_pose, current_pose, X);
        point_cloud.push_back(cv::Point3f(X.at<double>(0), X.at<double>(1), X.at<double>(2)));
    }
}

int main (int argc, char **argv) {
    cv::VideoCapture cap("../videos/test_freiburgxyz525.mp4");
    if (!cap.isOpened()) {
        std::cout << "video capture not open" << std::endl;
        return -1;
    }
    cv::Mat last_frame;
    std::vector<cv::KeyPoint> last_key_points;
    cv::Mat last_descriptors;
    bool first_frame = true;

    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    double min_threshold = 15;
    int image_index = 0;
    cv::Mat K = (cv::Mat_<double>(3,3)<<517.3, 0.0, 318.6, 0.0, 516.5, 255.3, 0.0, 0.0, 1);
    std::cout << "K" << K << std::endl;

    std::vector<cv::Mat> pose_history;
    std::vector<cv::Point3f> point_cloud;
    std::vector<cv::Mat> descriptor_history;
    std::vector<std::vector<cv::KeyPoint>> key_point_history;
    cv::Mat last_pose(4,4,CV_64F);
    int width, height;

    while (true) {
        cv::Mat original_frame;
        cv::Mat frame;
        cv::Mat current_pose = cv::Mat::eye(4, 4, CV_64F);
        std::vector<cv::KeyPoint> key_points;
        cv::Mat descriptors;
        std::vector<std::vector<cv::DMatch> > matches;

        cap >> original_frame;
        // std::cout << "frame size: " << frame.size() << std::endl;
        // cv::resize(original_frame, frame, cv::Size(), 0.5, 0.5);
        frame = original_frame.clone(); 
        orb->detectAndCompute(frame, cv::Mat(), key_points, descriptors);
        
        if (first_frame) {
            first_frame = false;
            last_frame = frame;
            last_descriptors = descriptors;
            last_key_points = key_points; 
            pose_history.push_back(current_pose.clone());
            descriptor_history.push_back(descriptors.clone());
            key_point_history.push_back(key_points);
            last_pose = current_pose.clone();
            cv::Size frame_size = frame.size();
            width = frame_size.width;
            height = frame_size.height;
            continue;
        }

        // matcher
        matcher->knnMatch(last_descriptors, descriptors, matches, 3);

        // matched points
        std::vector<cv::Point2f> points_1, points_2;
        std::vector<cv::DMatch> point_matches;
        for (int i=0; i<matches.size(); i++) {        
            point_matches.push_back(matches[i][0]);
            points_1.push_back(last_key_points[matches[i][0].queryIdx].pt);
            points_2.push_back(key_points[matches[i][0].trainIdx].pt);
        }
        
        std::vector<uchar> status(points_1.size());
        cv::Mat F = cv::findFundamentalMat(points_1, points_2, CV_FM_RANSAC, 2, 0.99, status);

        std::vector<cv::DMatch> inlier_matches;
        std::vector<cv::Point2f> inlier_1, inlier_2;
        for (int i=0; i<points_1.size(); i++) {
            if (status[i]) {
                inlier_matches.push_back(point_matches[i]);
                inlier_1.push_back(points_1[i]);
                inlier_2.push_back(points_2[i]);
            }
        } 
        
        // check parallax
        double average_parallax;
        double sum_parallax = 0;
        for (int i=0; i<inlier_1.size(); i++) {
            double parallax = cv::norm(inlier_1[i] - inlier_2[i]);
            // std::cout << "parallax: " << parallax << std::endl;
            sum_parallax += parallax;
        }
        average_parallax = sum_parallax / static_cast<double>(inlier_1.size());
        // std::cout << "average parallax: " << average_parallax << std::endl;
        if (average_parallax < 10) 
            continue;

        // essential matrix
        cv::Mat E = K.t() * F * K;
        if (cv::determinant(E) > 1e-7) {
            std::cout << "determinant(E)!=0: " << cv::determinant(E) << std::endl;
            continue;
        }
        // std::cout << "determinant(E): " << cv::determinant(E) << std::endl;
        // decompose [R t] from E
        cv::Mat R, t;
        DecomposeEssentialMatrix(inlier_1, inlier_2, E, K, R, t);
                
        R.copyTo(current_pose.colRange(0,3).rowRange(0,3));
        t.copyTo(current_pose.col(3).rowRange(0,3));
        
        current_pose *= last_pose;
        // std::cout << "R type: " << R.type() << std::endl;
        // reproject point cloud into the current frame, triangulate unseen points  
        TriangulateUnseenPoints(inlier_1, inlier_2, point_cloud, last_pose, current_pose, K, width, height); 

        cv::Mat output_img = DrawEpilines(last_frame, inlier_1, F, frame, inlier_2);
        cv::imwrite("../images/epiline"+std::to_string(image_index++)+".jpg", output_img);
        cv::Mat img_matches;
        cv::drawMatches(last_frame, last_key_points, frame, key_points, inlier_matches, img_matches, cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0));          
        
        last_frame = frame;
        last_descriptors = descriptors;
        last_key_points = key_points;
        last_pose = current_pose;
        
        // cv::imshow("image matches", output_img);
        // cv::imshow("image matches", output_img);
        if (cv::waitKey(30) > 0) break;
    }
    std::cout << "hello world" << std::endl;
    return 0;
}