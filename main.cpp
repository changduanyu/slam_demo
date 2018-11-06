#include <iostream>
#include <string>

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

    while (true) {
        cv::Mat original_frame;
        cv::Mat frame;
        std::vector<cv::KeyPoint> key_points;
        cv::Mat descriptors;
        std::vector<std::vector<cv::DMatch> > matches;

        cap >> original_frame;
        frame = original_frame.clone();
        // std::cout << "frame size: " << frame.size() << std::endl;
        // cv::resize(original_frame, frame, cv::Size(), 0.5, 0.5);
        orb->detectAndCompute(frame, cv::Mat(), key_points, descriptors);
        if (first_frame) {
            first_frame = false;
            last_frame = frame;
            last_descriptors = descriptors;
            last_key_points = key_points;
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

        // check inliers
        // for (int i=0; i<matches.size(); i++) {
        //     int num_matches = static_cast<int>(matches[i].size());
        //     double min_dist = matches[i][0].distance;
        //     // std::cout << "min distance: " << min_dist << std::endl;
        //     if (min_dist < min_threshold) {
        //         inlier_matches.push_back(matches[i][0]);
        //         inlier_1.push_back(last_key_points[matches[i][0].queryIdx]);
        //         inlier_2.push_back(key_points[matches[i][0].trainIdx]);
        //     }
            
        // }
        
        // std::cout << "match size: " << matches.size() << std::endl;
        // std::cout << "inlier matches size : " << inlier_matches.size() << std::endl;

        // cv::KeyPoint::convert(inlier_1, points_1);
        // cv::KeyPoint::convert(inlier_2, points_2);
        
        // for (int i=0; i<inlier_1.size(); i++) {
        //     std::cout << "points 1 : " << inlier_1[i].pt << std::endl;
        //     std::cout << "points 2 : " << inlier_2[i].pt << std::endl;
        // }
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
        // std::cout << "points_1 size: " << points_1.size() << std::endl;
        // std::cout << "inlier_1 size: " << inlier_1.size() << std::endl;
        // std::cout << "F: " << F << std::endl;
        
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
        std::cout << "determinant(E): " << cv::determinant(E) << std::endl;
        // decompose [R t] from E
        // svd
        cv::SVD svd(E, cv::SVD::MODIFY_A);
        cv::Mat svd_u = svd.u, svd_vt = svd.vt, svd_w = svd.w;
        // check equality
        double ratio = fabs(svd_w.at<double>(0) / svd_w.at<double>(1));
        if (ratio > 1.0) ratio = 1.0 / ratio;
        if (ratio <0.7) {
            std::cout << "sigular values are too far away: " << ratio << std::endl;
            continue;
        } 
        // TODO: compute average of svd_w, compute E, decopose E agian

        cv::Mat w = (cv::Mat_<double>(3,3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
        cv::Mat R0 = svd_u * w * svd_vt;
        cv::Mat R1 = svd_u * w.t() * svd_vt;
        cv::Mat t0 = svd_u.col(2);
        cv::Mat t1 = -svd_u.col(2);

        // test four solutions
        




        cv::Mat output_img = DrawEpilines(last_frame, inlier_1, F, frame, inlier_2);
        cv::imwrite("../images/epiline"+std::to_string(image_index++)+".jpg", output_img);

        cv::Mat img_matches;
        cv::drawMatches(last_frame, last_key_points, frame, key_points, inlier_matches, img_matches, cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0));
        
        last_frame = frame;
        last_descriptors = descriptors;
        last_key_points = key_points;
        
        // cv::imshow("image matches", output_img);
        // cv::imshow("image matches", output_img);
        if (cv::waitKey(30) > 0) break;
    }
    std::cout << "hello world" << std::endl;
    return 0;
}