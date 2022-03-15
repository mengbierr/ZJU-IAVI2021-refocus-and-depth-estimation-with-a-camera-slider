#include <filesystem>
#include <fstream>
#include <numbers>

#include "opencv2/opencv.hpp"

#include "guidedfilter.h"

namespace {

constexpr double kAlpha = 0.7;
constexpr double kTau1 = 1.0;
constexpr double kTau2 = 3.0;
constexpr int kDelta = 2;
constexpr double kSigmaS = 5.0;
constexpr double kSigmaC = 1.0;

constexpr int kRadius = 12;
constexpr double kEps = 0.02;

struct CameraParam {
    double fx;
    double fy;
    double cx;
    double cy;
};

CameraParam CalibCamera(const std::string &images_dir, int chess_x, int chess_y, double scale_factor) {
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < chess_y; i++) {
        for (int j = 0; j < chess_x; j++) {
            objp.emplace_back(j, i, 0.0);
        }
    }

    std::vector<std::vector<cv::Point3f>> objpoints;
    std::vector<std::vector<cv::Point2f>> imgpoints;

    std::vector<std::filesystem::path> files;
    for (const auto &entry : std::filesystem::directory_iterator(images_dir)) {
        files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());

    cv::Mat gray;
    for (const auto &image_name : files) {
        cv::Mat frame = cv::imread(image_name.string());
        cv::resize(frame, frame, cv::Size(), scale_factor, scale_factor);
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        const bool success = cv::findChessboardCorners(gray, cv::Size(chess_x, chess_y), corners,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        if (success) {
            objpoints.push_back(objp);
            imgpoints.emplace_back(corners);
        } else {
            std::cerr << "Failed to find chessboard on '" << image_name << "'\n";
        }
    }

    if (imgpoints.empty()) {
        std::cerr << "ERROR: no chessboard is detected\n";
        exit(-1);
    }

    cv::Mat camera_matrix;
    cv::Mat camera_dist;
    cv::Mat rvecs;
    cv::Mat tvecs;
    cv::calibrateCamera(objpoints, imgpoints, cv::Size(gray.rows, gray.cols), camera_matrix, camera_dist, rvecs, tvecs);

    CameraParam param {
        .fx = camera_matrix.at<double>(0, 0),
        .fy = camera_matrix.at<double>(1, 1),
        .cx = camera_matrix.at<double>(0, 2),
        .cy = camera_matrix.at<double>(1, 2),
    };
    return param;
}

double ColorDiff(const cv::Vec3f &a, const cv::Vec3f &b) {
    return std::abs(a(0) - b(0)) + std::abs(a(1) - b(1)) + std::abs(a(2) - b(2));
}

double ColorDiff(const cv::Vec3d &a, const cv::Vec3d &b) {
    return std::abs(a(0) - b(0)) + std::abs(a(1) - b(1)) + std::abs(a(2) - b(2));
}

void WritePly(const cv::Mat &depth, const cv::Mat &image, const CameraParam &param, double scale, double offset) {
    const int width = image.cols;
    const int height = image.rows;

    std::ofstream fout("recon.ply");

    fout << "ply\nformat ascii 1.0\n";
    fout << "element vertex " << width * height << "\n";
    fout << "property float x\n";
    fout << "property float y\n";
    fout << "property float z\n";
    fout << "property uchar red\n";
    fout << "property uchar green\n";
    fout << "property uchar blue\n";
    fout << "end_header\n";

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            const double z = depth.at<double>(i, j) * scale + offset;
            const double x = (j - param.cx) * z / param.fx;
            const double y = (i - param.cy) * z / param.fy;

            const auto color = image.at<cv::Vec3b>(i, j);

            fout << x << " " << y << " " << z << " " << static_cast<int>(color(2)) << " " << static_cast<int>(color(1))
                << " " << static_cast<int>(color(0)) << "\n";
        }
    }

    fout.close();
}

constexpr const char *kHelpStr = 
    "usage: epi-depth <images-dir> <chess-x> <chess-y>\n"
    "                 <scale-factor> <is-moving-to-right> <min-deg> <max-deg> <deg-step>\n"
    "                 <depth-scale> <depth-offset>\n"
    "\n"
    " images-dir           directory of images\n"
    "                      images used for calibration should be put in 'calib'\n"
    "                      images used for generate depth map should be put in 'imgs' sub dir\n"
    " chess-x, chess-y     size of chessboard used for calibration\n"
    " scale-factor         scale factor that will be applied to all input images\n"
    " is-moving-to-right   0 if camera moves to left, 1 if camera moves to right\n"
    " min-deg              min search degree from horizon\n"
    " max-deg              max search degree from horizon\n"
    " deg-step             search step of degree\n"
    " depth-scale          scale of 0-1 depth when generate point cloud\n"
    " depth-offset         offset of scaled depth when generate point cloud\n";

}

#define USE_COLOR true
// #define USE_COLOR false


int main(int argc, char *argv[]) {
    if (argc == 2 && strcmp(argv[2], "-h") == 0) {
        std::cout << kHelpStr;
        return 0;
    }

    if (argc != 11) {
        std::cout << kHelpStr << "Invalid input" << std::endl;
        return -1;
    }

    const std::string images_dir(argv[1]);
    const int chess_x = std::atoi(argv[2]);
    const int chess_y = std::atoi(argv[3]);

    const double scale_factor = std::atof(argv[4]);
    const bool moving_to_right = std::atoi(argv[5]) != 0;

    const double min_search_deg = moving_to_right ? 180.0 - std::atof(argv[6]) : std::atof(argv[6]);
    const double max_search_deg = moving_to_right ? 180.0 - std::atof(argv[7]) : std::atof(argv[7]);
    const double deg_search_step = moving_to_right ? -std::atof(argv[8]) : std::atof(argv[8]);
    const size_t num_search_deg = (max_search_deg - min_search_deg) / deg_search_step;

    const double depth_scale = std::atof(argv[9]);
    const double depth_offset = std::atof(argv[10]);

    const CameraParam cam_param = CalibCamera(images_dir + "/calib", chess_x, chess_y, scale_factor);

    std::vector<std::filesystem::path> files;
    for (const auto &entry : std::filesystem::directory_iterator(images_dir + "/imgs")) {
        files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    const int image_num = files.size();

    std::vector<cv::Mat> images;
    images.resize(image_num);

#if USE_COLOR
    std::vector<cv::Mat> images_01;
    images_01.resize(image_num);

    std::vector<cv::Mat> images_dx;
    images_dx.resize(image_num);

    std::vector<cv::Mat> images_dy;
    images_dy.resize(image_num);
#else
    std::vector<cv::Mat> images_gray;
    images_gray.resize(image_num);

    std::vector<cv::Mat> images_gray_dx;
    images_gray_dx.resize(image_num);

    std::vector<cv::Mat> images_gray_dy;
    images_gray_dy.resize(image_num);
#endif

#pragma omp parallel for
    for (int i = 0; i < image_num; ++i) {
        images[i] = std::move(cv::imread(files[i].string()));
        cv::resize(images[i], images[i], cv::Size(), scale_factor, scale_factor);

#if USE_COLOR        
        images[i].convertTo(images_01[i], CV_32F);
        images_01[i] /= 255.0;

        cv::Sobel(images_01[i], images_dx[i], CV_64F, 1, 0);
        cv::Sobel(images_01[i], images_dy[i], CV_64F, 0, 1);
#else
        cv::cvtColor(images[i], images_gray[i], cv::COLOR_BGR2GRAY);
        images_gray[i].convertTo(images_gray[i], CV_32F);
        images_gray[i] /= 255.0;

        cv::Sobel(images_gray[i], images_gray_dx[i], CV_64F, 1, 0);
        cv::Sobel(images_gray[i], images_gray_dy[i], CV_64F, 0, 1);
#endif
    }

#if USE_COLOR
    const int type = images_01[0].type();
#else
    const int type = images_gray[0].type();
#endif
    const int height = images[0].rows;
    const int width = images[0].cols;
    std::cout << image_num << " images, width = " << width << ", height = " << height << "\n";

    const double theta_bias = moving_to_right ? 90.0 : 0.0;

    cv::Mat du(num_search_deg, image_num, CV_32S);
    for (int t = 0; t < num_search_deg; t++) {
        const double theta_deg = min_search_deg + t * deg_search_step;
        du.at<int>(t, 0) = 0;

        const double theta = theta_deg * std::numbers::pi / 180.0;
        const double cot = 1.0 / std::tan(theta);
        for (int i = 1; i < image_num; i++) {
            du.at<int>(t, i) = static_cast<int>(0.5 + cot * i);
        }
    }

    cv::Mat depth_map(height, width, CV_64F);
    for (int i = 0; i < height; i++) {
        const bool show_img = height / 2 == i;

        cv::Mat epi(image_num, width, type);
        for (int j = 0; j < image_num; ++j) {
#if USE_COLOR
            images_01[j].row(i).copyTo(epi.row(j));
#else
            images_gray[j].row(i).copyTo(epi.row(j));
#endif
        }
        if (show_img) {
            cv::imshow("epi", epi);
            cv::waitKey();
        }

        std::cout << i << "/" << height << "\n";

#if USE_COLOR
        cv::Mat epi_dx(image_num, width, CV_64FC3);
#else
        cv::Mat epi_dx(image_num, width, CV_64F);
#endif
        for (int j = 0; j < image_num; ++j) {
#if USE_COLOR
            images_dx[j].row(i).copyTo(epi_dx.row(j));
#else
            images_gray_dx[j].row(i).copyTo(epi_dx.row(j));
#endif
        }
        if (show_img) {
            cv::imshow("epi", epi_dx);
            cv::waitKey();
        }

#if USE_COLOR
        cv::Mat epi_dy(image_num, width, CV_64FC3);
#else
        cv::Mat epi_dy(image_num, width, CV_64F);
#endif
        for (int j = 0; j < image_num; ++j) {
#if USE_COLOR
            images_dy[j].row(i).copyTo(epi_dy.row(j));
#else
            images_gray_dy[j].row(i).copyTo(epi_dy.row(j));
#endif
        }
        if (show_img) {
            cv::imshow("epi", epi_dy);
            cv::waitKey();
        }

        cv::Mat c_theta(width, num_search_deg, CV_64F);
#pragma omp parallel for
        for (int j = 0; j < width; j++) {
            for (int t = 0; t < num_search_deg; t++) {
                double sum = 0.0;
                int count = 0;
                for (int k = 0; k < image_num; k++) {
                    const int jj = j + du.at<int>(t, k);
                    if (jj < 0 || jj >= width) {
                        break;
                    }

#if USE_COLOR
                    const double diff_i = ColorDiff(epi.at<cv::Vec3f>(0, j), epi.at<cv::Vec3f>(k, jj));
                    const double diff_dx = ColorDiff(epi_dx.at<cv::Vec3d>(0, j), epi_dx.at<cv::Vec3d>(k, jj));
                    const double diff_dy = ColorDiff(epi_dy.at<cv::Vec3d>(0, j), epi_dy.at<cv::Vec3d>(k, jj));
#else
                    const double diff_i = std::abs(epi.at<float>(0, j) - epi.at<float>(k, jj));
                    const double diff_dx = std::abs(epi_dx.at<double>(0, j) - epi_dx.at<double>(k, jj));
                    const double diff_dy = std::abs(epi_dy.at<double>(0, j) - epi_dy.at<double>(k, jj));
#endif

                    const double c =
                        (1.0 - kAlpha) * std::min(diff_i, kTau1) + kAlpha * std::min(diff_dx + diff_dy, kTau2);
                    sum += c;

                    ++count;
                }
                c_theta.at<double>(j, t) = sum / std::max(count, 1);
            }
        }

        cv::Mat c_theta_e(width, num_search_deg, CV_64F, cv::Scalar(0));
#pragma omp parallel for
        for (int j = 0; j < width; j++) {
            for (int dj = -kDelta; dj <= kDelta; dj++) {
                const int jj = j + dj;
                if (jj < 0 || jj >= width) {
                    continue;
                }

#if USE_COLOR
                const double diff_i = ColorDiff(epi.at<cv::Vec3f>(0, j), epi.at<cv::Vec3f>(0, jj));
#else
                const double diff_i = std::abs(epi.at<float>(0, j) - epi.at<float>(0, jj));
#endif
                const double exp = std::exp(-std::abs(dj) / kSigmaS - diff_i / kSigmaC);
                for (int t = 0; t < num_search_deg; t++) {
                    const double item = exp * c_theta.at<double>(jj, t);
                    c_theta_e.at<double>(j, t) += item;
                }
            }
        }

        cv::Mat epi_color;
#if USE_COLOR
        epi.copyTo(epi_color);
#else
        cv::cvtColor(epi, epi_color, cv::COLOR_GRAY2BGR);
#endif

#pragma omp parallel for
        for (int j = 0; j < width; j++) {
            double min_c = std::numeric_limits<double>::max();
            int min_theta = -1;

            for (int t = 0; t < num_search_deg; t++) {
                const double c_val = c_theta_e.at<double>(j, t);
                if (c_val < min_c) {
                    min_c = c_val;
                    min_theta = t;
                }
            }
            const double theta = min_search_deg + min_theta * deg_search_step;
            const double tan = std::tan(theta * std::numbers::pi / 180.0);
            depth_map.at<double>(i, j) = std::abs(tan);

            if (show_img) {
                if (j % 30 == 0) {
                    const int dj = du.at<int>(min_theta, image_num - 1);
                    cv::line(epi_color, cv::Point(j, 0), cv::Point(j + dj, image_num - 1), cv::Scalar(0.0, 0.0, 1.0));
                }
            }
        }

        if (show_img) {
            cv::imshow("epi", epi_color);
            cv::waitKey();
        }
    }

    std::cout << "normalizing depth\n";
    double actual_min_depth = std::numeric_limits<double>::max();
    double actual_max_depth = std::numeric_limits<double>::lowest();
#pragma omp parallel for
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            const double depth = depth_map.at<double>(i, j);
            actual_min_depth = std::min(actual_min_depth, depth);
            actual_max_depth = std::max(actual_max_depth, depth);
        }
    }
    std::cout << "min depth = " << actual_min_depth << ", max depth = " << actual_max_depth << "\n";

    const double actual_depth_range = actual_max_depth - actual_min_depth;
#pragma omp parallel for
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            depth_map.at<double>(i, j) = (depth_map.at<double>(i, j) - actual_min_depth) / (actual_depth_range);
        }
    }
    // depth_map.convertTo(depth_map, CV_32F);

    std::cout << "filtering depth\n";
#if USE_COLOR
    cv::Mat filtered_depth_map = guidedFilter(images_01[0], depth_map, kRadius, pow(kEps, 2));
#else
    cv::Mat filtered_depth_map = guidedFilter(images_gray[0], depth_map, kRadius, pow(kEps, 2));
#endif

    std::cout << "reconstructing\n";
    WritePly(filtered_depth_map, images[0], cam_param, depth_scale, depth_offset);

    filtered_depth_map.convertTo(depth_map, CV_32F);

    cv::Mat depth_image(height, width, CV_8UC3);

    cv::cvtColor(depth_map, depth_image, cv::COLOR_GRAY2BGR);
    depth_image.convertTo(depth_image, CV_8UC3, 255.0);
    cv::imwrite("depth.jpg", depth_image);

    cv::imshow("depth", depth_image);
    cv::waitKey();

    cv::destroyAllWindows();

    return 0;
}
