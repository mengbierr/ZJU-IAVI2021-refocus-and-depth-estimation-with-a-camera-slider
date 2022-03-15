#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

int main(int argc, char* argv[]) {
  std::vector<cv::Mat> images;

  std::vector<std::filesystem::path> files;
  for (const auto& entry :
      //  std::filesystem::directory_iterator("./final/LEGO/imgs"))
       std::filesystem::directory_iterator("./qsc/20-1-1-22-35"))
    files.push_back(entry.path());
  std::sort(files.begin(), files.end());
  const int image_num = files.size();
  // const int image_num = 15;
  images.resize(image_num);

#pragma omp parallel for
  for (int i = 0; i < image_num; ++i) {
    images[i] = std::move(cv::imread(files[i]));
  }
  std::cout << image_num << " images were found." << std::endl;

  const auto type = images[0].type();
  const auto height = images[0].rows;
  const auto width = images[0].cols;

  cv::Mat M(2, 3, CV_32F);
  M.at<float>(0, 0) = 1;
  // if (argc == 2) {
  if (argc >= 2) {
    M.at<float>(0, 1) = std::stof(argv[1]);
  } else {
    M.at<float>(0, 1) = 0;
    // train [-2, 0]
    // chess [0, 2]
  }
  M.at<float>(0, 2) = 0;
  M.at<float>(1, 0) = 0;
  M.at<float>(1, 1) = 1;
  M.at<float>(1, 2) = 0;

  cv::Mat output_image(height, width, type);
#pragma omp parallel for
  for (int i = 0; i < height; ++i) {
    cv::Mat epi(image_num, width, type);
    for (int j = 0; j < image_num; ++j) images[j].row(i).copyTo(epi.row(j));
    warpAffine(epi, epi, M, epi.size());
    if (!(i % (height / 10))) {
      char number[10];
      sprintf(number, "%d", i);
      cv::imwrite(std::string(number) + "epi.jpg", epi);
    }
    cv::Mat row_mean;
    cv::reduce(epi, row_mean, 0, cv::REDUCE_AVG);
    row_mean.row(0).copyTo(output_image.row(i));
  }

  cv::imwrite("result.jpg", output_image);
  // cv::String window_name = "result";
  // cv::namedWindow(window_name);
  // cv::imshow(window_name, output_image);
  // cv::waitKey(0);
  // cv::destroyWindow(window_name);
  return 0;
}
