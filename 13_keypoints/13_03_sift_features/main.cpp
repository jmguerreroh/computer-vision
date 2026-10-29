/**
 * @file main.cpp
 * @brief SIFT keypoints and their repeatability under a change of scale
 * @author José Miguel Guerrero Hernández
 * @note This example detects SIFT keypoints, draws each one with its scale and
 *       orientation, and measures how many of them are found again after
 *       shrinking the image, which is the test Harris fails.
 *
 * SIFT (Scale-Invariant Feature Transform):
 *   1. Scale space: the image is blurred with Gaussians of growing sigma,
 *      grouped in octaves
 *   2. Difference of Gaussians (DoG): keypoints are the extrema in space AND
 *      scale, each value compared with its 26 neighbours. Low-contrast extrema
 *      and those lying on an edge are discarded
 *   3. Orientation: the dominant direction of the gradient histogram around
 *      the point
 *   4. Descriptor: gradient histograms of 8 orientations on a 4x4 grid,
 *      128 floating-point values compared with the Euclidean distance
 *
 * Every keypoint carries its scale (cv::KeyPoint::size, a diameter in pixels).
 * Small keypoints vanish when the image is shrunk, because their detail is
 * lost; large ones survive. The trackbars let you see both effects.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>  // SIFT, drawKeypoints
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace
{
// The images this example works on are about 1400 px on the long side, and
// several windows at that size do not fit on a normal screen. The processing
// always runs at full resolution: only the copy sent to the screen is reduced,
// with INTER_AREA, which is the interpolation meant for shrinking
constexpr int MAX_DISPLAY_SIDE = 800;

// Returns the copy that goes to the screen, already reduced. Whatever is drawn
// on the result keeps its size in screen pixels, so labels are written here and
// not on the full-resolution frame: drawn before the reduction they shrink with
// it and stop being readable
cv::Mat fitToScreen(const cv::Mat & image)
{
  const int side = std::max(image.cols, image.rows);
  if (side <= MAX_DISPLAY_SIDE || image.empty()) {
    return image.clone();
  }
  const double factor = static_cast<double>(MAX_DISPLAY_SIDE) / side;
  cv::Mat reduced;
  cv::resize(image, reduced, cv::Size(), factor, factor, cv::INTER_AREA);
  return reduced;
}
}  // namespace

namespace Config
{
// Repeatability test: a keypoint counts as repeated if one of the keypoints
// detected on the shrunk image, taken back to the original frame, lies closer
// than this. It is the same criterion the book uses for Harris
constexpr double TOLERANCE_PX = 4.0;

// Trackbars
constexpr int DEFAULT_SCALE_PERCENT = 50;   // shrink factor, in percent
constexpr int MIN_SCALE_PERCENT = 10;
constexpr int MAX_SCALE_PERCENT = 100;
constexpr int DEFAULT_MIN_SIZE = 16;        // smallest diameter drawn, in pixels
constexpr int MAX_MIN_SIZE = 64;

// Visualization parameters
const cv::Scalar KEYPOINT_COLOR(0, 255, 0);
const cv::Scalar TEXT_COLOR(0, 255, 0);
constexpr double TEXT_FONT_SCALE = 0.6;
constexpr int TEXT_THICKNESS = 1;
constexpr int TEXT_X = 10;
constexpr int TEXT_Y = 25;
constexpr int TEXT_LINE_HEIGHT = 22;
constexpr int PANEL_HEIGHT = 80;

// Key codes
constexpr int KEY_ESC = 27;
constexpr int KEY_RESET = 'r';
constexpr int WAITKEY_DELAY = 50;
}

/**
 * @brief Application state shared with the trackbar callback
 */
struct SIFTApp
{
  cv::Mat src;                           // original image, in color
  cv::Mat gray;                          // original image, in gray levels
  cv::Ptr<cv::SIFT> sift;
  std::vector<cv::KeyPoint> keypoints;   // keypoints of the original image

  const char * window_original = "SIFT keypoints";
  const char * window_scaled = "SIFT on the shrunk image";
  const char * trackbar_scale = "Scale %";
  const char * trackbar_size = "Min diameter";
};

SIFTApp app;

/**
 * @brief Keypoints whose diameter is at least min_size pixels
 */
std::vector<cv::KeyPoint> largerThan(const std::vector<cv::KeyPoint> & keypoints,
                                     float min_size)
{
  std::vector<cv::KeyPoint> out;
  for (const auto & kp : keypoints) {
    if (kp.size >= min_size) {
      out.push_back(kp);
    }
  }
  return out;
}

/**
 * @brief For each original keypoint, whether it is detected again
 * @param original Keypoints of the original image
 * @param transformed Keypoints of the shrunk image
 * @param inverse Affine transform that takes the shrunk image back to the original
 */
std::vector<bool> repeated(const std::vector<cv::KeyPoint> & original,
                           const std::vector<cv::KeyPoint> & transformed,
                           const cv::Mat & inverse)
{
  // Take the keypoints of the shrunk image back to the original frame
  std::vector<cv::Point2f> found, back;
  for (const auto & kp : transformed) {
    found.push_back(kp.pt);
  }
  if (!found.empty()) {
    cv::transform(found, back, inverse);
  }

  std::vector<bool> out(original.size(), false);
  for (size_t i = 0; i < original.size(); ++i) {
    for (const auto & p : back) {
      if (cv::norm(original[i].pt - p) < Config::TOLERANCE_PX) {
        out[i] = true;
        break;
      }
    }
  }
  return out;
}

/**
 * @brief Writes the lines of text on a dark band at the top of the image
 */
void drawPanel(cv::Mat & image, const std::vector<std::string> & lines)
{
  cv::rectangle(image, cv::Point(0, 0), cv::Point(image.cols, Config::PANEL_HEIGHT),
                cv::Scalar(0, 0, 0), cv::FILLED);
  int y = Config::TEXT_Y;
  for (const auto & line : lines) {
    cv::putText(image, line, cv::Point(Config::TEXT_X, y), cv::FONT_HERSHEY_SIMPLEX,
                Config::TEXT_FONT_SCALE, Config::TEXT_COLOR, Config::TEXT_THICKNESS,
                cv::LINE_AA);
    y += Config::TEXT_LINE_HEIGHT;
  }
}

std::string percent(double fraction)
{
  std::ostringstream s;
  s << std::fixed << std::setprecision(1) << 100.0 * fraction << " %";
  return s.str();
}

/**
 * @brief Shrinks the image, detects again and measures the repeatability
 */
void update(int, void *)
{
  const int scale_percent = std::max(Config::MIN_SCALE_PERCENT,
                                     cv::getTrackbarPos(app.trackbar_scale,
                                                        app.window_original));
  const float min_size = static_cast<float>(cv::getTrackbarPos(app.trackbar_size,
                                                               app.window_original));
  const double factor = scale_percent / 100.0;

  // ========================================
  // Shrink the Image
  // ========================================
  // The image is shrunk around its center and kept on a canvas of the same
  // size, filled with the mean gray level: the keypoints stay where they can
  // be compared, and the border adds no corners of its own
  const cv::Point2f center(app.gray.cols / 2.0f, app.gray.rows / 2.0f);
  const cv::Mat forward = cv::getRotationMatrix2D(center, 0.0, factor);
  cv::Mat inverse;
  cv::invertAffineTransform(forward, inverse);
  cv::Mat shrunk;
  cv::warpAffine(app.gray, shrunk, forward, app.gray.size(), cv::INTER_AREA,
                 cv::BORDER_CONSTANT, cv::Scalar::all(cv::mean(app.gray)[0]));

  // ========================================
  // Detect Again and Compare
  // ========================================
  std::vector<cv::KeyPoint> found;
  app.sift->detect(shrunk, found);
  const std::vector<bool> ok = repeated(app.keypoints, found, inverse);

  size_t n_all = 0, n_large = 0, large = 0;
  for (size_t i = 0; i < app.keypoints.size(); ++i) {
    n_all += ok[i];
    if (app.keypoints[i].size >= min_size) {
      ++large;
      n_large += ok[i];
    }
  }
  const double rate_all = app.keypoints.empty() ? 0.0 :
    static_cast<double>(n_all) / app.keypoints.size();
  const double rate_large = large == 0 ? 0.0 : static_cast<double>(n_large) / large;

  // ========================================
  // Draw
  // ========================================
  // DRAW_RICH_KEYPOINTS draws each keypoint as a circle of its diameter, with
  // a radius marking its orientation
  cv::Mat left, right;
  cv::drawKeypoints(app.src, largerThan(app.keypoints, min_size), left,
                    Config::KEYPOINT_COLOR, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::Mat shrunk_color;
  cv::cvtColor(shrunk, shrunk_color, cv::COLOR_GRAY2BGR);
  cv::drawKeypoints(shrunk_color, largerThan(found, min_size * static_cast<float>(factor)),
                    right, Config::KEYPOINT_COLOR,
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  left = fitToScreen(left);
  right = fitToScreen(right);
  const std::string size_text = std::to_string(static_cast<int>(min_size)) + " px";
  drawPanel(left, {
    std::to_string(app.keypoints.size()) + " keypoints, " + std::to_string(large) +
    " with diameter >= " + size_text,
    "Drawn: diameter >= " + size_text});
  drawPanel(right, {
    "Scale " + std::to_string(scale_percent) + " %: " + std::to_string(found.size()) +
    " keypoints",
    "Repeated: " + percent(rate_all) + " of all, " + percent(rate_large) +
    " of diameter >= " + size_text});
  cv::imshow(app.window_original, left);
  cv::imshow(app.window_scaled, right);

  std::cout << "\r[SIFT] scale " << std::setw(3) << scale_percent << " %: repeated "
            << percent(rate_all) << " of all, " << percent(rate_large)
            << " of diameter >= " << size_text << "     " << std::flush;
}

/**
 * @brief Display help information
 */
void printHelp()
{
  std::cout << "========================================" << std::endl;
  std::cout << "SIFT Keypoints and Scale" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << "Each keypoint is drawn as a circle of its diameter, with a" << std::endl;
  std::cout << "radius marking its orientation." << std::endl;
  std::cout << "A keypoint counts as repeated if it is detected again less than "
            << Config::TOLERANCE_PX << " px" << std::endl;
  std::cout << "from where it should be after shrinking the image." << std::endl;
  std::cout << "----------------------------------------" << std::endl;
  std::cout << "Controls:" << std::endl;
  std::cout << "  Scale %       - Shrink factor of the second image" << std::endl;
  std::cout << "  Min diameter  - Smallest keypoint drawn and counted apart" << std::endl;
  std::cout << "  r             - Reset to defaults" << std::endl;
  std::cout << "  ESC           - Exit" << std::endl;
  std::cout << "========================================\n" << std::endl;
}

int main(int argc, char ** argv)
{
  // ========================================
  // Load and Prepare Image
  // ========================================
  // Command-line arguments; --help prints the usage
  cv::CommandLineParser parser(argc, argv,
    "{help h | | Show this help message}"
    "{@input | ../../data/chess.png | Input file}");
  if (parser.has("help")) {
    parser.printMessage();
    return EXIT_SUCCESS;
  }
  const std::string filename = parser.get<std::string>("@input");

  // Without this, a malformed value (--frames=xyz) is reported by the parser
  // but the example carries on with the default, which is the hardest kind
  // of failure to diagnose. Note it validates values, not option names:
  // cv::CommandLineParser ignores an unknown option without complaining
  if (!parser.check()) {
    parser.printErrors();
    return EXIT_FAILURE;
  }

  app.src = cv::imread(cv::samples::findFile(filename, false), cv::IMREAD_COLOR);

  if (app.src.empty()) {
    std::cerr << "Error: Could not load image!" << std::endl;
    std::cerr << "Path: " << filename << std::endl;
    std::cerr << "Usage: " << argv[0] << " [image_path]" << std::endl;
    return EXIT_FAILURE;
  }

  cv::cvtColor(app.src, app.gray, cv::COLOR_BGR2GRAY);
  std::cout << "Image loaded: " << app.src.cols << "x" << app.src.rows
            << " pixels\n" << std::endl;

  printHelp();

  // ========================================
  // Detect SIFT Keypoints
  // ========================================
  // cv::SIFT::create() with its defaults keeps every keypoint it finds
  // (nfeatures = 0), three layers per octave, and discards extrema whose
  // contrast is below 0.04 or whose ratio of Hessian eigenvalues exceeds 10
  app.sift = cv::SIFT::create();
  app.sift->detect(app.gray, app.keypoints);

  std::vector<float> sizes;
  for (const auto & kp : app.keypoints) {
    sizes.push_back(kp.size);
  }
  std::sort(sizes.begin(), sizes.end());
  if (!sizes.empty()) {
    std::cout << app.keypoints.size() << " keypoints, median diameter "
              << std::fixed << std::setprecision(1) << sizes[sizes.size() / 2]
              << " px\n" << std::endl;
  }

  // ========================================
  // Create Interactive GUI
  // ========================================
  cv::namedWindow(app.window_original, cv::WINDOW_AUTOSIZE);
  cv::namedWindow(app.window_scaled, cv::WINDOW_AUTOSIZE);
  // Both trackbars exist before either is moved: moving one calls update(),
  // which reads the two of them
  cv::createTrackbar(app.trackbar_scale, app.window_original, nullptr,
                     Config::MAX_SCALE_PERCENT, update);
  cv::createTrackbar(app.trackbar_size, app.window_original, nullptr,
                     Config::MAX_MIN_SIZE, update);
  cv::setTrackbarPos(app.trackbar_scale, app.window_original,
                     Config::DEFAULT_SCALE_PERCENT);
  cv::setTrackbarPos(app.trackbar_size, app.window_original, Config::DEFAULT_MIN_SIZE);

  update(0, nullptr);

  // ========================================
  // Main Event Loop
  // ========================================
  while (true) {
    const int key = cv::waitKey(Config::WAITKEY_DELAY);
    if (key == Config::KEY_ESC) {
      break;
    }
    if (key == Config::KEY_RESET) {
      cv::setTrackbarPos(app.trackbar_scale, app.window_original,
                         Config::DEFAULT_SCALE_PERCENT);
      cv::setTrackbarPos(app.trackbar_size, app.window_original, Config::DEFAULT_MIN_SIZE);
      std::cout << "\nReset to default parameters" << std::endl;
      update(0, nullptr);
    }
  }

  std::cout << "\n" << std::endl;
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
