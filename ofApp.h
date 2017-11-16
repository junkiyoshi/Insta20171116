#include "ofMain.h"

// OpenCV 3.3.1
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

class ofApp : public ofBaseApp {

public:
	void setup();
	void update();
	void draw();
	void detection();

	void keyPressed(int key) {};
	void keyReleased(int key) {};
	void mouseMoved(int x, int y) {};
	void mouseDragged(int x, int y, int button) {};
	void mousePressed(int x, int y, int button);
	void mouseReleased(int x, int y, int button);
	void mouseEntered(int x, int y) {};
	void mouseExited(int x, int y) {};
	void windowResized(int w, int h) {};
	void dragEvent(ofDragInfo dragInfo) {};
	void gotMessage(ofMessage msg) {};

	vector<ofPolyline> lines;
	int line_index;
	ofImage dog_image;

	cv::Mat frame;
	ofImage frame_image;
	cv::VideoCapture cap;

	const size_t network_width = 416;
	const size_t network_height = 416;

	String modelConfiguration = "D:\\yolo_data\\yolo.cfg";
	String modelBinary		  = "D:\\yolo_data\\yolo.weights";

	dnn::Net net;
	cv::Mat resized, inputBlob, detectionMat;

	const vector<string> class_names = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
		"truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
		"bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
		"zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
		"frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
		"baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
		"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
		"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
		"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
		"oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
		"teddy bear", "hair drier", "toothbrush" };
};