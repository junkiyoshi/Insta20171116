#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetFrameRate(60);
	ofBackground(255);
	ofSetWindowTitle("Insta");

	ofNoFill();

	this->frame_image.allocate(ofGetWidth(), ofGetHeight(), OF_IMAGE_COLOR);
	this->frame = cv::Mat(this->frame_image.getHeight(), this->frame_image.getWidth(), CV_MAKETYPE(CV_8UC3, this->frame_image.getPixels().getNumChannels()), this->frame_image.getPixels().getData(), 0);

	this->dog_image.load("D:\\yolo_data\\dog.jpg");

	this->net = readNetFromDarknet(modelConfiguration, modelBinary);
	if (net.empty()) {
		cout << "dnn Net is empty" << endl;
	}

	this->line_index = 0;
}

//--------------------------------------------------------------
void ofApp::update() {
	if (ofGetMousePressed()) {
		this->lines[this->line_index].addVertex(ofGetMouseX(), ofGetMouseY());
	}
}

//--------------------------------------------------------------
void ofApp::draw() {

	ofSetColor(0);
	for (int i = 0; i < this->lines.size(); i++) {
		this->lines[i].draw();
	}

	ofSetColor(255);
	this->dog_image.draw(0, ofGetHeight() / 2 - this->dog_image.getHeight() / 2);

	this->frame_image.grabScreen(0, 0, ofGetWidth(), ofGetHeight());
	cv::cvtColor(this->frame, this->frame, CV_BGR2RGB);

	if (!this->frame.empty()) {
		this->detection();
	}

	cv::imshow("frame", this->frame);
	cv::waitKey(1);
}

//--------------------------------------------------------------
void ofApp::detection() {

	cv::resize(this->frame, this->resized, cv::Size(this->network_width, this->network_height));
	this->inputBlob = blobFromImage(this->resized, 1 / 255.F);		//Convert Mat to batch of images
	this->net.setInput(this->inputBlob, "data");					//set the network input
	this->detectionMat = this->net.forward("detection_out");		//compute output

	float confidenceThreshold = 0.15;
	for (int i = 0; i < detectionMat.rows; i++)
	{
		const int probability_index = 5;
		const int probability_size = detectionMat.cols - probability_index;
		float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);

		size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
		float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

		if (confidence > confidenceThreshold)
		{
			float x = detectionMat.at<float>(i, 0);
			float y = detectionMat.at<float>(i, 1);
			float width = detectionMat.at<float>(i, 2);
			float height = detectionMat.at<float>(i, 3);
			float xLeftBottom = (x - width / 2) * frame.cols;
			float yLeftBottom = (y - height / 2) * frame.rows;
			float xRightTop = (x + width / 2) * frame.cols;
			float yRightTop = (y + height / 2) * frame.rows;

			cv::Rect object(
				(int)xLeftBottom,
				(int)yLeftBottom,
				(int)(xRightTop - xLeftBottom),
				(int)(yRightTop - yLeftBottom));

			cv::Rect title(
				(int)xLeftBottom,
				(int)yLeftBottom - 15,
				100,
				15);

			std::string object_name = "";
			if (objectClass < class_names.size()) {
				object_name = class_names[objectClass];
			}
			cv::rectangle(frame, title, Scalar(0, 255, 0), -1);
			cv::putText(frame, object_name, cv::Point((int)xLeftBottom, (int)yLeftBottom - 2), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255), 2);
			cv::rectangle(frame, object, Scalar(0, 255, 0));
		}
	}
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
	this->lines.push_back(ofPolyline());
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {
	this->line_index += 1;
}

//--------------------------------------------------------------
int main() {
	ofSetupOpenGL(720, 720, OF_WINDOW);
	ofRunApp(new ofApp());
}

