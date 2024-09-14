#include <iostream>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;

void detectFace(Mat frame)
{
    // Blurring/smoothing the frame with gaussian blur algorithm
    // This re-evaluate the pixels to weighted averages of its neighbors, with respect to the kernel size
    GaussianBlur(frame, frame, Size(5,5), 0);

    // Not sure if making it color and comparing all three rgb field is more accurate
    // Greyscale is faster and more efficient
    Mat grey_frame;
    // Takes in first param and converts acordingly to the thrid param and output to second param
    cvtColor(frame, grey_frame, COLOR_BGR2GRAY);

    // Face detection
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(grey_frame, faces);

    for(size_t i = 0; i < faces.size(); i++)
    {
        Point top_left(faces[i].x, faces[i].y);
        Point bottom_right(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
        rectangle(frame, top_left, bottom_right, Scalar(255, 255, 0),5);
    }

    imshow("Face detection", frame);
}

int main(int argc, char** argv) 
{
    // Loading the pre trained harr cascade classifier
    // string face_classifier_path = "/usr/local/Cellar/opencv/4.9.0_7/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml";
    string face_classifier_path = "../trained_cascade7/cascade.xml";

    if(!face_cascade.load(face_classifier_path))
    {
        cout << "Classifier loading unsuccessful";
        return -1;
    }

    cout << "Classifier loading successful" << endl;

    // Open video input source
    VideoCapture webcam(0);

    if(!webcam.isOpened())
    {
        cout << "Video source not found";
        return -1;
    }

    Mat frame;
    while(webcam.read(frame))
    {
        if(frame.empty())
        {
            cout << "No frame captured by the camera";
            break;
        }

        // If frame is being captured, apply classifier on the frame
        detectFace(frame);

        // Press q to quit
        if(waitKey(10) == 'q')
        {
            break;
        }
    }
    return 0;
}