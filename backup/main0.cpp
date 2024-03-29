#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>


using namespace std;
using namespace cv;


void detectFace(Mat&, CascadeClassifier&, vector<Rect>&, double, bool);

string cascadeName="./data/haarcascades/haarcascade_frontalface_alt.xml";
//string cascadeName="./data/lbpcascades/lbpcascade_frontalface.xml";
int limitms = cvRound(1000 / 15);//15 FPS


int main(){
  /*
  VideoCapture capture;
  Mat frame, image;
  string inputName;
  bool tryflip;
  */
  vector<Rect> faces;
  Mat frame,image;
  Mat mosaic;
  Mat hsv, hue, mask, hist, backproj;
  VideoCapture capture;
  const double scale=1.2;

  //Face Detector
  //Load Haar Cascade
  CascadeClassifier cascade;
  try{
    cascade.load(cascadeName);
  } catch (cv::Exception e){}
  if(cascade.empty()){
    cout<<"ERROR: Could't load Cascade Classifier Face Detector ("<<cascadeName<<")!"<<endl;
  }

    image = imread("./opencv_sample/Test-081.JPG",1);

    //Mosaic
    resize( image, mosaic, Size(), 0.05, 0.05, INTER_AREA );
    resize( mosaic, mosaic, Size(), 20, 20, INTER_NEAREST );

    detectFace(image,cascade,faces,scale,0);

    while(!faces.empty()){

      //Rect ROI(cvRound(faces.back().x*scale), cvRound(faces.back().y*scale), cvRound(faces.back().width*scale), cvRound(faces.back().height*scale) );
      //mosaic(ROI).copyTo(image(ROI));
      rectangle(image, cvPoint(cvRound(faces.back().x*scale),cvRound(faces.back().y*scale)), cvPoint(cvRound((faces.back().x+faces.back().width)*scale), cvRound((faces.back().y+faces.back().height)*scale)), Scalar(255,0,0), 3, 8, 0 );
      faces.pop_back();
    }

    imshow( "Face Detector", image );
    waitKey(0);
}

void detectFace(Mat& image, CascadeClassifier& cascade,vector<Rect>& faces,double scale,bool tryflip){
  double timer = 0;
  vector<Rect> facesflip;

  Mat gray, smallImage;
  //预处理
  if      (image.channels() == 3) cvtColor(image, gray, COLOR_BGR2GRAY);
  else if (image.channels() == 4) cvtColor(image, gray, COLOR_BGRA2GRAY);
  else                            gray = image;
  double fx = 1 / scale;
  resize( gray, smallImage, Size(), fx, fx, INTER_LINEAR );
  equalizeHist(smallImage, smallImage );
  //Face Detect
  timer = (double)getTickCount();
  cascade.detectMultiScale( smallImage, faces, 1.1, 2, 0
      //|CASCADE_FIND_BIGGEST_OBJECT,
      //|CASCADE_DO_ROUGH_SEARCH,
      |CASCADE_SCALE_IMAGE,
      Size(20, 20) );
  if( tryflip ){
      flip(smallImage, smallImage, 1);
      cascade.detectMultiScale( smallImage, facesflip, 1.1, 2, 0
                                |CASCADE_SCALE_IMAGE,
                                //|CASCADE_FIND_BIGGEST_OBJECT
                                //|CASCADE_DO_ROUGH_SEARCH
                                Size(20, 20) );
      for( vector<Rect>::const_iterator r = facesflip.begin(); r != facesflip.end(); ++r ){
          faces.push_back(Rect(smallImage.cols - r->x - r->width, r->y, r->width, r->height));
      }
  }

  timer = (double)getTickCount() - timer;
  cout<<"detection time = "<<timer*1000/getTickFrequency()<<" ms."<<endl;
  cout<<((timer*1000/getTickFrequency()-limitms < 0)?"Success":"Failed")<<endl;
  cout<<"Detected "<<faces.size()<<" faces."<<endl;
  /*for ( size_t i = 0; i < faces.size(); i++ ){
      Rect ROI = faces[i];
      //Mat temp=image
      //double aspect_ratio = (double)r.width/r.height;//长宽比
      rectangle(image, cvPoint(cvRound(ROI.x*scale),cvRound(ROI.y*scale)), cvPoint(cvRound((ROI.x+ROI.width)*scale), cvRound((ROI.y+ROI.height)*scale)), Scalar(255,0,0), 3, 8, 0 );
  }*/
}
