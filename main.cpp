#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void detectFace(Mat&, CascadeClassifier&, double, bool);

//string cascadeName="./data/haarcascades/haarcascade_frontalface_alt.xml";
string cascadeName="./data/lbpcascades/lbpcascade_frontalface.xml";
int limitms = cvRound(1 / 30 * 1000);//30 FPS


int main(){
  /*
  VideoCapture capture;
  Mat frame, image;
  string inputName;
  bool tryflip;
  */
  Mat frame, image;
  CascadeClassifier cascade;
  Size maxSize = Size(800,600);
  double scale=1.2;
  VideoCapture capture;
  try{
    cascade.load(cascadeName);
  } catch (cv::Exception e){}
  if(cascade.empty()){
    cout<<"ERROR: Could't load Cascade Classifier Face Detector ("<<cascadeName<<")!"<<endl;
  }

  image = imread("./opencv_sample/Test-080.JPG",1);
  //resize(image,image,Size(),0.4,0.4,CV_INTER_AREA);
  cout<<image.size()<<endl;
  //imshow("Source",frame);
  //waitKey(0);
  detectFace(image,cascade,scale,1);
}

void detectFace(Mat& image, CascadeClassifier& cascade,double scale,bool tryflip){
  double timer = 0;
  vector<Rect> faces, facesflip;
  const static Scalar colors[] = {
      Scalar(255,0,0),
      Scalar(255,128,0),
      Scalar(255,255,0),
      Scalar(0,255,0),
      Scalar(0,128,255),
      Scalar(0,255,255),
      Scalar(0,0,255),
      Scalar(255,0,255)
  };
  Mat gray, smallImage;
  //预处理
  if      (image.channels() == 3) cvtColor(image, gray, COLOR_BGR2GRAY);
  else if (image.channels() == 4) cvtColor(image, gray, COLOR_BGRA2GRAY);
  else                            gray = image;
  double fx = 1 / scale;
  resize( gray, smallImage, Size(), fx, fx, INTER_LINEAR );
  equalizeHist( smallImage, smallImage );

  timer = (double)getTickCount();
  cascade.detectMultiScale( smallImage, faces, 1.1, 2, 0
      //|CASCADE_FIND_BIGGEST_OBJECT
      //|CASCADE_DO_ROUGH_SEARCH
      |CASCADE_SCALE_IMAGE,
      Size(20, 20) );
  if( tryflip ){
      flip(smallImage, smallImage, 1);
      cascade.detectMultiScale( smallImage, facesflip, 1.1, 2, 0
                                |CASCADE_SCALE_IMAGE,
                                //|CASCADE_FIND_BIGGEST_OBJECT
                                //|CASCADE_DO_ROUGH_SEARCH
                                Size(30, 30) );
      for( vector<Rect>::const_iterator r = facesflip.begin(); r != facesflip.end(); ++r ){
          faces.push_back(Rect(smallImage.cols - r->x - r->width, r->y, r->width, r->height));
      }
  }

  timer = (double)getTickCount() - timer;
  cout<<"detection time = "<<timer*1000/getTickFrequency()<<" ms."<<endl;
  cout<<((timer*1000/getTickFrequency()-1000/30 < 0)?"Success":"Failed")<<endl;
  cout<<"Detected "<<faces.size()<<"faces."<<endl;
  for ( size_t i = 0; i < faces.size(); i++ ){
      Rect ROI = faces[i];
      vector<Rect> nestedObjects;
      Scalar color = colors[i%8];
      //Mat temp=image
      //double aspect_ratio = (double)r.width/r.height;//长宽比
      rectangle(image, cvPoint(cvRound(ROI.x*scale),cvRound(ROI.y*scale)), cvPoint(cvRound((ROI.x+ROI.width)*scale), cvRound((ROI.y+ROI.height)*scale)), color, 3, 8, 0 );

  }
  imshow( "result", image );
  waitKey();
}

//void mosaic(){

//}
