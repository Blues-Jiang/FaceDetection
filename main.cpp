#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace cv;

Rect changeRect(Rect A,Rect B);//Change B into A

void detectFace(Mat&, CascadeClassifier&, vector<Rect>&, double, bool);

string cascadeName="./data/haarcascades/haarcascade_frontalface_alt2.xml";
//string cascadeName="./data/lbpcascades/lbpcascade_frontalface.xml";
int limitms = cvRound(1000 / 15);//15 FPS

const int frameWidth = 320;
const int frameHeight = 240;

double timer_start,timer_stop,FPS;
int faceNum=0;
bool openMosaic = true;

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

  //Open Camera
  //capture.open(CAP_V4L2);
  capture.open(CAP_ANY);
  if(!capture.isOpened()){
    cout << "***Could not initialize capturing...***"<<endl;
    return -1;
  }
  cout << "Video capturing has been started ..." << endl;
  namedWindow( "Face Detector", 0 );

  //Set Camera
  capture.set(CAP_PROP_FRAME_WIDTH, frameWidth);
  capture.set(CAP_PROP_FRAME_HEIGHT, frameHeight);
  //capture.set(CAP_PROP_FPS, 15);

  while(1){
    capture >> frame;
    timer_start = (double)getTickCount();
    if( frame.empty() ) break;

    image = frame.clone();
    //cout<<"FPS:"<<capture.get(CAP_PROP_FPS)<<endl;
    //Mosaic
    if(openMosaic){
      resize( image, mosaic, Size(), 0.1, 0.1, INTER_AREA );
      resize( mosaic, mosaic, Size(), 10, 10, INTER_NEAREST );
    }

    detectFace(image,cascade,faces,scale,0);
    faceNum = faces.size();
    while(!faces.empty()){
      Rect ROI(cvRound(faces.back().x*scale), cvRound(faces.back().y*scale), cvRound(faces.back().width*scale), cvRound(faces.back().height*scale) );
      if(openMosaic){
        mosaic(ROI).copyTo(image(ROI));
      }
      else{
        rectangle(image, cvPoint(ROI.x,ROI.y), cvPoint((ROI.x+ROI.width), (ROI.y+ROI.height)), Scalar(200,0,0), 2, 8, 0 );
      }
      faces.pop_back();
    }
    timer_stop = (double)getTickCount();
    FPS = getTickFrequency()/(timer_stop - timer_start);
    cout<<FPS<<endl;
    putText(  image,
              "FPS:"+to_string(FPS).substr(0,5),
              Point(5,15),
              CV_FONT_HERSHEY_COMPLEX_SMALL,
              0.8,//font scale
              Scalar(255,255,255),  //color
              1,
              LINE_8,
              false);

    putText(  image,
              "Detected "+to_string(faceNum)+" faces",
              Point(5,frameHeight-5),
              CV_FONT_HERSHEY_COMPLEX_SMALL,
              0.8,//font scale
              Scalar(255,255,255),  //color
              1,
              LINE_8,
              false);
    imshow( "Face Detector", image );
    char c = (char)waitKey(10);
    if( c == 'm' || c == 'M') openMosaic = !openMosaic;
    if( c == 27 || c == 'q' || c == 'Q' ) break;
  }

  /*
  image = imread("./1.jpg",1);
  //Mosaic
  resize( image, image, Size(), 0.05, 0.05, INTER_AREA );
  resize( image, image, Size(), 20, 20, INTER_NEAREST );
  imshow("Mosaic", image);
  waitKey();
  */
}

void detectFace(Mat& image, CascadeClassifier& cascade,vector<Rect>& faces,double scale,bool tryflip){
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
  //timer = (double)getTickCount();
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

  //timer = (double)getTickCount() - timer;
  //cout<<"detection time = "<<timer*1000/getTickFrequency()<<" ms."<<endl;
  //cout<<((timer*1000/getTickFrequency()-limitms < 0)?"Success":"Failed")<<endl;
  //cout<<"Detected "<<faces.size()<<" faces."<<endl;

}

Rect changeRect(Rect A,Rect B){//Change B into A
  if(B.x<A.x)
    B.x=A.x;
  if(B.x+B.width>A.width)
    B.width=A.width-B.x;
  if(B.y<A.y)
    B.y=A.y;
  if(B.y+B.height>A.height)
    B.height=A.height-B.y;
  return(B);
}
