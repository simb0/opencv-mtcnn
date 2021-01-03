#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <mtcnn/detector.h>
#include <string>

#include <jni.h>        // JNI header provided by JDK
#include <stdio.h>      // C Standard IO Header
#include "de_mypicardo_facer_face_detection_mtcnn_business_OpenCvMtcnnFaceDetector.h"   // Generated


using rectPoints = std::pair<cv::Rect, std::vector<cv::Point>>;

JNIEXPORT jstring JNICALL Java_de_mypicardo_facer_face_detection_mtcnn_business_OpenCvMtcnnFaceDetector_findFaces(JNIEnv *env, jobject thisObj, jstring mdir, jstring imagePath, jfloat minFaceSize, jfloat scaleFactor) {

  const char *modelPath = env->GetStringUTFChars(mdir, 0);
  const char *imgPath = env->GetStringUTFChars(imagePath, 0);

  std::string modelDir(modelPath);

  ProposalNetwork::Config pConfig;
  pConfig.caffeModel = modelDir +"/det1.caffemodel";
  pConfig.protoText =  modelDir +"/det1.prototxt";
  pConfig.threshold = 0.6f;

  RefineNetwork::Config rConfig;
  rConfig.caffeModel = modelDir +"/det2.caffemodel";
  rConfig.protoText =  modelDir +"/det2.prototxt";
  rConfig.threshold = 0.7f;

  OutputNetwork::Config oConfig;
  oConfig.caffeModel = modelDir +"/det3.caffemodel";
  oConfig.protoText =  modelDir +"/det3.prototxt";
  oConfig.threshold = 0.7f;

  MTCNNDetector detector(pConfig, rConfig, oConfig);
  cv::Mat img = cv::imread(imgPath);

  std::vector<Face> faces;


  faces = detector.detect(img, minFaceSize, scaleFactor);


  std::cout << "Number of faces found in the supplied image - " << faces.size()
            << std::endl;

std::string json = "["; 

 for (size_t i = 0; i < faces.size(); ++i) {
	std::string elEnd = ",";
	if(i == faces.size()-1) {
		elEnd = "";
	}
	json  = json + "{ " + "\"left\":" +std::to_string((int)faces[i].bbox.x1) + ", \"right\":"+std::to_string((int)faces[i].bbox.x2) + ", \"top\":" +std::to_string((int)faces[i].bbox.y1) + ", \"bottom\":" +std::to_string((int)faces[i].bbox.y2) + " }"+elEnd;
 }

json = json + "]";

std::cout << "json " << json << std::endl;

    env->ReleaseStringUTFChars(mdir, modelPath);
    env->ReleaseStringUTFChars(imagePath, imgPath);

  return env->NewStringUTF(json.c_str());
}
