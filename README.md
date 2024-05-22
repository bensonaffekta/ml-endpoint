# Description

## Face Detector Models
Models and other supporting files to assist the face detection benchmarking process

## Score Files
Various score files used for deploying the ML model in Azure

## deploy_model_to_azure.ipynb

## Benchmark Detection Methods
While researching more about face detection methods, realized CascadeClassifier is time-consuming and affects the performance of the endpoint. Upon further reading, we noticed that  DLIB CNN and OPENCV DNN had better performance compared to CascadeClassifier. This colab notebook consists the inference obtained after the benchmarking process. 

  Conclusion: We noticed that dlib cnn performs better among the three and has much better detection time as the number of images increase

