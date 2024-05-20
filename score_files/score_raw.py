import json
import numpy as np
import logging
import onnxruntime as rt
import base64
from PIL import Image
import io
import cv2
import os
import requests
import time
import tempfile
from azureml.core.model import Model

from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse


def init():
    logging.basicConfig(level=logging.DEBUG)
    global session

    try:
        # Use the model name and optionally specify the version
        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "Final_model_2.onnx")
        logging.debug(f"Model found at path: {model_path}")

        # Load the model using the path
        session = rt.InferenceSession(model_path)
        logging.debug("Model loaded successfully")

    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        return json.dumps({
            "error": "Unable to load model from path",
            "exception": str(e)
        })


# def preprocess_video(video_data):
#   # Create a temporary file to save the video data
#   with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
#     temp_video_file.write(video_data)
#     temp_video_file_path = temp_video_file.name

#   # extract frames from video data
#   cap = cv2.VideoCapture(temp_video_file_path)
#   frames = []

  
#   try:
#     while cap.isOpened():

#       ret, frame = cap.read()
#       if not ret:
#         break
      
#       frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#       pil_img = Image.fromarray(frame_rgb)
#       buffered = io.BytesIO()
#       pil_img.save(buffered, format="JPEG")
#       frames.append(buffered.getvalue())

#     cap.release()
#     return frames
  
#   finally:
#     os.unlink(temp_video_file_path)


def preprocess(img_data):

  try:
    # Convert binary data to image
    start_time = time.time()
    image = Image.open(io.BytesIO(img_data))
    end_time = time.time()
    logging.info(f"Binary data converted to: {type(image)} and took: {end_time-start_time}s")

    # Convert image to OpenCV format
    start_time = time.time()
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) 
    end_time = time.time()
    logging.info(f"Image converted to: {type(image)} and took: {end_time-start_time}s")

    # Load OpenCV face detector
    start_time = time.time()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    end_time = time.time()
    logging.info(f"Loading OpenCV face detector took: {end_time - start_time}s")
    
    # Convert to grayscale
    start_time = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    end_time = time.time()
    logging.info(f"Conversion to grayscale took: {end_time-start_time}")
    
    # Detect faces
    start_time = time.time()
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    end_time = time.time()
    logging.info(f"Faces detected and took: {end_time - start_time}")

    for x, y, w, h in faces:
        roi_color = frame[y:y+h, x:x+w]
        final_image = cv2.resize(roi_color, (224, 224))  # Resize to match model's expected input dimensions
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image.astype(np.float32) / 255.0  # Normalize the image
        return final_image
    
    return None  # Return None if no faces are detected

  except Exception as e:
    return json.dumps({
      "error":"Preprocessing failed"
    })

def predict_frame(img_data):
  logging.info("In predict frame function")
  try:
    logging.info("In preprocessing")
    image_data = preprocess(img_data)

    # change: ignore if no face exists
    if image_data is None:
      return 6

    input_name = session.get_inputs()[0].name
    prediction = session.run(None, {input_name: image_data})[0]

    # Process the result
    prediction = np.squeeze(prediction)
    p_class = np.argmax(prediction)

    return int(p_class)

  except Exception as e:
    return json.dumps({
      "error":e
    })



@rawhttp
def run(request):
    print("This is run()")
    
    if request.method == 'GET':
        # For this example, just return the URL for GETs.
        respBody = str.encode(request.full_path)
        return AMLResponse(respBody, 200)
    elif request.method == 'POST':
        images = request.files.getlist("images")
        logging.info(f"Images size: {len(images)}")
        predictions = []

        for img in images:
           img_data = img.read()
           logging.info(f"Image data type: {type(img_data)}")
           predicted_class = predict_frame(img_data)
           logging.info(f"predicted class: {predicted_class}")
           predictions.append(predicted_class)
        logging.info(f"Predictions size: {len(predictions)}")
        return AMLResponse(json.dumps({"predictions": predictions}), 200)

    else:
        return AMLResponse("bad request", 500)
