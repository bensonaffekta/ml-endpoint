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
import tempfile
from azureml.core.model import Model


def init():
    logging.basicConfig(level=logging.DEBUG)
    global session
    model_path = Model.get_model_path('emotional_state_model')
    
    logging.debug("Model found from path")
    
    try:
        logging.debug("Model is being loaded")
        session = rt.InferenceSession(model_path)
        logging.debug("Model loaded successfully")
    except Exception as e:

        return json.dumps({
            "error":"Unable to load model from path"
        })



def preprocess_video(video_data_base64):
  # Create a temporary file to save the video data
  with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
    temp_video_file.write(video_data)
    temp_video_file_path = temp_video_file.name

  # extract frames from video data
  cap = cv2.VideoCapture(temp_video_file_path)
  frames_base64 = []

  
  try:
    while cap.isOpened():

      ret, frame = cap.read()
      if not ret:
        break
      
      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      pil_img = Image.fromarray(frame_rgb)
      buffered = io.BytesIO()
      pil_img.save(buffered, format="JPEG")
      img_str = base64.b64encode(buffered.getvalue())
      frames_base64.append(img_str.decode('utf-8'))

    cap.release()
    return frames_base64
  
  finally:
    os.unlink(temp_video_file_path)


def preprocess(base64_string):

  try:

    # Decode the base64 string
    img_data = base64.b64decode(base64_string)
    
    # Convert binary data to image
    image = Image.open(io.BytesIO(img_data))
    
    # Convert image to OpenCV format
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Load OpenCV face detector
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        roi_color = frame[y:y+h, x:x+w]
        final_image = cv2.resize(roi_color, (224, 224))  # Resize to match model's expected input dimensions
        final_image = np.expand_dims(final_image, axis=0)
        final_image = final_image.astype(np.float32) / 255.0  # Normalize the image
        image_list = final_image.tolist()
        return final_image
    
    return None  # Return None if no faces are detected

  except Exception as e:
    return json.dumps({
      "error":"Preprocessing failed"
    })

def predict_frame(base64_string):

  try:

    image_data = preprocess(base64_string)

    # change: ignore if no face exists
    if image_data is None:
      return 6


      

    input_name = session.get_inputs()[0].name
    prediction = session.run(None, {input_name: image_data})[0]

    # Process the result
    prediction = np.squeeze(prediction)
    p_class = np.argmax(prediction)

    predicted_class = convert_int64(p_class)


    return predicted_class

  except Exception as e:
    return json.dumps({
      "error":e
    })

def convert_int64(obj):
  if isinstance(obj, np.int64):
    return int(obj)
  raise TypeError

def run(raw_data):
    
    try:

        parsed_data = json.loads(raw_data)
        logging.debug("Data is parsed")

        # video_url = parsed_data["data"]

        # ensure data in expected format: a list rep
        video_data = parsed_data["data"]

        if video_data is None:
          raise ValueError("Failed to download video")
          
        frames_base64 = preprocess_video(video_data)

        results = [predict_frame(frame) for frame in frames_base64]

        return json.dumps(results)
    except Exception as e:
        return json.dumps({
            "error":e
        })