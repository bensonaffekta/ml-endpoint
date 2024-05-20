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
import time
from azureml.core.model import Model



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


def download_video(video_url):

  # download video from url
  response = requests.get(video_url, stream=True)

  if response.status_code == 200:
    video_data = b''

    for chunk in response.iter_content(chunk_size=512*512):
      video_data += chunk
   
    return video_data
 
  else:
    return None


def preprocess_video(video_data):
    # Create a temporary file to save the video data

    start_time = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        temp_video_file.write(video_data)
        temp_video_file_path = temp_video_file.name
    end_time = time.time()
    logging.info(f"  -- Saving video data to temp file took : {end_time - start_time} seconds")

    # Extract frames from video data
    start_time = time.time()
    cap = cv2.VideoCapture(temp_video_file_path)
    frames_base64 = []

    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
    frame_interval = 0.2  # 200 ms, which is the time interval you want between frames
    next_frame_time = 0  # Initialize next frame capture time

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current timestamp in seconds

            if not ret:
                break

            if current_time >= next_frame_time:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                buffered = io.BytesIO()
                pil_img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue())
                frames_base64.append(img_str.decode('utf-8'))
                next_frame_time += frame_interval

        cap.release()
        end_time = time.time()
        logging.info(f"  -- Frame count: {len(frames_base64)}")
        logging.info(f"  -- Extracting frames took : {end_time - start_time} seconds")
    
        return frames_base64

    finally:
        os.unlink(temp_video_file_path)



def preprocess(base64_string):

  try:
    
    # Decode the base64 string
    start_time = time.time()
    img_data = base64.b64decode(base64_string)
    end_time = time.time()
    logging.info(f"     -- Decoding base64 string took: {end_time - start_time}")
   
    # Convert binary data to image
    start_time = time.time()
    image = Image.open(io.BytesIO(img_data))
    end_time = time.time()
    logging.info(f"     -- Converting binary data to image  took: {end_time - start_time}")
   
   
    # Convert image to OpenCV format
    start_time = time.time()
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    end_time = time.time()
    logging.info(f"     -- Converting image to OpenCV took: {end_time - start_time}")
   
    # Load OpenCV face detector
    start_time = time.time()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    end_time = time.time()
    logging.info(f"     -- Loading OpenCV face detector took: {end_time - start_time}")
   
    # Convert to grayscale
    start_time = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    end_time = time.time()
    logging.info(f"     -- Converting to grayscale took: {end_time - start_time}")
   
   
    # Detect faces
    # endpoint array of pictures
    start_time = time.time()
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    end_time = time.time()
    logging.info(f"     -- Detecting faces from grayscale took: {end_time - start_time}")

    
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

def predict_frame(base64_string):

  try:

    start_time =  time.time()
    # step unnecessary?
    image_data = preprocess(base64_string)
    end_time = time.time()
    logging.info(f"  -- Preprocessing before prediction : {end_time - start_time} seconds")
    
    
    # change: ignore if no face exists
    if image_data is None:
      return 6     

    input_name = session.get_inputs()[0].name

    start_time = time.time()
    prediction = session.run(None, {input_name: image_data})[0]
    end_time = time.time()
    logging.info(f"  -- Prediction took : {end_time - start_time} seconds")
    
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
        # Attempt to parse the input data as JSON
        try:
            logging.info("1. Loading JSON data")
            start_time = time.time()
            parsed_data = json.loads(raw_data)
            end_time = time.time()

            logging.info(f"Loading JSON file took: {end_time - start_time} seconds")
            # If this succeeds, proceed with JSON-based handling
            if 'data' in parsed_data and parsed_data['data'].startswith('http'):
                video_url = parsed_data['data']
                video_data = download_video(video_url)
            elif 'payload' in parsed_data:
                
                
                video_base64 = parsed_data['payload']
                logging.info("2. Decoding base64 payload")
                start_time = time.time()
                video_data = base64.b64decode(video_base64)
                end_time = time.time()

                logging.info(f"Decoding base64 data took : {end_time - start_time} seconds")
            else:
                raise ValueError("Invalid JSON input format")
        except json.JSONDecodeError:
            # If JSON parsing fails, assume raw_data is binary and handle as video
            video_data = raw_data

        # Process the video data to extract and predict on frames
        logging.info("3. Video Pre-processing")
        start_time = time.time()
        frames_base64 = preprocess_video(video_data)
        end_time = time.time()
        logging.info(f"Video Pre-processing took : {end_time - start_time} seconds")

        # predict expression on each frame
        logging.info("4. Prediction on Frames")
        start_time = time.time()
        results = [predict_frame(frame) for frame in frames_base64]
        end_time = time.time()
        logging.info(f"Prediction on frames took : {end_time - start_time} seconds")

        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e)})
