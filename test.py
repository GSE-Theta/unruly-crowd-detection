import tensorflow as tf
import cv2

model = tf.keras.models.load_model('model/baseline')

# Open the video file
cap = cv2.VideoCapture('dataset/Crowd-Activity-All.avi')

# Check if the video is opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Read and display each frame of the video
while cap.isOpened():
    # Read the next frame
    ret, frame = cap.read()
    
    # If the frame was not read successfully, break the loop
    if not ret:
        break
    
    input = frame[26:, :]
    input = tf.keras.applications.xception.preprocess_input(input)
    input = tf.expand_dims(input, axis=0)
    prediction = model.predict(input, verbose=0)[0]

    if tf.argmax(prediction) == 0:
        # Define the text to write
        text = 'Abnormal'

        # Define the font and scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1

        # Define the color and thickness
        color = (255, 0, 0)
        thickness = 2

        # Get the size of the text
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        # Calculate the position of the text
        text_x = int((frame.shape[1] - text_size[0]) / 2)
        text_y = int((frame.shape[0] + text_size[1]) / 2)
        org = (text_x, text_y)

        # Write the text on the image
        cv2.putText(frame, text, org, font, font_scale, color, thickness)

    # Display the frame
    cv2.imshow('Video', frame)
    
    # Wait for the user to press a key
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
