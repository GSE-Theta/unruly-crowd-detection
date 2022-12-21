from keras.applications.xception import preprocess_input
from tensorflow import expand_dims, argmax
from model import get_model
import cv2

model = get_model()
model.load_weights('model/fold-2_weights.h5')

# Open the video file
cap = cv2.VideoCapture('video/Crowd-Activity-All.avi')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('video/output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))

counter = [0, 0]
detect = False

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
    
    input = cv2.resize(frame, (320, 214), interpolation=cv2.INTER_LINEAR)
    input = preprocess_input(input)
    input = expand_dims(input, axis=0)
    prediction = model.predict(input, verbose=0)[0]

    if argmax(prediction) == 0:
        counter[1] = 0
        if counter[0] < 30:
            counter[0] += 1
        else:
            counter[0] = 0
            detect = True
    elif detect:
        counter[0] = 0
        if counter[1] < 10:
            counter[1] += 1
        else:
            counter[1] = 0
            detect = False
    
    if detect:
        # Define the text to write
        text = 'Abnormal'

        # Define the font and scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1

        # Define the color and thickness
        color = (0, 0, 255)
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
    out.write(frame)
    
    # Wait for the user to press a key
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
out.release()

# Destroy all windows
cv2.destroyAllWindows()
