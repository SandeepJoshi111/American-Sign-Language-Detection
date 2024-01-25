import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# Load the saved model architecture from JSON file
model_path = "models/model-bw.json"
with open(model_path, 'r') as json_file:
    loaded_model_json = json_file.read()

# Create the model from the loaded JSON
loaded_model = model_from_json(loaded_model_json)

# Load weights into the model
loaded_model.load_weights("models/model-bw.h5")
print(loaded_model.input_shape)

# Compile the model
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load class names from label.txt
label_path = "models/label.txt"
with open(label_path, "r") as label_file:
    class_names = label_file.read().splitlines()

# Set up the video capture
cap = cv2.VideoCapture(0)

# Define the ROI parameters
roi_size = 300
frame_size = 900

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Resize the frame to 900x900
    frame_resized = cv2.resize(frame, (frame_size, frame_size))

    # Flip the frame horizontally
    frame_resized = cv2.flip(frame_resized, 1)

    # Create a black frame for the ROI
    roi_frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)

    # Copy the flipped webcam frame to the ROI region

    roi_frame[0:roi_size, frame_size-roi_size:frame_size, :] = frame_resized[0:roi_size, 0:roi_size, :]
    # Draw a border around the ROI
    cv2.rectangle(frame_resized, (frame_size-roi_size, 0), (frame_size, roi_size), (0, 255, 0), 2)

    # Create a white box inside frame_resized
    box_height = 200
    cv2.rectangle(frame_resized, (0, frame_size - box_height), (frame_size, frame_size), (255, 255, 255), -1)

    # Extract the ROI for prediction
    roi = frame_resized[0:roi_size, frame_size-roi_size:frame_size, :]

    # Extract the ROI for prediction
    roi = frame_resized[0:roi_size, frame_size-roi_size:frame_size, :]

    # Convert the ROI to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(roi_gray, (5, 5), 2)

    # Apply adaptive thresholding
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # # Erosion and dilation
    # kernel = np.ones((2, 2), np.uint8)
    # th3 = cv2.erode(th3, kernel, iterations=1)
    # th3 = cv2.dilate(th3, kernel, iterations=1)

    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # Display the frame with ROI and predicted class label
    cv2.imshow('Thresholded ROI', res)

    # Make predictions
    img = cv2.resize(res, (128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    predictions = loaded_model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    # Get the predicted class label
    predicted_class_label = class_names[predicted_class_index]

    # Display the predicted class label inside the white box
    cv2.putText(frame_resized, f"Predicted Class: {predicted_class_label}", (10, frame_size - int(box_height) + 40),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2) 

    # Display the live webcam feed with predicted class label
    cv2.imshow('Live Webcam Feed with Prediction', frame_resized)

    # Display the predicted class label 
    print(f"Predicted Class: {predicted_class_label}")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
