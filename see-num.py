import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# Load the saved model architecture from JSON file
model_path = "model-num/model-num.json"
with open(model_path, 'r') as json_file:
    loaded_model_json = json_file.read()

# Create the model from the loaded JSON
loaded_model = model_from_json(loaded_model_json)

# Load weights into the model
loaded_model.load_weights("model-num/model-num.h5")
print(loaded_model.input_shape)

# Compile the model
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load class names from label.txt
label_path = "model-num/label-num.txt"
with open(label_path, "r") as label_file:
    class_names = label_file.read().splitlines()

# Set up the video capture
cap = cv2.VideoCapture(0)

# Define the ROI parameters
roi_size = 300
frame_size = 900

# Initialize counters
symbol_counters = {label: 0 for label in class_names}
word = ""
sentence = ""

blank_counter = 0

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

    # Convert the ROI to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(roi_gray, (5, 5), 2)

    # Apply adaptive thresholding
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    th3 = cv2.erode(th3, kernel, iterations=1)
    th3 = cv2.dilate(th3, kernel, iterations=1)
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

    # Increment the symbol counter
    symbol_counters[predicted_class_label] += 1

    # Display the predicted class label inside the white box
    cv2.putText(frame_resized, f"Predicted Sign: {predicted_class_label}", (10, frame_size - int(box_height) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    print(symbol_counters[predicted_class_label])
    # Update the word if the count is greater than 30
    if symbol_counters[predicted_class_label] >= 50:
        word += predicted_class_label
        # Check if the word contains 'mul' for multiplication
        if '*' in word:
            # Extract numeric labels before and after 'mul'
            parts = word.split('*')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                operand1 = int(parts[0])
                operand2 = int(parts[1])
                # Perform multiplication and update the word
                result = operand1 * operand2
                sentence = str(result)
            else:
                # Handle invalid 'mul' operation
                print("Invalid operation: Missing operand")

        # Check if the word contains 'add' for addition
        elif '+' in word:
            parts = word.split('+')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                operand1 = int(parts[0])
                operand2 = int(parts[1])
                result = operand1 + operand2
                sentence = str(result)
            else:
                print("Invalid operation: Missing operand")

        # Check if the word contains 'div' for division
        elif '/' in word:
            parts = word.split('/')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and int(parts[1]) != 0:
                operand1 = int(parts[0])
                operand2 = int(parts[1])
                result = operand1 / operand2
                sentence = str(result)
            else:
                print("Invalid operation: Missing or zero divisor")

        # Check if the word contains 'sub' for subtraction
        elif '-' in word:
            parts = word.split('-')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                operand1 = int(parts[0])
                operand2 = int(parts[1])
                result = operand1 - operand2
                sentence = str(result)
            else:
                print("Invalid operation: Missing operand")
        for label in class_names:
            symbol_counters[label] = 0   # Reset the counter after updating the word


    # Increment blank frame counter if predicted class label is 'blank'
    if predicted_class_label == "blank":
        blank_counter += 1
    else:
        blank_counter = 0

    # Check if consecutive blank frames are reached
    if blank_counter >= 30:
        # Reset word, sentence, and symbol counters after 20 consecutive blank frames
        word = ""
        sentence = ""
        for label in class_names:
            symbol_counters[label] = 0  

    # Display the live webcam feed with predicted class label, word, and sentence
    cv2.putText(frame_resized, f"Word: {word}", (10, frame_size - int(box_height) + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame_resized, f"Result: {sentence}", (10, frame_size - int(box_height) + 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow('Live Webcam Feed with Prediction', frame_resized)

    # Display the predicted class label
    print(f"Predicted Class: {predicted_class_label}")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
