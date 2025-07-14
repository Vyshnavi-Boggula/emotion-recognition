# Import required libraries
import cv2  # OpenCV for video capture and image processing
from facial_emotion_recognition import EmotionRecognition  # Pre-trained emotion recognition model

def main():
    # Initialize the EmotionRecognition model (using CPU)
    er = EmotionRecognition(device='cpu')

    # Open the default webcam (0 = built-in webcam, 1 = external webcam)
    cam = cv2.VideoCapture(0)

    # Check if the webcam is accessible
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press ESC to exit.")

    # Start capturing frames in a loop
    while True:
        # Read a frame from the webcam
        success, frame = cam.read()
        
        # If frame was not read successfully, exit loop
        if not success:
            print("Failed to grab frame.")
            break

        # Detect and annotate emotions on the frame
        frame = er.recognise_emotion(frame, return_type='BGR')

        # Display the annotated frame in a window
        cv2.imshow("Emotion Recognition", frame)

        # Wait for 1 millisecond and check if ESC (27) key was pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release the webcam and close all OpenCV windows
    cam.release()
    cv2.destroyAllWindows()

# Run the main function when the script is executed
if __name__ == "__main__":
    main()
