import cv2
import sys # Import sys for better exit handling

# --- Configuration ---
# Replace with your camera's actual RTSP URL, username, password, and IP
RTSP_URL = "rtsp://PuppyCam:8616Calvin@192.168.0.101:88/videoMain" 
# Consider using /videoSub for a lower resolution stream if performance is an issue.

print(f"Attempting to connect to: {RTSP_URL}")

# Create a VideoCapture object
# OpenCV can be particular about RTSP streams. Sometimes adding specific backend flags helps.
# For some cameras, CAP_FFMPEG or CAP_GSTREAMER might be needed.
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG) # Using FFMPEG backend

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream. Please check the RTSP URL, camera status, and network configuration.")
    print("If the URL is correct, try different OpenCV backend flags (e.g., cv2.CAP_ANY, cv2.CAP_GSTREAMER).")
    sys.exit(1) # Exit the script if connection fails

print("Successfully connected to the camera. Press 'q' to quit.")

while True:
    ret, frame = cap.read() # Read a frame from the video stream

    if not ret:
        print("Error: Failed to grab frame, stream ended or connection lost.")
        break # Exit loop if frame cannot be read

    # --- Your pet detection logic will go here ---
    # For now, just display the frame
    
    cv2.imshow('Foscam Pet Cam', frame) # Display the frame in a window

    # Press 'q' on the keyboard to exit the stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
print("Stream closed and windows destroyed.")