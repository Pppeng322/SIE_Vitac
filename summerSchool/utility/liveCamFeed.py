#!/usr/bin/env python3
import cv2

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit(1)

# Read and save the reference frame
print("Capturing reference frame... Please hold still.")
ret, ref_frame = cap.read()
if not ret:
    print("Error: Failed to grab reference frame")
    cap.release()
    exit(1)
print("Reference frame saved. Beginning live subtraction.")
print("Press 'q' to quit.")

# Display the reference for verification
cv2.imshow("Reference Frame", ref_frame)
cv2.waitKey(500)  # show for 500ms

# Main loop: grab new frames, subtract, and display
while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to grab frame")
        break

    # Compute absolute difference against reference in color
    diff = cv2.absdiff(frame, ref_frame)

    # Show only the diff image
    cv2.imshow("Live Diff", diff)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
