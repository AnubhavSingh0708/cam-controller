import cv2
import numpy as np
import pyautogui

# --- SETTINGS ---
# Suppress the PyAutoGUI fail-safe to allow moving the mouse to the corner to stop.
pyautogui.FAILSAFE = False 

# Define the HSV color range for the RED anchor point
# You might need to tune these values for your specific lighting and red object
red_lower = np.array([0, 120, 70])
red_upper = np.array([10, 255, 255])
# An alternative range for red if the first one doesn't work well
# red_lower2 = np.array([170, 120, 70])
# red_upper2 = np.array([180, 255, 255])

# Define the HSV color range for the BLUE joystick point
# You might need to tune these values for your specific lighting and blue object
blue_lower = np.array([90, 80, 50])
blue_upper = np.array([130, 255, 255])

# Frame dimensions (can be adjusted)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Joystick deadzone radius (how far you need to move from the center to trigger an action)
DEAD_ZONE = 40

# --- GLOBAL VARIABLES ---
# Calibration state
calibrated = False
center_x, center_y = 0, 0 # This will store the calibrated center of the blue object

# --- HELPER FUNCTION ---
def find_colored_point(frame, lower_bound, upper_bound):
    """
    Finds the center of the largest contour of a specified color in a frame.
    Returns (x, y) coordinates or None if no contour is found.
    """
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the specified color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Optional: For red color which wraps around 0-180 in HSV, you might need two masks
    # mask = cv2.inRange(hsv, lower_bound1, upper_bound1)
    # mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
    # mask = cv2.bitwise_or(mask1, mask2)

    # Clean up the mask with morphological operations
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are found
    if len(contours) > 0:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        # Get the minimum enclosing circle
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        
        # Calculate the moments to find the centroid
        M = cv2.moments(c)
        if M["m00"] > 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # Only consider contours of a reasonable size
            if radius > 10:
                return center, int(radius)

    return None, None

# --- MAIN PROGRAM ---
# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a more intuitive "mirror" view
    frame = cv2.flip(frame, 1)

    # Find the red and blue points
    red_center, red_radius = find_colored_point(frame, red_lower, red_upper)
    blue_center, blue_radius = find_colored_point(frame, blue_lower, blue_upper)

    # --- CALIBRATION LOGIC ---
    if not calibrated:
        # Display instructions for calibration
        cv2.putText(frame, "Align points and press 'c' to calibrate", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if red_center and blue_center:
            # Draw circles around the detected points
            cv2.circle(frame, red_center, red_radius, (0, 0, 255), 3)
            cv2.circle(frame, blue_center, blue_radius, (255, 0, 0), 3)
    
    # --- JOYSTICK LOGIC (after calibration) ---
    else:
        direction_text = "NEUTRAL"
        if blue_center:
            # Draw the calibrated center and the dead zone
            cv2.circle(frame, (center_x, center_y), DEAD_ZONE, (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Draw the current position of the blue object
            cv2.circle(frame, blue_center, blue_radius, (255, 0, 0), 3)
            # Draw a line from the center to the current position
            cv2.line(frame, (center_x, center_y), blue_center, (0, 255, 255), 2)
            
            # Calculate the displacement vector
            dx = blue_center[0] - center_x
            dy = blue_center[1] - center_y
            
            # Check for movement outside the dead zone
            # Note: dy is inverted because y-coordinates increase downwards
            if abs(dx) > DEAD_ZONE or abs(dy) > DEAD_ZONE:
                if dy < -DEAD_ZONE:
                    direction_text = "UP"
                    pyautogui.press('w') # or 'up'
                elif dy > DEAD_ZONE:
                    direction_text = "DOWN"
                    pyautogui.press('s') # or 'down'
                
                if dx < -DEAD_ZONE:
                    direction_text += " LEFT" if direction_text != "NEUTRAL" else "LEFT"
                    pyautogui.press('a') # or 'left'
                elif dx > DEAD_ZONE:
                    direction_text += " RIGHT" if direction_text != "NEUTRAL" else "RIGHT"
                    pyautogui.press('d') # or 'right'

        # Display the current direction
        cv2.putText(frame, f"Direction: {direction_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'r' to re-calibrate", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    # --- DISPLAY THE FRAME ---
    cv2.imshow("CV Joystick", frame)

    # --- KEY HANDLING ---
    key = cv2.waitKey(1) & 0xFF

    # Calibrate on 'c'
    if key == ord('c') and not calibrated:
        if blue_center:
            center_x, center_y = blue_center
            calibrated = True
            print(f"Calibrated! Center at: ({center_x}, {center_y})")

    # Reset calibration on 'r'
    if key == ord('r'):
        calibrated = False
        print("Calibration reset.")

    # Quit on 'q'
    if key == ord('q'):
        break

# --- CLEANUP ---
cap.release()
cv2.destroyAllWindows()