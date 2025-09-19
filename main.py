import cv2
import numpy as np
import pydirectinput


# Define the HSV color range for the RED anchor point
red_lower = np.array([0, 120, 70])
red_upper = np.array([10, 255, 255])

# Define the HSV color range for the BLUE joystick point
blue_lower = np.array([90, 80, 50])
blue_upper = np.array([130, 255, 255])

# Frame dimensions
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Joystick deadzone radius
DEAD_ZONE = 40

# --- GLOBAL VARIABLES ---
calibrated = False
center_x, center_y = 0, 0
blue_visible_last_frame = False # <<< NEW: State variable to track if blue was visible

# --- HELPER FUNCTION ---
def find_colored_point(frame, lower_bound, upper_bound):
    """
    Finds the center of the largest contour of a specified color in a frame.
    Returns (x, y) coordinates or None if no contour is found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if M["m00"] > 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if radius > 10:
                return center, int(radius)
    return None, None

# --- MAIN PROGRAM ---
cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    red_center, red_radius = find_colored_point(frame, red_lower, red_upper)


    if not calibrated:
        cv2.putText(frame, "Align points and press 'c' to calibrate", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if red_center:
            cv2.circle(frame, red_center, red_radius, (255, 0, 0), 3)
    
    else:

        pydirectinput.keyUp('s')
        pydirectinput.keyUp('d')
        pydirectinput.keyUp('w')
        pydirectinput.keyUp('a')
        pydirectinput.keyUp('space')
        # <<< MODIFIED SECTION START >>>
        direction_text = "NEUTRAL"
        
        # --- Logic if the BLUE point IS visible ---
        if red_center:
            cv2.circle(frame, (center_x, center_y), DEAD_ZONE, (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, red_center, red_radius, (255, 0, 0), 3)
            cv2.line(frame, (center_x, center_y), red_center, (0, 255, 255), 2)
            
            dx = red_center[0] - center_x
            dy = red_center[1] - center_y
            
            if abs(dx) > DEAD_ZONE or abs(dy) > DEAD_ZONE:
                if dy < -DEAD_ZONE:
                    direction_text = "Down"
                    pydirectinput.keyDown('s')
                elif dy > DEAD_ZONE:
                    direction_text = "UP"
                    pydirectinput.keyDown('w')
                
                if dx < -DEAD_ZONE:
                    direction_text += " LEFT" if direction_text != "NEUTRAL" else "LEFT"
                    pydirectinput.keyDown('a')
                elif dx > DEAD_ZONE:
                    direction_text += " RIGHT" if direction_text != "NEUTRAL" else "RIGHT"
                    pydirectinput.keyDown('d')
        
        # --- Logic if the BLUE point is NOT visible ---
        else:
            direction_text = "ACTION (SPACE)"
            # If the point was visible in the last frame but is gone now, press space
            if blue_visible_last_frame:
                pydirectinput.keyDown('space')
                print("Red object disappeared. Pressing SPACE.")

        # Update the state for the next frame
        blue_visible_last_frame = red_center is not None
        # <<< MODIFIED SECTION END >>>

        cv2.putText(frame, f"Direction: {direction_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'r' to re-calibrate", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("CV Joystick", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and not calibrated:
        if red_center:
            center_x, center_y = red_center
            calibrated = True
            # <<< NEW: Reset visibility state on calibration
            blue_visible_last_frame = True 
            print(f"Calibrated! Center at: ({center_x}, {center_y})")

    if key == ord('r'):
        calibrated = False
        print("Calibration reset.")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()