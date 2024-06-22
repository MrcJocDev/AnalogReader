import cv2 # type:ignore
import numpy as np # type:ignore
import math

def detect_gauge_reading(image_path, min_angle=30, max_angle=330, min_value=-1, max_value=1):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect the circle (gauge) using Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=50, maxRadius=200)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        center_x, center_y, radius = circles[0]
        
        # Draw the circle and center on the original image
        cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)
        cv2.circle(image, (center_x, center_y), 2, (0, 0, 255), 3)

        # Create a mask to isolate the gauge area
        mask = np.zeros_like(gray)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)
        masked_image = cv2.bitwise_and(gray, gray, mask=mask)

        # Detect edges and lines to find the needle
        edges = cv2.Canny(masked_image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=30, maxLineGap=10)   # ////////////////////////////////////////////////////////////

        if lines is not None:
            needle_line = None
            max_length = 0
            for line in lines:
                for x1, y1, x2, y2 in line:
                    # Calculate the distance from line endpoints to the center
                    dist1 = np.sqrt((x1 - center_x) ** 2 + (y1 - center_y) ** 2)
                    dist2 = np.sqrt((x2 - center_x) ** 2 + (y2 - center_y) ** 2)
                    # Only consider lines that are within the radius of the gauge
                    if dist1 < radius and dist2 < radius:
                        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        if length > max_length:
                            max_length = length
                            needle_line = (x1, y1, x2, y2)

            if needle_line is not None:
                x1, y1, x2, y2 = needle_line

                # Draw the detected needle line
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Calculate the angle of the needle relative to the center
                angle = math.degrees(math.atan2(y2 - center_y, x2 - center_x))
                if angle < 0:
                    angle += 360

                # Calculate the angle relative to the gauge start angle
                needle_angle = angle - min_angle
                if needle_angle < 0:
                    needle_angle += 360

                # Map the angle to the pressure value
                angle_range = max_angle - min_angle
                value_range = max_value - min_value
                value = min_value + (needle_angle / angle_range) * value_range

                print(f'Gauge Center: ({center_x}, {center_y})')
                print(f'Needle Angle: {needle_angle:.2f} degrees')
                print(f'Gauge Reading: {value:.2f}')
                
                # Display the reading on the image
                cv2.putText(image, f'Reading: {value:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2)
            else:
                print("No needle line found")
        else:
            print("No needle detected")
    else:
        print("No gauge circle detected")

    # Display the results
    cv2.imshow('Detected Gauge and Needle', image)
    cv2.imwrite('img1.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage

detect_gauge_reading('images/45.jpg')
