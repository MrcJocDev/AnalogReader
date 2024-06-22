import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

while True:
    def read_gauge(image_path):
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or path is incorrect")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(blurred,
                                   cv2.HOUGH_GRADIENT,
                                   dp=1.2,
                                   minDist=100,
                                   param1=50,
                                   param2=30,
                                   minRadius=0,
                                   maxRadius=0)

        if circles is None:
            raise ValueError("No circles detected")

        circles = np.round(circles[0, :]).astype("int")

        # Assume the first circle is our gauge
        gauge_circle = circles[0]
        x, y, r = gauge_circle
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        cv2.circle(image, (x, y), 2, (0, 0, 255), 2)

        # Crop the gauge area
        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), r, 255, thickness=-1)  # type:ignore

        #line detection
        # image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or path is incorrect")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        masked_gray = cv2.bitwise_and(gray, mask)

        # Edge detection
        edges = cv2.Canny(masked_gray, 50, 150, apertureSize=3)

        # Line detection using HoughLinesP
        lines = cv2.HoughLinesP(edges,
                                1,
                                np.pi / 180,
                                threshold=100,
                                minLineLength=50,
                                maxLineGap=10)

        if lines is None:
            raise ValueError("No lines detected")

        # Find the longest line, assuming it's the needle
        max_len = 0
        needle_line = None
        for line in lines:
            for x1, y1, x2, y2 in line:
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                #cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                if length > max_len:
                    max_len = length
                    needle_line = (x1, y1, x2, y2)

        if needle_line is None:
            raise ValueError("No needle detected")


        x1, y1, x2, y2 = needle_line
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        #CREATE ANOTHER LINE FOR ZERO READING
        zero_line = None
        length_n = np.sqrt((x1 - x)**2 + (y1 - y)**2)

        x4 = int(x + (length_n * math.cos(np.radians(135))))
        y4 = int(y + (length_n * math.sin(np.radians(135))))

        numerator = (y1 * (x - x4)) + (y * (x4 - x1)) + (y4 * (x1 - x))
        denominator = (x1 - x) * (x - x4) + (y1 - y) * (y - y4)
        ratio = numerator / denominator
        print(x, y, x1, y1, x4, y4)
        angle_cp = np.degrees(math.atan(ratio))

        #x4 = x  , y2) = (x1 + l⋅cos(a), y1 + l⋅sin(a))

        # Calculate angle of the needle
        angle = np.degrees(np.arctan2(y2 - y, x2 - x))

        # Normalize angle to the range [0, 360)
        angle = (angle + 360) % 360

        # Assuming the gauge's minimum and maximum values correspond to specific angles:
        min_angle = 45 # Example angle for minimum value
        max_angle = 315  # Example angle for maximum value

        # Calculate the value range
        min_value = 0  # Example minimum value
        max_value = 100  # Example maximum value

        # Map angle to value
        if min_angle <= angle_cp <= max_angle:
            value = min_value + (angle_cp) * ((max_value - min_value) / (315 - 45))

        else:
            value = max_value if angle > max_angle else min_value

        cv2.line(image, (x, y), (x4, y4), (0, 255, 0), 2)

        # Draw the results
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Reading: {value:.2f}')
        plt.show()


        print(x, x1, x2, y, y1, y2)
        return value, min_angle, max_angle, angle

    # Example usage
    imgNum = input("enter image number")
    image_path = 'images/'+imgNum+'.jpg'
    value, min_angle, max_angle, angle = read_gauge(image_path)
    print(f'Reading: {value:.2f}')
    print(
        f'Min Angle: {min_angle}°, Max Angle: {max_angle}°, Needle Angle: {angle}°'
    )
