#! /usr/bin/env python
import sys
import argparse
import cv2
import numpy
import pyautogui as m

class LaserTracker(object):

    def __init__(self, cam_width=640, cam_height=480,
                 hue_min_red=20, hue_max_red=160,
                 sat_min_red=100, sat_max_red=255,
                 val_min_red=200, val_max_red=256,
                 hue_min_green=65, hue_max_green=80,
                 sat_min_green=60, sat_max_green=255,
                 val_min_green=60, val_max_green=255,
                 display_thresholds=False):
        """
        * ``cam_width`` x ``cam_height`` -- This should be the size of the
        image coming from the camera. Default is 640x480.

        HSV color space Threshold values for a RED laser pointer are determined
        by:

            * ``hue_min_green``, ``hue_max_green`` -- Min/Max allowed hue_red values
        * ``sat_min_green``, ``sat_max_green`` -- Min/Max allowed Saturation values
        * ``val_min_green``, ``val_max_green`` -- Min/Max allowed pixel values

        If the dot from the laser pointer doesn't fall within these values, it
        will be ignored.

        * ``display_thresholds`` -- if True, additional windows will display
          values for threshold image channels.

        """

        self.cam_width = cam_width
        self.cam_height = cam_height

        #for red
        self.hue_min_red = hue_min_red
        self.hue_max_red = hue_max_red
        self.sat_min_red = sat_min_red
        self.sat_max_red = sat_max_red
        self.val_min_red = val_min_red
        self.val_max_red = val_max_red
        
        #for green
        self.hue_min_green = hue_min_green
        self.hue_max_green = hue_max_green
        self.sat_min_green = sat_min_green
        self.sat_max_green = sat_max_green
        self.val_min_green = val_min_green
        self.val_max_green = val_max_green
        
        self.display_thresholds = display_thresholds

        self.capture = None  # camera capture device
        self.channels = {
            'hue_red': None,
            'saturation_red': None,
            'value_red': None,
            'hue_green': None,
            'saturation_green': None,
            'value_green': None,
            'laser': None,
        }

        self.previous_position = None
        self.trail = numpy.zeros((self.cam_height, self.cam_width, 3),
                                 numpy.uint8)

    def create_and_position_window(self, name, xpos, ypos):
        """Creates a named widow placing it on the screen at (xpos, ypos)."""
        # Create a window
        cv2.namedWindow(name)
        # Resize it to the size of the camera image
        cv2.resizeWindow(name, self.cam_width, self.cam_height)
        # Move to (xpos,ypos) on the screen
        cv2.moveWindow(name, xpos, ypos)

    def setup_camera_capture(self, device_num=0):
        """Perform camera setup for the device number (default device = 0).
        Returns a reference to the camera Capture object.

        """
        try:
            device = int(device_num)
            sys.stdout.write("Using Camera Device: {0}\n".format(device))
        except (IndexError, ValueError):
            # assume we want the 1st device
            device = 0
            sys.stderr.write("Invalid Device. Using default device 0\n")

        # Try to start capturing frames
        self.capture = cv2.VideoCapture(device)
        if not self.capture.isOpened():
            sys.stderr.write("Failed to Open Capture device. Quitting.\n")
            sys.exit(1)

        # set the wanted image size from the camera
        self.capture.set(
            cv2.cv.CV_CAP_PROP_FRAME_WIDTH if cv2.__version__.startswith('2') else cv2.CAP_PROP_FRAME_WIDTH,
            self.cam_width
        )
        self.capture.set(
            cv2.cv.CV_CAP_PROP_FRAME_HEIGHT if cv2.__version__.startswith('2') else cv2.CAP_PROP_FRAME_HEIGHT,
            self.cam_height
        )
        return self.capture

    def handle_quit(self, delay=10):
        """Quit the program if the user presses "Esc" or "q"."""
        key = cv2.waitKey(delay)
        c = chr(key & 255)
        if c in ['c', 'C']:
            self.trail = numpy.zeros((self.cam_height, self.cam_width, 3),
                                     numpy.uint8)
        if c in ['q', 'Q', chr(27)]:
            sys.exit(0)

    def move_mouse(self,x,y):
    	new_x = x/640*1920
    	new_y = y/480*1080
    	print(str(new_x)+" "+str(new_y))
    	m.moveTo(1920-new_x,new_y)

    def click_mouse(self, x, y):
        new_x = x / 640 * 1920
        new_y = y / 480 * 1080
        print(str(new_x) + " " + str(new_y))
        m.dragTo(1920 - new_x, new_y)

    def threshold_image(self, channel):
        if channel == "hue_red":
            flag = 1
            minimum = self.hue_min_red
            maximum = self.hue_max_red
        elif channel == "saturation_red":
            flag = 1
            minimum = self.sat_min_red
            maximum = self.sat_max_red
        elif channel == "value_red":
            flag = 1
            minimum = self.val_min_red
            maximum = self.val_max_red
        elif channel == "hue_green":
            flag = 2
            minimum = self.sat_min_green
            maximum = self.sat_max_green
        elif channel == "saturation_green":
            flag = 2
            minimum = self.sat_min_green
            maximum = self.sat_max_green
        elif channel == "value_green":
            flag = 2
            minimum = self.val_min_green
            maximum = self.val_max_green

        (t, tmp) = cv2.threshold(
            self.channels[channel],  # src
            maximum,  # threshold value
            0,  # we dont care because of the selected type
            cv2.THRESH_TOZERO_INV  # t type
        )

        (t, self.channels[channel]) = cv2.threshold(
            tmp,  # src
            minimum,  # threshold value
            255,  # maxvalue
            cv2.THRESH_BINARY  # type
        )

        if channel == 'hue_red':
            # only works for filtering red color because the range for the hue_red
            # is split
            self.channels['hue_red'] = cv2.bitwise_not(self.channels['hue_red'])
        elif channel == 'hue_green':
            self.channels['hue_green'] = cv2.bitwise_not(self.channels['hue_green'])

    def track(self, frame, mask):
        """
        Track the position of the laser pointer.

        
                """
        center = None

        countours = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[-2]

        # only proceed if at least one contour was found
        if len(countours) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(countours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            #print(str(x)+" "+str(y))
            # if(flag==1):
            self.move_mouse(float(x), float(y))
            # elif(flag==2):
            #     self.click_mouse(float(x), float(y))
            moments = cv2.moments(c)
            if moments["m00"] > 0:
                center = int(moments["m10"] / moments["m00"]), \
                         int(moments["m01"] / moments["m00"])
            else:
                center = int(x), int(y)

            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                cv2.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 255, 0), -1)
                # then update the ponter trail
                if self.previous_position:
                    cv2.line(self.trail, self.previous_position, center,
                             (255, 255, 255), 2)



        #cv2.add(self.trail, frame, frame)
        #self.previous_position = center

    def detect(self, frame):
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # split the video frame into color channels
        h, s, v = cv2.split(hsv_img)
        # print(s);
        # if(200<v<256):
        self.channels['hue_red'] = h
        self.channels['saturation_red'] = s
        self.channels['value_red'] = v
        self.threshold_image("hue_red")
        self.threshold_image("saturation_red")
        self.threshold_image("value_red")
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['hue_red'],
            self.channels['value_red']
        )
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['saturation_red'],
            self.channels['laser']
        )

        hsv_image = cv2.merge([
            self.channels['hue_red'],
            self.channels['saturation_red'],
            self.channels['value_red'],
        ])

        # self.track(frame, self.channels['laser'])
        #
        # return hsv_image

        # elif(<v<255):
        self.channels['hue_green'] = h
        self.channels['saturation_green'] = s
        self.channels['value_green'] = v
        self.threshold_image("hue_green")
        self.threshold_image("saturation_green")
        self.threshold_image("value_green")
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['hue_green'],
            self.channels['value_green']
        )
        self.channels['laser'] = cv2.bitwise_and(
            self.channels['saturation_green'],
            self.channels['laser']
        )

        hsv_image = cv2.merge([
            self.channels['hue_green'],
            self.channels['saturation_green'],
            self.channels['value_green'],
        ])

        self.track(frame, self.channels['laser'])

        return hsv_image

    def display(self, img, frame):
        """Display the combined image and (optionally) all other image channels
        NOTE: default color space in OpenCV is BGR.
        """
        cv2.imshow('RGB_VideoFrame', frame)
        cv2.imshow('LaserPointer', self.channels['laser'])
        if self.display_thresholds:
            cv2.imshow('Thresholded_HSV_Image', img)
            cv2.imshow('hue_red', self.channels['hue_red'])
            cv2.imshow('Saturation', self.channels['saturation'])
            cv2.imshow('Value', self.channels['value'])

    def setup_windows(self):
        sys.stdout.write("Using OpenCV version: {0}\n".format(cv2.__version__))

        # create output windows
        self.create_and_position_window('LaserPointer', 0, 0)
        self.create_and_position_window('RGB_VideoFrame',
                                        10 + self.cam_width, 0)
        if self.display_thresholds:
            self.create_and_position_window('Thresholded_HSV_Image', 10, 10)
            self.create_and_position_window('hue_red', 20, 20)
            self.create_and_position_window('Saturation', 30, 30)
            self.create_and_position_window('Value', 40, 40)

    def run(self):
        # Set up window positions
        self.setup_windows()
        # Set up the camera capture
        self.setup_camera_capture()

        while True:
            # 1. capture the current image
            success, frame = self.capture.read()
            if not success:  # no image captured... end the processing
                sys.stderr.write("Could not read camera frame. Quitting\n")
                sys.exit(1)

            hsv_image = self.detect(frame)
            self.display(hsv_image, frame)
            self.handle_quit()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Run the Laser Tracker')
    # parser.add_argument('-W', '--width',
    #                     default=640,
    #                     type=int,
    #                     help='Camera Width')
    # parser.add_argument('-H', '--height',
    #                     default=480,
    #                     type=int,
    #                     help='Camera Height')
    # parser.add_argument('-u', '--hue_redmin',
    #                     default=65,
    #                     type=int,
    #                     help='hue_red Minimum Threshold')
    # parser.add_argument('-U', '--hue_redmax',
    #                     default=80,
    #                     type=int,
    #                     help='hue_red Maximum Threshold')
    # parser.add_argument('-s', '--satmin',
    #                     default=60,
    #                     type=int,
    #                     help='Saturation Minimum Threshold')
    # parser.add_argument('-S', '--satmax',
    #                     default=255,
    #                     type=int,
    #                     help='Saturation Maximum Threshold')
    # parser.add_argument('-v', '--valmin',
    #                     default=60,
    #                     type=int,
    #                     help='Value Minimum Threshold')
    # parser.add_argument('-V', '--valmax',
    #                     default=255,
    #                     type=int,
    #                     help='Value Maximum Threshold')
    # parser.add_argument('-d', '--display',
    #                     action='store_true',
    #                     help='Display Threshold Windows')
    # params = parser.parse_args()

    tracker = LaserTracker(
        cam_width=640,
        cam_height=480,
        hue_min_red=20, hue_max_red=160,
        sat_min_red=100, sat_max_red=255,
        val_min_red=200, val_max_red=256,
        hue_min_green=65,hue_max_green=80,
        sat_min_green=60, sat_max_green=255,
        val_min_green=60, val_max_green=255,
        display_thresholds=False
    )
    tracker.run()
