import threading
import time
from typing import TYPE_CHECKING, Tuple

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QDoubleSpinBox

from database import Database

if TYPE_CHECKING:
    from user_interface import UserInterface


class FiberCamera(QWidget):
    use_binary_for_edges = True
    CALIBRATION_DIAMETER_MM = 0.94

    def __init__(self, target_diameter: QDoubleSpinBox,
                 gui: "UserInterface") -> None:
        super().__init__()

        self.raw_image = QLabel()
        self.canny_image = QLabel()
        self.processed_image = QLabel()

        self.target_diameter = target_diameter
        self.gui = gui

        self.capture = cv2.VideoCapture(0)
        self.capture_lock = threading.Lock()

        if not self.capture.isOpened():
            print("WARNING: Could not open camera.")

        try:
            self.diameter_coefficient = float(
                Database.get_calibration_data("diameter_coefficient")
            )
        except (TypeError, ValueError):
            self.diameter_coefficient = -1.0

        self.previous_time = 0.0

        self.erode_enabled = True
        self.dilate_enabled = True
        self.gaussian_enabled = True
        self.binary_enabled = True

        # These values are changed by the sliders in user_interface.py
        self.canny_lower = 100
        self.canny_upper = 250
        self.hough_threshold = 30

    def _read_frame(self):
        with self.capture_lock:
            success, frame = self.capture.read()

        if not success or frame is None:
            return None

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _crop_frame(frame):
        height = frame.shape[0]
        return frame[height // 4:3 * height // 4, :]

    def _detect_lines(self, frame):
        edges, binary_frame = self.get_edges(frame)

        detected_lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            max(1, int(self.hough_threshold)),
            minLineLength=30,
            maxLineGap=100
        )

        return edges, binary_frame, detected_lines

    @staticmethod
    def _segments(lines):
        """
        Makes HoughLinesP compatible with both:
        (N, 1, 4) and (N, 4) OpenCV output formats.
        """
        if lines is None:
            return []

        segments = []

        for line in lines:
            values = np.asarray(line).reshape(-1)

            if values.size < 4:
                continue

            x0, y0, x1, y1 = values[:4]

            segments.append((
                int(x0), int(y0),
                int(x1), int(y1)
            ))

        return segments

    @classmethod
    def _diameter_pixels(cls, lines):
        segments = cls._segments(lines)

        if len(segments) <= 1:
            return 0.0

        left_positions = [
            min(x0, x1)
            for x0, _, x1, _ in segments
        ]

        right_positions = [
            max(x0, x1)
            for x0, _, x1, _ in segments
        ]

        diameter = (
            (max(left_positions) - min(left_positions))
            +
            (max(right_positions) - min(right_positions))
        ) / 2.0

        if not np.isfinite(diameter) or diameter <= 0:
            return 0.0

        return float(diameter)

    def camera_loop(self) -> None:
        try:
            current_time = time.time()

            frame = self._read_frame()

            if frame is None:
                print("Camera warning: failed to capture frame.")
                return

            frame = self._crop_frame(frame)

            edges, binary_frame, detected_lines = \
                self._detect_lines(frame)

            frame_with_lines = self.plot_lines(
                frame.copy(),
                detected_lines
            )

            self._show_rgb(
                self.raw_image,
                frame_with_lines
            )

            self._show_gray(
                self.canny_image,
                edges
            )

            self._show_gray(
                self.processed_image,
                binary_frame
            )

            self.previous_time = current_time

        except Exception as e:
            print(f"Error in camera loop: {e}")

    def get_edges(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        kernel = np.ones((5, 5), np.uint8)

        if self.erode_enabled:
            gray = cv2.erode(
                gray, kernel, iterations=2
            )

        if self.dilate_enabled:
            gray = cv2.dilate(
                gray, kernel, iterations=2
            )

        if self.gaussian_enabled:
            gray = cv2.GaussianBlur(
                gray, (5, 5), 0
            )

        if self.binary_enabled:
            _, binary_frame = cv2.threshold(
                gray,
                100,
                255,
                cv2.THRESH_BINARY
            )
        else:
            binary_frame = gray.copy()

        if FiberCamera.use_binary_for_edges:
            canny_source = binary_frame
        else:
            canny_source = gray

        lower = max(0, int(self.canny_lower))
        upper = max(lower + 1, int(self.canny_upper))

        edges = cv2.Canny(
            canny_source,
            lower,
            upper,
            apertureSize=3
        )

        return edges, binary_frame

    def get_fiber_diameter(self, lines):
        diameter_pixels = self._diameter_pixels(lines)

        if diameter_pixels <= 0:
            return 0.0

        try:
            coefficient = float(self.diameter_coefficient)
        except (TypeError, ValueError):
            return 0.0

        if not np.isfinite(coefficient) or coefficient <= 0:
            return 0.0

        return diameter_pixels * coefficient

    def get_fiber_diameter_in_pixels(self, lines):
        return self._diameter_pixels(lines)

    def plot_lines(self, frame, lines):
        for x0, y0, x1, y1 in self._segments(lines):
            cv2.line(
                frame,
                (x0, y0),
                (x1, y1),
                (255, 0, 0),
                2
            )

        return frame

    def calibrate(self):
        num_samples = 50
        valid_samples = []

        print("Starting camera calibration...")

        for _ in range(num_samples):
            frame = self._read_frame()

            if frame is None:
                continue

            # Use the same ROI used during normal measurement.
            frame = self._crop_frame(frame)

            _, _, detected_lines = \
                self._detect_lines(frame)

            diameter_pixels = \
                self.get_fiber_diameter_in_pixels(
                    detected_lines
                )

            if (
                np.isfinite(diameter_pixels)
                and diameter_pixels > 0
            ):
                valid_samples.append(
                    float(diameter_pixels)
                )

            time.sleep(0.02)

        if len(valid_samples) == 0:
            print(
                "Calibration failed: "
                "no valid fiber was detected."
            )
            return False

        average_diameter = float(
            np.mean(valid_samples)
        )

        if (
            not np.isfinite(average_diameter)
            or average_diameter <= 0
        ):
            print(
                "Calibration failed: "
                "invalid average diameter."
            )
            return False

        new_coefficient = (
            self.CALIBRATION_DIAMETER_MM
            / average_diameter
        )

        if (
            not np.isfinite(new_coefficient)
            or new_coefficient <= 0
        ):
            print(
                "Calibration failed: "
                "invalid calibration coefficient."
            )
            return False

        self.diameter_coefficient = \
            float(new_coefficient)

        print(
            f"Average width of wire: "
            f"{average_diameter:.3f} pixels"
        )

        print(
            f"Diameter coefficient: "
            f"{self.diameter_coefficient:.8f} mm/pixel"
        )

        Database.update_calibration_data(
            "diameter_coefficient",
            str(self.diameter_coefficient)
        )

        print("Camera calibration completed.")
        return True

    def camera_feedback(self, current_time: float) -> None:
        try:
            frame = self._read_frame()

            if frame is None:
                return

            frame = self._crop_frame(frame)

            _, _, detected_lines = \
                self._detect_lines(frame)

            current_diameter = \
                self.get_fiber_diameter(
                    detected_lines
                )

            self.gui.diameter_plot.update_plot(
                current_time,
                current_diameter,
                self.target_diameter.value()
            )

            Database.camera_timestamps.append(
                current_time
            )

            Database.diameter_readings.append(
                current_diameter
            )

            Database.diameter_setpoint.append(
                self.target_diameter.value()
            )

            Database.diameter_delta_time.append(
                current_time - self.previous_time
            )

            self.previous_time = current_time

        except Exception as e:
            print(
                f"Error in camera feedback: {e}"
            )

    @staticmethod
    def _show_rgb(label, image):
        image = np.ascontiguousarray(image)

        height, width = image.shape[:2]

        qimage = QImage(
            image.data,
            width,
            height,
            image.strides[0],
            QImage.Format_RGB888
        ).copy()

        label.setPixmap(
            QPixmap.fromImage(qimage)
        )

    @staticmethod
    def _show_gray(label, image):
        image = np.ascontiguousarray(image)

        height, width = image.shape[:2]

        qimage = QImage(
            image.data,
            width,
            height,
            image.strides[0],
            QImage.Format_Grayscale8
        ).copy()

        label.setPixmap(
            QPixmap.fromImage(qimage)
        )

    def closeEvent(self, event):
        with self.capture_lock:
            if self.capture is not None:
                self.capture.release()

        event.accept()