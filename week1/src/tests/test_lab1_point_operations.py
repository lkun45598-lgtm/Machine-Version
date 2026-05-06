import unittest

import numpy as np

from lab1_point_operations import (
    apply_pseudocolor,
    equalize_histogram,
    gray_inversion,
    threshold_image,
    to_grayscale,
)


class PointOperationTests(unittest.TestCase):
    def test_to_grayscale_converts_bgr_image_to_uint8_gray(self):
        bgr = np.array(
            [
                [[0, 0, 0], [255, 255, 255]],
                [[0, 0, 255], [0, 255, 0]],
            ],
            dtype=np.uint8,
        )

        gray = to_grayscale(bgr)

        self.assertEqual(gray.shape, (2, 2))
        self.assertEqual(gray.dtype, np.uint8)
        self.assertEqual(int(gray[0, 0]), 0)
        self.assertEqual(int(gray[0, 1]), 255)

    def test_gray_inversion_maps_each_pixel_to_255_minus_value(self):
        gray = np.array([[0, 127], [200, 255]], dtype=np.uint8)

        inverted = gray_inversion(gray)

        np.testing.assert_array_equal(
            inverted,
            np.array([[255, 128], [55, 0]], dtype=np.uint8),
        )

    def test_threshold_image_uses_binary_threshold(self):
        gray = np.array([[0, 126], [127, 255]], dtype=np.uint8)

        thresholded = threshold_image(gray, threshold=127)

        np.testing.assert_array_equal(
            thresholded,
            np.array([[0, 0], [0, 255]], dtype=np.uint8),
        )

    def test_equalize_histogram_keeps_shape_and_expands_low_contrast_values(self):
        gray = np.array(
            [
                [100, 100, 110, 110],
                [100, 100, 110, 110],
                [120, 120, 130, 130],
                [120, 120, 130, 130],
            ],
            dtype=np.uint8,
        )

        equalized = equalize_histogram(gray)

        self.assertEqual(equalized.shape, gray.shape)
        self.assertEqual(equalized.dtype, np.uint8)
        self.assertGreaterEqual(int(equalized.min()), 0)
        self.assertEqual(int(equalized.max()), 255)

    def test_apply_pseudocolor_returns_three_channel_color_image(self):
        gray = np.array([[0, 128], [200, 255]], dtype=np.uint8)

        color = apply_pseudocolor(gray)

        self.assertEqual(color.shape, (2, 2, 3))
        self.assertEqual(color.dtype, np.uint8)
        self.assertFalse(np.array_equal(color[:, :, 0], color[:, :, 1]))


if __name__ == "__main__":
    unittest.main()
