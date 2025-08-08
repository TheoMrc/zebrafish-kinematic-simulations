import numpy as np

from video_preprocessing.experiment import Frame


def test_rgb2gray_weights():
    # Single pixel RGB to grayscale using known weights
    rgb = np.array([[[100.0, 150.0, 200.0]]], dtype=float)
    gray = Frame.rgb2gray(rgb)
    expected = 100.0 * 0.2989 + 150.0 * 0.5870 + 200.0 * 0.1140
    assert np.allclose(gray[0, 0], expected, atol=1e-6)


def test_custom_hist_adjust_uint8_bounds():
    frame = np.array([[0.0, 0.5], [0.25, 1.0]], dtype=float)
    out = Frame.custom_hist_adjust(frame, gamma=1.0)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_create_fish_centered_frame_shape_and_center():
    # Two points around origin; with half_size=10 and shift=(0,0)
    fish_zone = np.array([[0, 0], [0, 1]], dtype=int)
    centered_frame, mass_center = Frame.create_fish_centered_frame(
        fish_zone, half_size=10, shift=(0, 0)
    )
    assert centered_frame.shape == (20, 20)
    # Mass center should be between the two points on y-axis
    assert 0.0 <= mass_center[0] <= 0.5
    assert 0.0 <= mass_center[1] <= 1.0
    # There should be some ones in the centered frame
    assert int(centered_frame.sum()) >= 2


def test_calculate_rotation_angle_vertical():
    # Points aligned on positive Y axis relative to mass center -> ~90 degrees
    fish_zone = {(0, 5), (0, 10), (0, 15)}
    mass_center = (0.0, 0.0)
    angle = Frame.calculate_rotation_angle(fish_zone, mass_center)
    assert 80.0 <= angle <= 100.0


def test_image_standardisation_no_nan():
    arr = np.array([[1.0, 2.0, 3.0]], dtype=float)
    std = Frame.image_standardisation(arr)
    # Should be numeric and same shape
    assert std.shape == arr.shape
    assert np.isfinite(std).all()


