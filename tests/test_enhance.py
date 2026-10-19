"""Automatic clean-up: find the film, crop borders, denoise -- and decline safely."""

import cv2
import numpy as np

from radiscan.data.enhance import body_box, denoise, enhance, find_film


def _fake_chest(h=600, w=500, seed=0) -> np.ndarray:
    """Greyscale stand-in for a digital radiograph: dark lungs, bright mediastinum."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 170, np.uint8)
    cv2.ellipse(img, (w // 3, h // 2), (w // 7, h // 3), 0, 0, 360, 60, -1)
    cv2.ellipse(img, (2 * w // 3, h // 2), (w // 7, h // 3), 0, 0, 360, 60, -1)
    img = cv2.add(img, rng.integers(0, 40, img.shape, dtype=np.uint8))
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def _photo_of_film() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Colour photo: a dark film with anatomy, held against a bright warm room."""
    photo = np.zeros((900, 700, 3), np.uint8)
    photo[:] = (205, 190, 165)  # beige wall: colour, as a phone photo has
    x0, y0, x1, y1 = 120, 150, 580, 800
    film = cv2.cvtColor(_fake_chest(y1 - y0, x1 - x0), cv2.COLOR_RGB2GRAY) // 2  # film is dark
    photo[y0:y1, x0:x1] = cv2.cvtColor(film, cv2.COLOR_GRAY2RGB)
    return photo, (x0, y0, x1, y1)


def test_photo_film_is_located_and_background_removed():
    photo, (x0, y0, x1, y1) = _photo_of_film()
    out = enhance(photo, reduce_noise=False, crop=False)
    assert out.film_detected
    assert any("film" in s for s in out.steps)
    h, w = out.image.shape[:2]
    # Within 5% of the true film size: background gone, film not clipped.
    assert abs(h - (y1 - y0)) < 0.05 * (y1 - y0)
    assert abs(w - (x1 - x0)) < 0.05 * (x1 - x0)


def test_digital_radiograph_is_never_treated_as_a_photo():
    # The failure this guards against: a dark lung framed by bright ribs looks
    # exactly like a dark film in a bright room, and the "film" found is one lung.
    img = _fake_chest()
    out = enhance(img, reduce_noise=False)
    assert not out.film_detected
    assert out.image.shape[:2] == img.shape[:2]


def test_find_film_declines_without_a_background():
    assert find_film(cv2.cvtColor(_fake_chest(), cv2.COLOR_RGB2GRAY)) is None


def test_padding_is_cropped_but_anatomy_kept():
    chest = _fake_chest(400, 300)
    padded = np.zeros((600, 600, 3), np.uint8)
    padded[100:500, 150:450] = chest
    out = enhance(padded, reduce_noise=False)
    assert out.crop_box is not None
    h, w = out.image.shape[:2]
    assert 400 <= h < 600 and 300 <= w < 600  # padding removed, no anatomy lost


def test_body_box_declines_on_a_blank_image():
    assert body_box(np.zeros((200, 200), np.uint8)) is None


def test_denoise_reduces_noise_and_keeps_shape():
    rng = np.random.default_rng(1)
    flat = np.clip(128 + rng.normal(0, 12, (256, 256)), 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(flat, cv2.COLOR_GRAY2RGB)
    out = denoise(rgb)
    assert out.shape == rgb.shape and out.dtype == np.uint8
    assert out[..., 0].std() < flat.std() * 0.7


def test_noise_estimate_tracks_true_noise():
    from radiscan.data.enhance import estimate_noise

    rng = np.random.default_rng(2)
    for sigma in (3, 8, 15):
        img = np.clip(128 + rng.normal(0, sigma, (256, 256)), 0, 255).astype(np.uint8)
        assert abs(estimate_noise(img) - sigma) < 0.25 * sigma + 1


def test_enhance_never_raises_on_odd_input():
    for img in (np.zeros((50, 50, 3), np.uint8), np.full((80, 30), 255, np.uint8)):
        out = enhance(img)
        assert out.image.ndim == 3 and out.steps
