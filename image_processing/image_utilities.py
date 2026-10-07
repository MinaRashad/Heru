import cv2
import numpy as np

def face_crop_quality(crop: np.ndarray) -> float:
    """Return a quality score where higher is better."""
    if crop is None or crop.size == 0:
        return -1.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 1. Sharpness.
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 2. Resolution / size.
    height, width = gray.shape
    area = width * height

    # 3. Exposure quality.
    mean_brightness = gray.mean()

    # Penalize very dark / very bright crops.
    exposure_score = 1.0 - min(
        abs(mean_brightness - 128.0) / 128.0,
        1.0,
    )

    # Diminishing returns for very large images.
    size_score = np.sqrt(area)

    return sharpness * size_score * exposure_score