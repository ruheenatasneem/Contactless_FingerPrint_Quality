"""
=========================================================
Contactless Fingerprint Quality Assessment
quality_assessment.py

Author : Ruheena Tasneem
Description:
Core Quality Assessment Module

This module performs

1. Blur Detection
2. Brightness Detection
3. Glare Detection
4. ROI Completeness
5. Ridge Clarity
6. Composite Score
7. Quality Gate

=========================================================
"""

import cv2
import numpy as np


# =========================================================
# Blur Detection
# =========================================================

def check_blur(image_bgr, threshold=10.0):
    """
    Detects whether image is blurry using
    Laplacian Variance.
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    result = {
        "blur_score": round(float(blur_score), 2),
        "is_blurry": blur_score < threshold
    }

    return result


# =========================================================
# Brightness Detection
# =========================================================

def check_brightness(
        image_bgr,
        min_thresh=50,
        max_thresh=210
):

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(gray)

    result = {
        "brightness": round(float(brightness), 2),
        "too_dark": brightness < min_thresh,
        "too_bright": brightness > max_thresh
    }

    return result


# =========================================================
# Glare Detection
# =========================================================

def check_glare(
        image_bgr,
        max_glare_ratio=0.05
):

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    glare_pixels = np.sum(gray > 240)

    total_pixels = gray.size

    glare_fraction = glare_pixels / total_pixels

    result = {
        "has_glare": glare_fraction > max_glare_ratio,
        "glare_fraction": round(float(glare_fraction), 4)
    }

    return result


# =========================================================
# ROI Completeness
# =========================================================

def check_roi_completeness(
        image_bgr,
        min_roi_ratio=0.15
):

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    foreground_pixels = np.sum(thresh > 0)

    total_pixels = gray.size

    roi_fraction = foreground_pixels / total_pixels

    result = {
        "roi_fraction": round(float(roi_fraction), 4),
        "roi_complete": roi_fraction >= min_roi_ratio
    }

    return result


# =========================================================
# Ridge Clarity
# =========================================================

def check_ridge_clarity(
        image_bgr,
        threshold=15.0
):

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    kernel = cv2.getGaborKernel(
        (21, 21),
        sigma=5,
        theta=np.pi / 4,
        lambd=10,
        gamma=0.5,
        psi=0
    )

    filtered = cv2.filter2D(
        gray,
        cv2.CV_64F,
        kernel
    )

    ridge_score = np.var(filtered) / 100

    result = {
        "ridge_score": round(float(ridge_score), 2),
        "ridges_clear": ridge_score >= threshold
    }

    return result


# =========================================================
# Normalize Blur
# =========================================================

def normalize_blur(score):

    return min(1.0, score / 50.0)


# =========================================================
# Normalize Brightness
# =========================================================

def normalize_brightness(score):

    value = 1.0 - abs(score - 128) / 128

    return max(0.0, value)


# =========================================================
# Normalize Glare
# =========================================================

def normalize_glare(glare):

    value = 1.0 - glare / 0.05

    return max(0.0, value)


# =========================================================
# Normalize ROI
# =========================================================

def normalize_roi(roi):

    return min(1.0, roi / 0.35)


# =========================================================
# Normalize Ridge
# =========================================================

def normalize_ridge(score):

    return min(1.0, score / 30.0)


# =========================================================
# Composite Score
# =========================================================

def calculate_composite_score(
        blur_result,
        bright_result,
        glare_result,
        roi_result,
        ridge_result
):

    n_blur = normalize_blur(
        blur_result["blur_score"]
    )

    n_brightness = normalize_brightness(
        bright_result["brightness"]
    )

    n_glare = normalize_glare(
        glare_result["glare_fraction"]
    )

    n_roi = normalize_roi(
        roi_result["roi_fraction"]
    )

    n_ridge = normalize_ridge(
        ridge_result["ridge_score"]
    )

    score = (
        0.25 * n_blur +
        0.15 * n_brightness +
        0.15 * n_glare +
        0.20 * n_roi +
        0.25 * n_ridge
    ) * 100

    return round(score, 2)

# =========================================================
# Guidance Message
# =========================================================

def get_guidance(
        blur_result,
        bright_result,
        glare_result,
        roi_result,
        ridge_result
):
    """
    Returns guidance message based on failed quality check.
    """

    if blur_result["is_blurry"]:
        return "Too blurry - Hold your phone steady and refocus."

    if bright_result["too_dark"]:
        return "Too dark - Move to a brighter area or turn on flash."

    if bright_result["too_bright"]:
        return "Too bright - Reduce direct lighting."

    if glare_result["has_glare"]:
        return "Glare detected - Tilt your finger slightly."

    if not roi_result["roi_complete"]:
        return "Finger not fully visible - Move finger closer."

    if not ridge_result["ridges_clear"]:
        return "Fingerprint ridges unclear - Clean the camera lens."

    return "Good capture - Ready for processing."


# =========================================================
# Master Quality Gate
# =========================================================

def quality_gate(image_path_or_array):
    """
    Main function for evaluating fingerprint quality.
    """

    # Read Image
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
    else:
        img = image_path_or_array

    if img is None:
        raise ValueError("Image could not be loaded.")

    # Run all quality checks
    blur_result = check_blur(img)

    bright_result = check_brightness(img)

    glare_result = check_glare(img)

    roi_result = check_roi_completeness(img)

    ridge_result = check_ridge_clarity(img)

    # Calculate Composite Score
    composite_score = calculate_composite_score(
        blur_result,
        bright_result,
        glare_result,
        roi_result,
        ridge_result
    )

    # Hard Failure Conditions
    hard_failure = (
        blur_result["is_blurry"] or
        bright_result["too_dark"] or
        bright_result["too_bright"] or
        glare_result["has_glare"] or
        not roi_result["roi_complete"] or
        not ridge_result["ridges_clear"]
    )

    passed = composite_score >= 60 and not hard_failure

    guidance = get_guidance(
        blur_result,
        bright_result,
        glare_result,
        roi_result,
        ridge_result
    )

    result = {
        "passed": passed,
        "composite_score": composite_score,
        "blur": blur_result,
        "brightness": bright_result,
        "glare": glare_result,
        "roi": roi_result,
        "ridge": ridge_result,
        "guidance": guidance
    }

    return result


# =========================================================
# Print Results
# =========================================================

def print_result(result):
    """
    Prints results in a readable format.
    """

    print("\n")
    print("=" * 60)
    print("CONTACTLESS FINGERPRINT QUALITY REPORT")
    print("=" * 60)

    print(f"Composite Score : {result['composite_score']} / 100")

    if result["passed"]:
        print("Overall Result  : PASS")
    else:
        print("Overall Result  : FAIL")

    print("-" * 60)

    print("Blur Score      :", result["blur"]["blur_score"])
    print("Brightness      :", result["brightness"]["brightness"])
    print("Glare Fraction  :", result["glare"]["glare_fraction"])
    print("ROI Fraction    :", result["roi"]["roi_fraction"])
    print("Ridge Score     :", result["ridge"]["ridge_score"])

    print("-" * 60)

    print("Guidance:")
    print(result["guidance"])

    print("=" * 60)


# =========================================================
# Main Function
# =========================================================

if __name__ == "__main__":

    image_path = input("Enter image path : ")

    try:

        result = quality_gate(image_path)

        print_result(result)

    except Exception as e:

        print("Error :", e)