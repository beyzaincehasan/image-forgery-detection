import cv2
import os
import matplotlib.pyplot as plt

OUTPUT_DIR = "/home/beyza/image-forgery-detection/outputs/feature_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_PATH = "/home/beyza/image-forgery-detection/dataset/processed/test/tampered"

def get_sample_image(folder):
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            return os.path.join(folder, file)
    return None

def save_keypoints(image, keypoints, title, output_name):
    result = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    output_path = os.path.join(OUTPUT_DIR, output_name)
    cv2.imwrite(output_path, result)

    print(f"{title} keypoint sayısı: {len(keypoints)}")
    print(f"Kaydedildi: {output_path}")

def run_algorithm(name, detector, image_gray, image_color):
    keypoints, descriptors = detector.detectAndCompute(image_gray, None)
    save_keypoints(
        image_color,
        keypoints,
        name,
        f"{name.lower()}_keypoints.jpg"
    )

def main():
    image_path = get_sample_image(IMAGE_PATH)

    if image_path is None:
        print("Görüntü bulunamadı.")
        return

    print("Seçilen görüntü:", image_path)

    image_color = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    # SIFT
    sift = cv2.SIFT_create()
    run_algorithm("SIFT", sift, image_gray, image_color)

    # SURF
    try:
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        run_algorithm("SURF", surf, image_gray, image_color)
    except Exception as e:
        print("SURF çalıştırılamadı.")
        print("Sebep:", e)
        print("Not: SURF bazı OpenCV sürümlerinde patent/nonfree modül nedeniyle devre dışı olabilir.")

    # AKAZE
    akaze = cv2.AKAZE_create()
    run_algorithm("AKAZE", akaze, image_gray, image_color)

    # ORB
    orb = cv2.ORB_create(nfeatures=1000)
    run_algorithm("ORB", orb, image_gray, image_color)

if __name__ == "__main__":
    main()
