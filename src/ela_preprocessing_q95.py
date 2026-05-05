from PIL import Image, ImageChops, ImageEnhance
import os

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def convert_to_ela(image_path, quality=95):
    temp_path = "/tmp/temp_ela.jpg"

    image = Image.open(image_path).convert("RGB")
    image.save(temp_path, "JPEG", quality=quality)

    compressed_image = Image.open(temp_path)

    ela_image = ImageChops.difference(image, compressed_image)
    extrema = ela_image.getextrema()

    max_diff = max([channel[1] for channel in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1

    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    return ela_image

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    count = 0

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(VALID_EXTENSIONS):
            input_path = os.path.join(input_folder, file_name)

            output_name = os.path.splitext(file_name)[0] + ".jpg"
            output_path = os.path.join(output_folder, output_name)

            try:
                ela_image = convert_to_ela(input_path)
                ela_image.save(output_path, "JPEG")
                count += 1
            except Exception as e:
                print(f"Hata: {input_path} işlenemedi -> {e}")

    print(f"{output_folder} klasörüne {count} ELA görüntüsü kaydedildi.")

def main():
    input_base = "/home/beyza/image-forgery-detection/dataset/processed"
    output_base = "/home/beyza/image-forgery-detection/dataset/processed_ela_95"

    for split in ["train", "val", "test"]:
        for cls in ["authentic", "tampered"]:
            input_folder = os.path.join(input_base, split, cls)
            output_folder = os.path.join(output_base, split, cls)

            process_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()
