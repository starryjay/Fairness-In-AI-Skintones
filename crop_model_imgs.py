import cv2
import os

def read_image(image_path: str) -> cv2.Mat:
    """
    Reads an image from the specified path.
    Returns:
    - The image as a numpy array.
    """

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be read.")
    # Resize to 800x1200
    image = cv2.resize(image, (800, 1200))
    return image

def face_detector(images: list[str, cv2.Mat]) -> list[str, cv2.Mat]:
    """
    Detects faces in the image using OpenCV's Haar Cascade classifier.
    Returns:
    - A list of rectangles representing the detected faces.
    """
    faces = []
    for path, image in images:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray_image, scaleFactor=1.5, minNeighbors=3, minSize=(80, 80))
        faces.append((path, face))
    return faces

def crop_face(images: list[str, cv2.Mat], faces: list) -> list[str, cv2.Mat]:
    """
    Uses the coordinates of the detected faces to crop the images.
    Returns:
    - A list of tuples containing the image path and the cropped face image.
    """
    cropped_faces = []
    img_height, img_width = images[0][1].shape[:2]
    for tup, face_array in zip(images, faces):
        path = tup[0]
        image = tup[1]
        if len(face_array) > 0:  # Check if any face is detected
            for face_rect in face_array:
                x, y, w, h = face_rect
                x_end = min(x + w, img_width)
                y_end = min(y + h, img_height)
                x_start = max(x, 0)
                y_start = max(y, 0)
                cropped_face = image[y_start:y_end, x_start:x_end]
                cropped_faces.append((path, cropped_face))
    return cropped_faces

def save_cropped_faces(zipped_crops: list[cv2.Mat], output_dir: str) -> None:
    """
    Saves the cropped face images to the specified output directory.
    """
    for img_path, cropped_face in zipped_crops:
        filename = os.path.basename(img_path)
        new_filename = os.path.join(output_dir, f"cropped_{filename}")
        cv2.imwrite(new_filename, cropped_face)

def main() -> None:
    """
    Main function to read images, detect faces, crop them, and save the cropped images.
    """
    output_dir = "./cropped_faces"
    os.makedirs(output_dir, exist_ok=True)
    input_dir = './Fashion_Imgs'
    images = []
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(input_dir, filename)
            image = read_image(image_path)
            images.append((image_path, image))
    zipped_paths_faces = face_detector(images)
    faces = [item[1] for item in zipped_paths_faces]
    zipped_crops = crop_face(images, faces)
    save_cropped_faces(zipped_crops, output_dir)

if __name__ == "__main__":
    main()