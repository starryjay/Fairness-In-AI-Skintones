import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

def read_image(image_path: str) -> cv2.Mat:
    """
    Reads an image from the specified path.
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
    Returns a list of rectangles representing the detected faces.
    """
    faces = []
    for path, image in images:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #gray_image = cv2.equalizeHist(gray_image)
        #face = face_cascade.detectMultiScale(gray_image, scaleFactor=1.4, minNeighbors=3, minSize=(100, 100))  # this one was good but not perfect
        face = face_cascade.detectMultiScale(gray_image, scaleFactor=1.5, minNeighbors=3, minSize=(80, 80))
        faces.append((path, face))
    return faces

# def crop_face(images: list[str, cv2.Mat], faces: list) -> list[cv2.Mat]:
#     """
#     Crops the detected faces from the images.
#     """
#     cropped_faces = []
#     for tup, face_array in zip(images, faces):
#         path = tup[0]
#         image = tup[1]
#         print('\nimage:\n', image)
#         if len(face_array) > 0:  # Check if any face is detected
#             for face_rect in face_array:
#                 x, y, w, h = face_rect
#                 cropped_face = image[y:y+h, x:x+w]
#                 cropped_faces.append((path, cropped_face))
#     return cropped_faces

def crop_face(images: list[str, cv2.Mat], faces: list):
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

# def show_cropped_faces(cropped_faces: list[str, cv2.Mat]) -> None:
#     """
#     Displays the cropped faces using matplotlib.
#     """
#     # show in a table instead of a row
#     num_faces = len(cropped_faces)
#     num_cols = 5
#     num_rows = (num_faces + num_cols - 1) // num_cols  # Calculate number of rows needed
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
#     for i, (img_path, cropped_face) in enumerate(cropped_faces):
#         ax = axes[i // num_cols, i % num_cols]
#         ax.imshow(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
#         #ax.set_title(os.path.basename(img_path))
#         ax.axis('off')
#     # Hide any unused subplots
#     plt.tight_layout()
#     plt.suptitle("Cropped Faces", fontsize=16)
#     plt.show()

def save_cropped_faces(zipped_crops: list[cv2.Mat], output_dir: str) -> None:
    for img_path, cropped_face in zipped_crops:
        # Get the filename without the directory
        filename = os.path.basename(img_path)
        # Create a new filename for the cropped face
        new_filename = os.path.join(output_dir, f"cropped_{filename}")
        # Save the cropped face
        cv2.imwrite(new_filename, cropped_face)

def main():
    #output_dir = "cropped_faces_one"
    output_dir = "cropped_faces"
    os.makedirs(output_dir, exist_ok=True)
    #os.makedirs(output_dir_2, exist_ok=True)
    #input_dir = "./light_skinned_models"
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
    #zipped_crops_two = crop_face_two(images, faces)
    # show_cropped_faces(zipped_crops)
    #save_cropped_faces(zipped_crops, output_dir)
    save_cropped_faces(zipped_crops, output_dir)

if __name__ == "__main__":
    main()