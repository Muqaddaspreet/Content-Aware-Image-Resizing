import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_energy(image):
    """
    Calculate the energy map of the image using the gradient magnitude.
    """
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
    energy = np.abs(grad_x) + np.abs(grad_y)
    return energy

def carve_horizontally(image, pixels=50):
    """
    Reduce the width of the image by globally removing pixels with the lowest energy.
    The image is reshaped back into a rectangle after removal.
    """
    h, w, c = image.shape
    total_pixels_to_remove = pixels * h  # Total number of pixels to remove

    # Flatten the image and energy map
    energy = calculate_energy(image)
    flat_image = image.reshape(-1, c)
    flat_energy = energy.flatten()

    # Get indices of pixels to remove
    remove_indices = np.argsort(flat_energy)[:total_pixels_to_remove]

    # Create a mask to keep pixels not in remove_indices
    mask = np.ones(flat_image.shape[0], dtype=bool)
    mask[remove_indices] = False

    # Remove pixels
    reduced_image = flat_image[mask]

    # Calculate new width after pixel removal
    new_width = w - pixels

    # Ensure the number of pixels matches the new dimensions
    if reduced_image.shape[0] < h * new_width:
        # Pad the array if it's too small
        pad_size = h * new_width - reduced_image.shape[0]
        padding = np.zeros((pad_size, c), dtype=reduced_image.dtype)
        reduced_image = np.vstack((reduced_image, padding))
    elif reduced_image.shape[0] > h * new_width:
        # Trim the array if it's too large
        reduced_image = reduced_image[:h * new_width]

    # Reshape back to image
    carved_image = reduced_image.reshape(h, new_width, c)
    return carved_image

def enlarge_horizontally(image, pixels=50):
    """
    Increase the width of the image by globally duplicating pixels with the highest energy.
    The image is reshaped back into a rectangle after duplication.
    """
    h, w, c = image.shape
    total_pixels_to_duplicate = pixels * h  # Total number of pixels to duplicate

    # Flatten the image and energy map
    energy = calculate_energy(image)
    flat_image = image.reshape(-1, c)
    flat_energy = energy.flatten()

    # Get indices of pixels to duplicate
    duplicate_indices = np.argsort(flat_energy)[::-1][:total_pixels_to_duplicate]

    # Duplicate pixels
    duplicated_pixels = flat_image[duplicate_indices]

    # Append duplicated pixels to the flat image
    enlarged_flat_image = np.concatenate((flat_image, duplicated_pixels), axis=0)

    # Calculate new width after pixel duplication
    new_width = w + pixels

    # Ensure the number of pixels matches the new dimensions
    if enlarged_flat_image.shape[0] < h * new_width:
        # Pad the array if it's too small
        pad_size = h * new_width - enlarged_flat_image.shape[0]
        padding = np.zeros((pad_size, c), dtype=enlarged_flat_image.dtype)
        enlarged_flat_image = np.vstack((enlarged_flat_image, padding))
    elif enlarged_flat_image.shape[0] > h * new_width:
        # Trim the array if it's too large
        enlarged_flat_image = enlarged_flat_image[:h * new_width]

    # Reshape back to image
    enlarged_image = enlarged_flat_image.reshape(h, new_width, c)
    return enlarged_image

def carve_vertically(image, pixels=50):
    """
    Reduce the height of the image by globally removing pixels with the lowest energy.
    The image is reshaped back into a rectangle after removal.
    """
    # Transpose the image to work on height as width
    transposed_image = image.transpose(1, 0, 2)
    carved_transposed = carve_horizontally(transposed_image, pixels)
    carved_image = carved_transposed.transpose(1, 0, 2)
    return carved_image

def enlarge_vertically(image, pixels=50):
    """
    Increase the height of the image by globally duplicating pixels with the highest energy.
    The image is reshaped back into a rectangle after duplication.
    """
    # Transpose the image to work on height as width
    transposed_image = image.transpose(1, 0, 2)
    enlarged_transposed = enlarge_horizontally(transposed_image, pixels)
    enlarged_image = enlarged_transposed.transpose(1, 0, 2)
    return enlarged_image

def resize_image(image, target_width, target_height):
    """
    Resize the image to the target width and height using global pixel removal and duplication.
    """
    current_image = image.copy()
    current_height, current_width = current_image.shape[:2]
    delta_width = target_width - current_width
    delta_height = target_height - current_height

    # Resize width
    if delta_width < 0:
        current_image = carve_horizontally(current_image, abs(delta_width))
    elif delta_width > 0:
        current_image = enlarge_horizontally(current_image, delta_width)

    # Resize height
    if delta_height < 0:
        current_image = carve_vertically(current_image, abs(delta_height))
    elif delta_height > 0:
        current_image = enlarge_vertically(current_image, delta_height)

    return current_image

def display_with_ticks(image, title):
    """
    Display the image with x and y ticks at intervals of 50 pixels.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.xticks(range(0, image.shape[1], 50))
    plt.yticks(range(0, image.shape[0], 50))
    plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.show()

def main():
    """
    Main function to execute the global pixel removal and duplication operations.
    """
    # Load input image
    image = cv2.imread('image.jpg')
    if image is None:
        print("Error: Image not found or unable to read.")
        return

    # Display original image with energy map
    energy_map = calculate_energy(image)
    plt.figure(figsize=(8, 6))
    plt.imshow(energy_map, cmap='gray')
    plt.title("Energy Map")
    plt.colorbar()
    plt.show()

    display_with_ticks(image, "Original Image")

    # Carve and enlarge image horizontally
    carved_horizontal = carve_horizontally(image, pixels=50)
    display_with_ticks(carved_horizontal, "Image Carved Horizontally by 50 Pixels")

    enlarged_horizontal = enlarge_horizontally(image, pixels=50)
    display_with_ticks(enlarged_horizontal, "Image Enlarged Horizontally by 50 Pixels")

    # Carve and enlarge image vertically
    carved_vertical = carve_vertically(image, pixels=50)
    display_with_ticks(carved_vertical, "Image Carved Vertically by 50 Pixels")

    enlarged_vertical = enlarge_vertically(image, pixels=50)
    display_with_ticks(enlarged_vertical, "Image Enlarged Vertically by 50 Pixels")

    # Desired dimensions
    target_width = image.shape[1] - 50  # Adjust as needed
    target_height = image.shape[0] - 50  # Adjust as needed

    # Resize image to desired dimensions
    resized_image = resize_image(image, target_width, target_height)
    display_with_ticks(resized_image, f"Resized Image to {target_width}x{target_height}")

if __name__ == "__main__":
    main()
