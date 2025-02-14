import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_energy(image):
    """
    Calculate the energy map of the image using the gradient magnitude.

    The energy map represents the importance of each pixel in the image.
    Pixels with high energy are less likely to be removed during seam carving.
    The Sobel operator is used to compute the gradient magnitude in both the
    x and y directions.

    Args:
        image (numpy.ndarray): The input image in BGR format.

    Returns:
        numpy.ndarray: The energy map of the image.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Compute gradients along the x and y axis
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Calculate the energy map as the sum of absolute gradients
    energy = np.abs(grad_x) + np.abs(grad_y)
    return energy


def carve_horizontally_by_pixel_removal(image, pixels=50):
    """
    Reduce the width of the image by removing pixels with least energy in each row.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of pixels to remove.

    Returns:
        numpy.ndarray: The carved image.
    """
    current_image = image.copy()
    for _ in range(pixels):
        energy = calculate_energy(current_image)
        rows, cols = energy.shape
        # For each row, remove the pixel with least energy
        new_image = np.zeros((rows, cols - 1, 3), dtype=current_image.dtype)
        for i in range(rows):
            # Find the index of the pixel with least energy in the row
            min_energy_index = np.argmin(energy[i, :])
            # Remove that pixel from the row
            new_image[i, :, :] = np.delete(current_image[i, :, :], min_energy_index, axis=0)
        current_image = new_image
    return current_image


def enlarge_horizontally_by_pixel_duplication(image, pixels=50):
    """
    Increase the width of the image by duplicating pixels with maximum energy in each row.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of pixels to add.

    Returns:
        numpy.ndarray: The enlarged image.
    """
    current_image = image.copy()
    for _ in range(pixels):
        energy = calculate_energy(current_image)
        rows, cols = energy.shape
        # For each row, duplicate the pixel with maximum energy
        new_image = np.zeros((rows, cols + 1, 3), dtype=current_image.dtype)
        for i in range(rows):
            # Find the index of the pixel with maximum energy in the row
            max_energy_index = np.argmax(energy[i, :])
            # Duplicate that pixel in the row
            # Copy pixels up to the max_energy_index
            new_image[i, :max_energy_index + 1, :] = current_image[i, :max_energy_index + 1, :]
            # Insert the duplicate pixel
            new_image[i, max_energy_index + 1, :] = current_image[i, max_energy_index, :]
            # Copy the remaining pixels
            new_image[i, max_energy_index + 2:, :] = current_image[i, max_energy_index + 1:, :]
        current_image = new_image
    return current_image


def carve_vertically_by_pixel_removal(image, pixels=50):
    """
    Reduce the height of the image by removing pixels with least energy in each column.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of pixels to remove.

    Returns:
        numpy.ndarray: The carved image.
    """
    current_image = image.copy()
    for _ in range(pixels):
        energy = calculate_energy(current_image)
        rows, cols = energy.shape
        # For each column, remove the pixel with least energy
        new_image = np.zeros((rows - 1, cols, 3), dtype=current_image.dtype)
        for j in range(cols):
            # Find the index of the pixel with least energy in the column
            min_energy_index = np.argmin(energy[:, j])
            # Remove that pixel from the column
            new_image[:, j, :] = np.delete(current_image[:, j, :], min_energy_index, axis=0)
        current_image = new_image
    return current_image


def enlarge_vertically_by_pixel_duplication(image, pixels=50):
    """
    Increase the height of the image by duplicating pixels with maximum energy in each column.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of pixels to add.

    Returns:
        numpy.ndarray: The enlarged image.
    """
    current_image = image.copy()
    for _ in range(pixels):
        energy = calculate_energy(current_image)
        rows, cols = energy.shape
        # For each column, duplicate the pixel with maximum energy
        new_image = np.zeros((rows + 1, cols, 3), dtype=current_image.dtype)
        for j in range(cols):
            # Find the index of the pixel with maximum energy in the column
            max_energy_index = np.argmax(energy[:, j])
            # Duplicate that pixel in the column
            # Copy pixels up to the max_energy_index
            new_image[:max_energy_index + 1, j, :] = current_image[:max_energy_index + 1, j, :]
            # Insert the duplicate pixel
            new_image[max_energy_index + 1, j, :] = current_image[max_energy_index, j, :]
            # Copy the remaining pixels
            new_image[max_energy_index + 2:, j, :] = current_image[max_energy_index + 1:, j, :]
        current_image = new_image
    return current_image


def display_with_ticks(image, title):
    """
    Display the image with x and y ticks at intervals of 50 pixels.

    Args:
        image (numpy.ndarray): The image to display.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(8, 6))
    # Convert BGR image to RGB for correct color display
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    # Set x and y ticks
    plt.xticks(range(0, image.shape[1], 50))
    plt.yticks(range(0, image.shape[0], 50))
    # Add a grid for better visualization
    plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.show()


def resize_image(image, target_width, target_height):
    """
    Resize the image to the target width and height by removing or duplicating pixels
    with least or maximum energy accordingly.

    Args:
        image (numpy.ndarray): The input image.
        target_width (int): The desired width of the output image.
        target_height (int): The desired height of the output image.

    Returns:
        numpy.ndarray: The resized image.
    """
    current_image = image.copy()
    current_height, current_width = current_image.shape[:2]

    # Calculate the difference in dimensions
    delta_width = target_width - current_width
    delta_height = target_height - current_height

    # Resize width
    if delta_width < 0:
        # Need to remove pixels
        current_image = carve_horizontally_by_pixel_removal(current_image, abs(delta_width))
    elif delta_width > 0:
        # Need to duplicate pixels
        current_image = enlarge_horizontally_by_pixel_duplication(current_image, delta_width)

    # Resize height
    if delta_height < 0:
        # Need to remove pixels
        current_image = carve_vertically_by_pixel_removal(current_image, abs(delta_height))
    elif delta_height > 0:
        # Need to duplicate pixels
        current_image = enlarge_vertically_by_pixel_duplication(current_image, delta_height)

    return current_image


def main():
    """
    Main function to execute the pixel-based carving and enlargement operations.
    """
    # Load input image (ensure the image path is correct)
    image = cv2.imread('image.jpg')

    # Display original image with energy map
    energy_map = calculate_energy(image)
    plt.figure(figsize=(8, 6))
    plt.imshow(energy_map, cmap='gray')
    plt.title("Energy Map")
    plt.colorbar()
    plt.show()

    display_with_ticks(image, "Original Image")

    # Carve and enlarge image horizontally by pixel removal/duplication
    carved_horizontal = carve_horizontally_by_pixel_removal(image, pixels=50)
    display_with_ticks(carved_horizontal, "Image Carved Horizontally by 50 Pixels (Pixel Removal)")

    enlarged_horizontal = enlarge_horizontally_by_pixel_duplication(image, pixels=50)
    display_with_ticks(enlarged_horizontal, "Image Enlarged Horizontally by 50 Pixels (Pixel Duplication)")

    # Carve and enlarge image vertically by pixel removal/duplication
    carved_vertical = carve_vertically_by_pixel_removal(image, pixels=50)
    display_with_ticks(carved_vertical, "Image Carved Vertically by 50 Pixels (Pixel Removal)")

    enlarged_vertical = enlarge_vertically_by_pixel_duplication(image, pixels=50)
    display_with_ticks(enlarged_vertical, "Image Enlarged Vertically by 50 Pixels (Pixel Duplication)")

    # Desired dimensions (replace with your desired width and height)
    target_width = 501
    target_height = 351

    # Resize image to desired dimensions
    resized_image = resize_image(image, target_width, target_height)
    display_with_ticks(resized_image, f"Resized Image to {target_width}x{target_height}")


if __name__ == "__main__":
    main()
