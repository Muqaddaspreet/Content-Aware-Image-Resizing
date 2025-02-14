import cv2
import numpy as np
import matplotlib.pyplot as plt


def carve_horizontally(image, pixels=50):
    """
    Reduce the width of the image by cropping it.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of pixels to remove from the width.

    Returns:
        numpy.ndarray: The cropped image.
    """
    # Ensure we don't remove more pixels than the image width
    pixels = min(pixels, image.shape[1])
    # Crop equally from both sides
    left_crop = pixels // 2
    right_crop = pixels - left_crop
    cropped_image = image[:, left_crop:image.shape[1]-right_crop]
    return cropped_image


def enlarge_horizontally(image, pixels=50):
    """
    Increase the width of the image by upscaling it.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of pixels to add to the width.

    Returns:
        numpy.ndarray: The enlarged image.
    """
    # Calculate new dimensions
    new_width = image.shape[1] + pixels
    new_size = (new_width, image.shape[0])  # (width, height)
    # Resize image using linear interpolation
    enlarged_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return enlarged_image


def carve_vertically(image, pixels=50):
    """
    Reduce the height of the image by cropping it.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of pixels to remove from the height.

    Returns:
        numpy.ndarray: The cropped image.
    """
    # Ensure we don't remove more pixels than the image height
    pixels = min(pixels, image.shape[0])
    # Crop equally from top and bottom
    top_crop = pixels // 2
    bottom_crop = pixels - top_crop
    cropped_image = image[top_crop:image.shape[0]-bottom_crop, :]
    return cropped_image


def enlarge_vertically(image, pixels=50):
    """
    Increase the height of the image by upscaling it.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of pixels to add to the height.

    Returns:
        numpy.ndarray: The enlarged image.
    """
    # Calculate new dimensions
    new_height = image.shape[0] + pixels
    new_size = (image.shape[1], new_height)  # (width, height)
    # Resize image using linear interpolation
    enlarged_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return enlarged_image


def resize_image(image, target_width, target_height):
    """
    Resize the image to the target width and height using simple cropping and upscaling.

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

    # Adjust width
    if delta_width < 0:
        # Crop horizontally
        current_image = carve_horizontally(current_image, abs(delta_width))
    elif delta_width > 0:
        # Enlarge horizontally
        current_image = enlarge_horizontally(current_image, delta_width)

    # Adjust height
    if delta_height < 0:
        # Crop vertically
        current_image = carve_vertically(current_image, abs(delta_height))
    elif delta_height > 0:
        # Enlarge vertically
        current_image = enlarge_vertically(current_image, delta_height)

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


def main():
    """
    Main function to execute the image cropping and resizing operations.
    """
    # Load input image (ensure the image path is correct)
    image = cv2.imread('image.jpg')

    if image is None:
        print("Image not found. Please check the image path.")
        return

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

    # Desired dimensions (replace with your desired width and height)
    target_width = 501
    target_height = 351

    # Resize image to desired dimensions
    resized_image = resize_image(image, target_width, target_height)
    display_with_ticks(resized_image, f"Resized Image to {target_width}x{target_height}")


if __name__ == "__main__":
    main()
