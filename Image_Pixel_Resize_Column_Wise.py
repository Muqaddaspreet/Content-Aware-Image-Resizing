import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_energy(image):
    """
    Calculate the energy map of the image using the gradient magnitude.

    The energy map represents the importance of each pixel in the image.
    Pixels with high energy are less likely to be removed.

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


def carve_horizontally_columns(image, pixels=50):
    """
    Reduce the width of the image by removing columns with minimal energy.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of columns to remove.

    Returns:
        numpy.ndarray: The carved image.
    """
    current_image = image.copy()
    for _ in range(pixels):
        energy = calculate_energy(current_image)
        # Sum energy along columns
        column_energies = np.sum(energy, axis=0)
        # Find the column with minimal energy
        min_energy_col = np.argmin(column_energies)
        # Remove that column
        current_image = np.delete(current_image, min_energy_col, axis=1)
    return current_image


def enlarge_horizontally_columns(image, pixels=50):
    """
    Increase the width of the image by duplicating columns with maximal energy.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of columns to duplicate.

    Returns:
        numpy.ndarray: The enlarged image.
    """
    current_image = image.copy()
    energy = calculate_energy(current_image)
    # Sum energy along columns
    column_energies = np.sum(energy, axis=0)
    # Get indices of columns sorted by energy in descending order
    sorted_cols = np.argsort(-column_energies)
    # Select the top 'pixels' columns to duplicate
    cols_to_duplicate = sorted_cols[:pixels]
    # Sort the columns to duplicate to adjust for index shifting
    cols_to_duplicate.sort()
    # Duplicate each column
    for idx, col_idx in enumerate(cols_to_duplicate):
        col_idx += idx  # Adjust for previous insertions
        col_to_duplicate = current_image[:, col_idx, :].reshape(-1, 1, 3)
        current_image = np.insert(current_image, col_idx, col_to_duplicate, axis=1)
    return current_image


def carve_vertically_rows(image, pixels=50):
    """
    Reduce the height of the image by removing rows with minimal energy.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of rows to remove.

    Returns:
        numpy.ndarray: The carved image.
    """
    current_image = image.copy()
    for _ in range(pixels):
        energy = calculate_energy(current_image)
        # Sum energy along rows
        row_energies = np.sum(energy, axis=1)
        # Find the row with minimal energy
        min_energy_row = np.argmin(row_energies)
        # Remove that row
        current_image = np.delete(current_image, min_energy_row, axis=0)
    return current_image


def enlarge_vertically_rows(image, pixels=50):
    """
    Increase the height of the image by duplicating rows with maximal energy.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of rows to duplicate.

    Returns:
        numpy.ndarray: The enlarged image.
    """
    current_image = image.copy()
    energy = calculate_energy(current_image)
    # Sum energy along rows
    row_energies = np.sum(energy, axis=1)
    # Get indices of rows sorted by energy in descending order
    sorted_rows = np.argsort(-row_energies)
    # Select the top 'pixels' rows to duplicate
    rows_to_duplicate = sorted_rows[:pixels]
    # Sort the rows to duplicate to adjust for index shifting
    rows_to_duplicate.sort()
    # Duplicate each row
    for idx, row_idx in enumerate(rows_to_duplicate):
        row_idx += idx  # Adjust for previous insertions
        row_to_duplicate = current_image[row_idx, :, :].reshape(1, -1, 3)
        current_image = np.insert(current_image, row_idx, row_to_duplicate, axis=0)
    return current_image


def resize_image_columns(image, target_width, target_height):
    """
    Resize the image to the target width and height by removing or duplicating columns/rows.

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
        # Need to remove columns
        current_image = carve_horizontally_columns(current_image, abs(delta_width))
    elif delta_width > 0:
        # Need to duplicate columns
        current_image = enlarge_horizontally_columns(current_image, delta_width)

    # Resize height
    if delta_height < 0:
        # Need to remove rows
        current_image = carve_vertically_rows(current_image, abs(delta_height))
    elif delta_height > 0:
        # Need to duplicate rows
        current_image = enlarge_vertically_rows(current_image, delta_height)

    return current_image


def main():
    """
    Main function to execute the column/row carving and enlargement operations.
    """
    # Load input image (ensure the image path is correct)
    image = cv2.imread('image.jpg')

    # Check if the image was loaded successfully
    if image is None:
        print("Error: Image not found. Please check the image path.")
        return

    # Display original image with energy map
    energy_map = calculate_energy(image)
    plt.figure(figsize=(8, 6))
    plt.imshow(energy_map, cmap='gray')
    plt.title("Energy Map")
    plt.colorbar()
    plt.show()

    display_with_ticks(image, "Original Image")

    # Carve and enlarge image horizontally using columns
    carved_horizontal_columns = carve_horizontally_columns(image, pixels=50)
    display_with_ticks(carved_horizontal_columns, "Image Carved Horizontally by 50 Pixels (Columns)")

    enlarged_horizontal_columns = enlarge_horizontally_columns(image, pixels=50)
    display_with_ticks(enlarged_horizontal_columns, "Image Enlarged Horizontally by 50 Pixels (Columns)")

    # Carve and enlarge image vertically using rows
    carved_vertical_rows = carve_vertically_rows(image, pixels=50)
    display_with_ticks(carved_vertical_rows, "Image Carved Vertically by 50 Pixels (Rows)")

    enlarged_vertical_rows = enlarge_vertically_rows(image, pixels=50)
    display_with_ticks(enlarged_vertical_rows, "Image Enlarged Vertically by 50 Pixels (Rows)")

    # Desired dimensions (replace with your desired width and height)
    target_width = 501
    target_height = 351

    # Resize image to desired dimensions using columns/rows
    resized_image_columns = resize_image_columns(image, target_width, target_height)
    display_with_ticks(resized_image_columns, f"Resized Image to {target_width}x{target_height} (Columns/Rows)")


if __name__ == "__main__":
    main()
