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


def build_cost_matrix(energy):
    """
    Build the accumulated cost matrix for dynamic programming.

    The cost matrix helps in finding the seam with the lowest energy to remove
    or duplicate. It accumulates the minimum energy paths from the top to the
    bottom of the image.

    Args:
        energy (numpy.ndarray): The energy map of the image.

    Returns:
        numpy.ndarray: The accumulated cost matrix.
    """
    rows, cols = energy.shape
    # Initialize the cost matrix with the energy map
    cost = energy.copy()

    # Iterate over each row starting from the second
    for i in range(1, rows):
        for j in range(cols):
            # Handle edge cases for the first and last columns
            left = cost[i - 1, j - 1] if j - 1 >= 0 else float('inf')
            up = cost[i - 1, j]
            right = cost[i - 1, j + 1] if j + 1 < cols else float('inf')
            # Update the cost with the minimum of the three possible paths
            cost[i, j] += min(left, up, right)

    return cost


def find_seam(cost):
    """
    Find the seam to be removed based on the accumulated cost matrix.

    The seam is a path of pixels from the top to the bottom of the image,
    where each pixel in the path is adjacent to the previous one,
    and the total energy is minimized.

    Args:
        cost (numpy.ndarray): The accumulated cost matrix.

    Returns:
        numpy.ndarray: An array of column indices representing the seam.
    """
    rows, cols = cost.shape
    # Initialize the seam array
    seam = np.zeros(rows, dtype=np.int32)
    # Start from the bottom row and find the position with the minimum cost
    seam[-1] = np.argmin(cost[-1])

    # Backtrack from the bottom to the top row to find the seam path
    for i in range(rows - 2, -1, -1):
        prev_x = seam[i + 1]
        # Handle edge cases for the first and last columns
        left = cost[i, prev_x - 1] if prev_x - 1 >= 0 else float('inf')
        up = cost[i, prev_x]
        right = cost[i, prev_x + 1] if prev_x + 1 < cols else float('inf')
        # Find the direction with the minimum cost
        offset = np.argmin([left, up, right]) - 1
        seam[i] = prev_x + offset

    return seam


def adjust_seams(seams):
    """
    Adjust the seam indices to account for previous insertions.

    When inserting multiple seams, the positions of pixels shift.
    This function updates the seam indices to reflect these shifts.

    Args:
        seams (list of numpy.ndarray): A list of seams to be adjusted.

    Returns:
        list of numpy.ndarray: The adjusted seams.
    """
    adjusted_seams = []
    # Initialize a shift map to keep track of index shifts
    seam_shift = np.zeros_like(seams[0], dtype=np.int32)

    for seam in seams:
        # Adjust the current seam based on the shift map
        adjusted_seam = seam + seam_shift
        adjusted_seams.append(adjusted_seam)
        # Increment shift after inserting each seam
        for i in range(len(seam_shift)):
            seam_shift[i] += 1

    return adjusted_seams


def remove_seam(image, seam):
    """
    Remove the seam from the image.

    Args:
        image (numpy.ndarray): The input image.
        seam (numpy.ndarray): The seam to be removed.

    Returns:
        numpy.ndarray: The image with the seam removed.
    """
    rows, cols, _ = image.shape
    # Create a new image with one less column
    new_image = np.zeros((rows, cols - 1, 3), dtype=np.uint8)

    for i in range(rows):
        # Remove the pixel at the seam position for each color channel
        new_image[i, :, 0] = np.delete(image[i, :, 0], seam[i])
        new_image[i, :, 1] = np.delete(image[i, :, 1], seam[i])
        new_image[i, :, 2] = np.delete(image[i, :, 2], seam[i])

    return new_image


def insert_seam(image, seam):
    """
    Insert a seam into the image.

    Args:
        image (numpy.ndarray): The input image.
        seam (numpy.ndarray): The seam to be inserted.

    Returns:
        numpy.ndarray: The image with the seam inserted.
    """
    rows, cols, channels = image.shape
    # Create a new image with one additional column
    new_image = np.zeros((rows, cols + 1, channels), dtype=image.dtype)

    for i in range(rows):
        col = seam[i]
        for ch in range(channels):
            # Copy pixels before the seam
            new_image[i, :col, ch] = image[i, :col, ch]
            if col == 0:
                # If at the first column, duplicate the first pixel
                new_pixel = image[i, col, ch]
            else:
                # Average the left and right pixels to create a new pixel
                left_pixel = image[i, col - 1, ch]
                right_pixel = image[i, col, ch]
                new_pixel = ((left_pixel.astype(np.float32) + right_pixel.astype(np.float32)) / 2).astype(image.dtype)
            # Copy the pixel at the seam
            new_image[i, col, ch] = image[i, col, ch]
            # Insert the new pixel
            new_image[i, col + 1, ch] = new_pixel
            # Copy the remaining pixels
            new_image[i, col + 2:, ch] = image[i, col + 1:, ch]

    return new_image


def carve_horizontally(image, pixels=50):
    """
    Reduce the width of the image by removing vertical seams.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of pixels (seams) to remove.

    Returns:
        numpy.ndarray: The carved image.
    """
    current_image = image.copy()
    for _ in range(pixels):
        # Calculate the energy map
        energy = calculate_energy(current_image)
        # Build the cost matrix
        cost = build_cost_matrix(energy)
        # Find the seam with the least energy
        seam = find_seam(cost)
        # Remove the seam from the image
        current_image = remove_seam(current_image, seam)
    return current_image


def enlarge_horizontally(image, pixels=50):
    """
    Increase the width of the image by inserting vertical seams.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of pixels (seams) to insert.

    Returns:
        numpy.ndarray: The enlarged image.
    """
    rows, cols, channels = image.shape
    temp_image = image.copy()
    seams = []

    # Find multiple seams to insert
    for _ in range(pixels):
        energy = calculate_energy(temp_image)
        cost = build_cost_matrix(energy)
        seam = find_seam(cost)
        seams.append(seam)
        temp_image = remove_seam(temp_image, seam)

    # Reverse seams to insert them in the correct order
    seams = seams[::-1]
    # Adjust seams to account for previous insertions
    adjusted_seams = adjust_seams(seams)

    # Insert seams back into the original image
    enlarged_image = image.copy()
    for seam in adjusted_seams:
        enlarged_image = insert_seam(enlarged_image, seam)

    return enlarged_image


def carve_vertically(image, pixels=50):
    """
    Reduce the height of the image by removing horizontal seams.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of pixels (seams) to remove.

    Returns:
        numpy.ndarray: The carved image.
    """
    # Rotate the image to reuse horizontal seam functions
    rotated_image = np.rot90(image, k=1)
    # Carve horizontally (which is vertically in the original image)
    carved_image = carve_horizontally(rotated_image, pixels)
    # Rotate the image back to original orientation
    return np.rot90(carved_image, k=-1)


def enlarge_vertically(image, pixels=50):
    """
    Increase the height of the image by inserting horizontal seams.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of pixels (seams) to insert.

    Returns:
        numpy.ndarray: The enlarged image.
    """
    # Rotate the image to reuse horizontal seam functions
    rotated_image = np.rot90(image, k=1)
    # Enlarge horizontally (which is vertically in the original image)
    enlarged_image = enlarge_horizontally(rotated_image, pixels)
    # Rotate the image back to original orientation
    return np.rot90(enlarged_image, k=-1)


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
    Resize the image to the target width and height using seam carving and insertion.

    This function intelligently resizes the image by removing or inserting seams
    based on the energy map, thus preserving important content.

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
        # Need to remove seams
        current_image = carve_horizontally(current_image, abs(delta_width))
    elif delta_width > 0:
        # Need to insert seams
        current_image = enlarge_horizontally(current_image, delta_width)

    # Resize height
    if delta_height < 0:
        # Need to remove seams
        current_image = carve_vertically(current_image, abs(delta_height))
    elif delta_height > 0:
        # Need to insert seams
        current_image = enlarge_vertically(current_image, delta_height)

    return current_image


def main():
    """
    Main function to execute the seam carving and enlargement operations.
    """
    # Load input image (ensure the image path is correct)
    image = cv2.imread('image.jpg')

    # Display original image with energy map
    energy_map = calculate_energy(image)
    plt.figure(figsize=(8, 6))
    plt.imshow(energy_map)
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

    # Desired dimensions (replace with your desired width and height)
    target_width = 501
    target_height = 351

    # Resize image to desired dimensions
    resized_image = resize_image(image, target_width, target_height)
    display_with_ticks(resized_image, f"Resized Image to {target_width}x{target_height}")


if __name__ == "__main__":
    main()
