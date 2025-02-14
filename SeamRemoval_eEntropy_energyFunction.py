import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk


def calculate_energy(image):
    """
    Optimized entropy-based energy function for content-aware image resizing.

    This version minimizes artifacts by ensuring smoother transitions
    and reduces computational overhead using efficient calculations.

    Args:
        image (numpy.ndarray): The input image in BGR format.

    Returns:
        numpy.ndarray: The energy map of the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the disk-shaped structuring element for entropy calculation
    structuring_element = disk(4)

    # Compute entropy using skimage's entropy function
    entropy_map = entropy(gray, structuring_element)

    return entropy_map


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
    cost = energy.copy()

    for i in range(1, rows):
        for j in range(cols):
            left = cost[i - 1, j - 1] if j - 1 >= 0 else np.inf
            up = cost[i - 1, j]
            right = cost[i - 1, j + 1] if j + 1 < cols else np.inf
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
    seam = np.zeros(rows, dtype=np.int32)
    seam[-1] = np.argmin(cost[-1])

    for i in range(rows - 2, -1, -1):
        prev_x = seam[i + 1]
        left = cost[i, prev_x - 1] if prev_x - 1 >= 0 else np.inf
        up = cost[i, prev_x]
        right = cost[i, prev_x + 1] if prev_x + 1 < cols else np.inf
        offset = np.argmin([left, up, right]) - 1
        seam[i] = prev_x + offset

    return seam


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
    mask = np.ones((rows, cols), dtype=np.bool_)
    for i in range(rows):
        mask[i, seam[i]] = False
    new_image = image[mask].reshape((rows, cols - 1, 3))
    return new_image


def insert_seams(image, seams):
    """
    Insert multiple seams into the image at once.

    Args:
        image (numpy.ndarray): The input image.
        seams (list of numpy.ndarray): The list of seams to be inserted.

    Returns:
        numpy.ndarray: The image with the seams inserted.
    """
    rows, cols, channels = image.shape
    n_seams = len(seams)
    new_cols = cols + n_seams
    new_image = np.zeros((rows, new_cols, channels), dtype=image.dtype)

    # Create a 2D map to keep track of seam positions
    seam_map = np.zeros((rows, cols), dtype=np.int32)

    for seam_idx, seam in enumerate(seams):
        for i in range(rows):
            seam_map[i, seam[i]] += 1

    for i in range(rows):
        shift = 0
        for j in range(cols):
            count = seam_map[i, j]
            new_image[i, j + shift, :] = image[i, j, :]
            if count > 0:
                # Duplicate the pixel count times
                for k in range(count):
                    if j + shift + 1 < new_cols:
                        new_image[i, j + shift + 1, :] = image[i, j, :]
                    shift += 1

    return new_image


def adjust_seams(seams):
    """
    Adjust seams to account for the shifts caused by inserting previous seams.

    Since we're inserting all seams at once, we don't need to adjust seams here.

    Args:
        seams (list of numpy.ndarray): List of seams to adjust.

    Returns:
        list of numpy.ndarray: The same list of seams.
    """
    return seams


def enlarge_horizontally(image, pixels=50):
    """
    Increase the width of the image by inserting vertical seams.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of pixels (seams) to insert.

    Returns:
        numpy.ndarray: The enlarged image.
    """
    seams = []
    temp_image = image.copy()
    for _ in range(pixels):
        energy = calculate_energy(temp_image)
        cost = build_cost_matrix(energy)
        seam = find_seam(cost)
        seams.append(seam)
        temp_image = remove_seam(temp_image, seam)

    # Since we removed seams to find them, reverse the list
    seams = seams[::-1]

    # Adjust the seams (not needed here, but kept for consistency)
    adjusted_seams = adjust_seams(seams)

    # Insert all seams at once
    enlarged_image = insert_seams(image, adjusted_seams)

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
    rotated_image = np.rot90(image, k=1, axes=(0, 1))
    carved_image = carve_horizontally(rotated_image, pixels)
    return np.rot90(carved_image, k=-1, axes=(0, 1))


def enlarge_vertically(image, pixels=50):
    """
    Increase the height of the image by inserting horizontal seams.

    Args:
        image (numpy.ndarray): The input image.
        pixels (int): Number of pixels (seams) to insert.

    Returns:
        numpy.ndarray: The enlarged image.
    """
    rotated_image = np.rot90(image, k=1, axes=(0, 1))
    enlarged_image = enlarge_horizontally(rotated_image, pixels)
    return np.rot90(enlarged_image, k=-1, axes=(0, 1))


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
        energy = calculate_energy(current_image)
        cost = build_cost_matrix(energy)
        seam = find_seam(cost)
        current_image = remove_seam(current_image, seam)
    return current_image


def display_with_ticks(image, title):
    """
    Display an image with grid ticks.

    Args:
        image (numpy.ndarray): The image to display.
        title (str): The title of the image.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.xticks(range(0, image.shape[1], 50))
    plt.yticks(range(0, image.shape[0], 50))
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

    delta_width = target_width - current_width
    delta_height = target_height - current_height

    if delta_width < 0:
        current_image = carve_horizontally(current_image, abs(delta_width))
    elif delta_width > 0:
        current_image = enlarge_horizontally(current_image, delta_width)

    if delta_height < 0:
        current_image = carve_vertically(current_image, abs(delta_height))
    elif delta_height > 0:
        current_image = enlarge_vertically(current_image, delta_height)

    return current_image


def main():
    """
    Main function to execute the seam carving and enlargement operations.
    """
    image = cv2.imread('image.jpg')
    if image is None:
        print("Error: Image not found or unable to read.")
        return

    # Display original image
    display_with_ticks(image, "Original Image")

    # Display energy map
    energy_map = calculate_energy(image)
    plt.figure(figsize=(8, 6))
    plt.imshow(energy_map, cmap='viridis')
    plt.title("Energy Map (Entropy Function)")
    plt.colorbar()
    plt.show()

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
    target_width = 501
    target_height = 351

    # Resize image to desired dimensions
    resized_image = resize_image(image, target_width, target_height)
    display_with_ticks(resized_image, f"Resized Image to {target_width}x{target_height}")


if __name__ == "__main__":
    main()
