import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_energy(image):
    """
    Calculate the energy map of the image using the E1 energy function.

    The E1 energy function calculates the sum of absolute differences between
    a pixel and its neighboring pixels in the x and y directions.

    Args:
        image (numpy.ndarray): The input image in BGR format.

    Returns:
        numpy.ndarray: The energy map of the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.int32)
    # Shifted images to compute differences
    left = np.roll(gray, 1, axis=1)
    right = np.roll(gray, -1, axis=1)
    up = np.roll(gray, 1, axis=0)
    down = np.roll(gray, -1, axis=0)

    # At boundaries, set neighbors equal to current pixel
    left[:, 0] = gray[:, 0]
    right[:, -1] = gray[:, -1]
    up[0, :] = gray[0, :]
    down[-1, :] = gray[-1, :]

    # Compute differences
    delta_x = np.abs(right - left)
    delta_y = np.abs(down - up)

    # Sum the components
    energy = delta_x + delta_y

    # Convert energy to float64 to handle float values like 'inf'
    energy = energy.astype(np.float64)

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
    new_image = np.zeros((rows, cols + 1, channels), dtype=image.dtype)

    for i in range(rows):
        col = seam[i]
        for ch in range(channels):
            if col == 0:
                p = np.average(image[i, col:col + 2, ch])
                new_image[i, col, ch] = image[i, col, ch]
                new_image[i, col + 1, ch] = p
                new_image[i, col + 2:, ch] = image[i, col + 1:, ch]
            else:
                p = np.average(image[i, col - 1:col + 1, ch])
                new_image[i, :col, ch] = image[i, :col, ch]
                new_image[i, col, ch] = image[i, col, ch]
                new_image[i, col + 1, ch] = p
                new_image[i, col + 2:, ch] = image[i, col + 1:, ch]
    return new_image[:, :cols + 1, :]


def adjust_seams(seams, image_width):
    """
    Adjust seams to account for the shifts caused by inserting previous seams.

    Args:
        seams (list of numpy.ndarray): List of seams to adjust.
        image_width (int): Original width of the image before enlargement.

    Returns:
        list of numpy.ndarray: List of adjusted seams.
    """
    n_seams = len(seams)
    rows = seams[0].shape[0]

    # Initialize a seam_mask with dimensions sufficient to handle maximum indices
    seam_mask = np.zeros((rows, image_width + n_seams), dtype=np.int32)

    adjusted_seams = []
    for seam in seams:
        adjusted_seam = seam.copy()
        for i in range(rows):
            col = adjusted_seam[i]
            # Accumulate the shifts caused by previous seams
            col += seam_mask[i, col]
            adjusted_seam[i] = col
            # Increment the seam_mask to account for the new seam
            seam_mask[i, col:] += 1
        adjusted_seams.append(adjusted_seam)
    return adjusted_seams


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
    energy = calculate_energy(image)
    energy_map = energy.copy()

    for _ in range(pixels):
        cost = build_cost_matrix(energy_map)
        seam = find_seam(cost)
        seams.append(seam)
        # Increase the energy at the seam positions to prevent selecting them again
        for i in range(len(seam)):
            energy_map[i, seam[i]] = np.inf

    # Adjust the seams to account for previous insertions
    adjusted_seams = adjust_seams(seams, image.shape[1])
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
    Display the image with x and y ticks at intervals of 50 pixels.

    Args:
        image (numpy.ndarray): The image to display.
        title (str): The title of the plot.
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
    plt.imshow(energy_map, cmap='gray')
    plt.title("Energy Map (E1 Energy Function)")
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
