from zensvi.cv import Segmenter
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import cv2

def segmentation_dir(input_path, ouput_path):
    segmenter  = Segmenter()    
    # set arguments
    save_image_options = ["segmented_image", "blend_image"] # segmented_image (colored image), blend_image (blended image)
    segmenter.segment(input_path, ouput_path, ouput_path,
                    save_image_options = save_image_options)
    
def resize_dir(input_dir, ouput_dir, new_size = (2048, 1024)):

    if not os.path.exists(ouput_dir):
        os.makedirs(ouput_dir)
    # Loop through all files in the directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(ouput_dir, filename)
        # Open the image
        with Image.open(input_path) as img:
            # Resize the image
            img = img.resize(new_size)
            # Save the image, replacing the original
            img.save(output_path)

def rgb_to_int(rgb):
    red, green, blue = rgb
    return (red<<16) + (green<<8) + blue

def image_to_2d_array(image_path):
    img = Image.open(image_path)
    data = np.array(img)
    flat_data = data.reshape(-1, data.shape[-1])

    result = np.empty(flat_data.shape[0], dtype=np.int32)
    for i, rgb in enumerate(flat_data):
        result[i] = rgb_to_int(rgb)

    return result.reshape(data.shape[:2])

def get_transmittance(img_path, seg_path, kernel_size, border, binary_type = "brightness", model = "tcm"):
    if model == "tcm":
        img = Image.open(img_path)
        # Split the image into R, G, B components
        r, g, b = img.split()
        # Convert R, G, B components to float numpy arrays
        r = np.array(r, dtype=float)
        g = np.array(g, dtype=float)
        b = np.array(b, dtype=float)

        if binary_type == "rgb":
            # Calculate 'b' value for each pixel
            b_value = (r + g + b) / 3
        elif binary_type == "brightness":
            # Calculate 'b' value for each pixel
            b_value = (0.5 * r + g + 1.5 * b) / 3
        elif binary_type == "2bg":
            # Calculate 'b' value for each pixel
            b_value = 2 * b - g    
        elif binary_type == "gla":
            # Calculate 'b' value for each pixel
            gla = (2 * g - r - b) / (2 * g + r + b + 1e-10)   
            b_value = (1 - (gla + 1) / 2.0) * 255

        # Binarize based on the threshold
        binarized = b_value > border
        # Convert the binarized array back to a PIL image and convert to mode '1' for 1-bit pixels
        img_binary = Image.fromarray(binarized).convert('1')

        # Create binary 2d array
        array_binary = np.array(img_binary)
        # Convert binary image array into 1's (color) and 0's (white)
        array_binary = np.where(array_binary == True, 1, 0)
        # Create a new array of zeros with the same shape as array_binary for storing shaded image
        array_transmission = np.zeros_like(array_binary, dtype=float)
        array_seg_binary = np.zeros_like(array_binary, dtype=float)

        img_arr = image_to_2d_array(seg_path)
        sky_indices = img_arr[:, :] == 4620980
        veg_indices = img_arr[:, :] == 7048739
        img_arr[:, :] = 0
        img_arr[sky_indices] = 2
        img_arr[veg_indices] = 1

        # Iterate over the image, and calculate average value in kernel_size x kernel_size neighborhood
        for i in range(array_binary.shape[0]):
            for j in range(array_binary.shape[1]):
                if img_arr[i, j] == 1:
                    # Define boundaries for the neighborhood
                    i_min = max(0, i - kernel_size // 2)
                    i_max = min(array_binary.shape[0], i + kernel_size // 2 + 1)
                    j_min = max(0, j - kernel_size // 2)
                    j_max = min(array_binary.shape[1], j + kernel_size // 2 + 1)

                    # Extract the neighborhood
                    neighborhood = array_binary[i_min:i_max, j_min:j_max]
                    # Compute the mean of the neighborhood
                    array_transmission[i, j] = neighborhood.mean()

                    array_seg_binary[i, j] = array_binary[i, j]
                elif img_arr[i, j] == 2:
                    array_transmission[i, j] = 1
                    array_seg_binary[i, j] = 1
                else:
                    array_transmission[i, j] = 0
                    array_seg_binary[i, j] = 0
        
        # Convert 1 to black and 0 to white
        array_seg_binary = np.where(array_seg_binary == 0, 0, 1)
        
        return array_transmission, array_seg_binary
    elif model == "scm":
        img = Image.open(img_path)
        # Convert to grayscale
        gray_img = img.convert('L')
        # Convert the image to a numpy array
        array_2d = np.array(gray_img)
        # Create a new array of zeros with the same shape as array_binary for storing shaded image
        array_transmission = np.zeros_like(array_2d, dtype=float)
        array_seg_binary = np.zeros_like(array_2d, dtype=float)

        img_arr = image_to_2d_array(seg_path)
        sky_indices = img_arr[:, :] == 4620980
        veg_indices = img_arr[:, :] == 7048739
        img_arr[:, :] = 0
        img_arr[sky_indices] = 2
        img_arr[veg_indices] = 1

        # Iterate over the image, and calculate average value in kernel_size x kernel_size neighborhood
        for i in range(array_2d.shape[0]):
            for j in range(array_2d.shape[1]):
                if img_arr[i, j] == 1:
                    array_transmission[i, j] = 0
                    array_seg_binary[i, j] = 0
                elif img_arr[i, j] == 2:
                    array_transmission[i, j] = 1
                    array_seg_binary[i, j] = 1
                else:
                    array_transmission[i, j] = 0
                    array_seg_binary[i, j] = 0
        
        # Convert 1 to black and 0 to white
        array_seg_binary = np.where(array_seg_binary == 0, 0, 1)
        
        return array_transmission, array_seg_binary
    elif model == 'bin':
        img = Image.open(img_path)
        # Split the image into R, G, B components
        r, g, b = img.split()
        # Convert R, G, B components to float numpy arrays
        r = np.array(r, dtype=float)
        g = np.array(g, dtype=float)
        b = np.array(b, dtype=float)

        if binary_type == "rgb":
            # Calculate 'b' value for each pixel
            b_value = (r + g + b) / 3
        elif binary_type == "brightness":
            # Calculate 'b' value for each pixel
            b_value = (0.5 * r + g + 1.5 * b) / 3
        elif binary_type == "2bg":
            # Calculate 'b' value for each pixel
            b_value = 2 * b - g    
        elif binary_type == "gla":
            # Calculate 'b' value for each pixel
            gla = (2 * g - r - b) / (2 * g + r + b + 1e-10)   
            b_value = (1 - (gla + 1) / 2.0) * 255

        # Binarize based on the threshold
        binarized = b_value > border
        # Convert the binarized array back to a PIL image and convert to mode '1' for 1-bit pixels
        img_binary = Image.fromarray(binarized).convert('1')

        # Create binary 2d array
        array_binary = np.array(img_binary)
        # Convert binary image array into 1's (color) and 0's (white)
        array_binary = np.where(array_binary == True, 1, 0)
        # Create a new array of zeros with the same shape as array_binary for storing shaded image
        array_transmission = np.zeros_like(array_binary, dtype=float)
        array_seg_binary = np.zeros_like(array_binary, dtype=float)

        # Iterate over the image, and calculate average value in kernel_size x kernel_size neighborhood
        for i in range(array_binary.shape[0]):
            for j in range(array_binary.shape[1]):
                # Define boundaries for the neighborhood
                i_min = max(0, i - kernel_size // 2)
                i_max = min(array_binary.shape[0], i + kernel_size // 2 + 1)
                j_min = max(0, j - kernel_size // 2)
                j_max = min(array_binary.shape[1], j + kernel_size // 2 + 1)

                # Extract the neighborhood
                neighborhood = array_binary[i_min:i_max, j_min:j_max]
                # Compute the mean of the neighborhood
                array_transmission[i, j] = neighborhood.mean()

                array_seg_binary[i, j] = array_binary[i, j]
        
        # Convert 1 to black and 0 to white
        array_seg_binary = np.where(array_seg_binary == 0, 0, 1)
        
        return array_transmission, array_seg_binary

def get_transmittance_center_of_modes(img_path, seg_path, kernel_size, binary_type = "brightness", model = "tcm", type = "q2"):
    if model == "tcm":
        img = Image.open(img_path)
        # Split the image into R, G, B components
        r, g, b = img.split()
        # Convert R, G, B components to float numpy arrays
        r = np.array(r, dtype=float)
        g = np.array(g, dtype=float)
        b = np.array(b, dtype=float)

        if binary_type == "rgb":
            # Calculate 'b' value for each pixel
            b_value = (r + g + b) / 3
        elif binary_type == "brightness":
            # Calculate 'b' value for each pixel
            b_value = (0.5 * r + g + 1.5 * b) / 3
        elif binary_type == "2bg":
            # Calculate 'b' value for each pixel
            b_value = 2 * b - g    
        elif binary_type == "gla":
            # Calculate 'b' value for each pixel
            gla = (2 * g - r - b) / (2 * g + r + b + 1e-10)   
            b_value = (1 - (gla + 1) / 2.0) * 255

        img_arr = image_to_2d_array(seg_path)
        sky_indices = img_arr[:, :] == 4620980
        veg_indices = img_arr[:, :] == 7048739
        # bld_indices = img_arr[:, :] == 4605510
       
        # sky_ave = np.mean(b_value[sky_indices])
        veg_b_value = b_value[veg_indices]
        sky_b_value = b_value[sky_indices]
        # bld_b_value = b_value[bld_indices]

        bin_width = 10
        bins = np.arange(-200, 340 + bin_width, bin_width)

        bins_veg = np.arange(-200, 100 + bin_width, bin_width)
        # Compute the histogram
        veg_hist, veg_bin_edges = np.histogram(veg_b_value, bins=bins_veg)
        # Find the peak of the histogram (mode)
        # The peak will correspond to the bin with the maximum count
        veg_peak_bin = np.argmax(veg_hist)
        veg_peak_value = veg_bin_edges[veg_peak_bin] + bin_width/2

        # Compute the histogram
        sky_hist, sky_bin_edges = np.histogram(sky_b_value, bins=bins)
        # Find the peak of the histogram (mode)
        # The peak will correspond to the bin with the maximum count
        sky_peak_bin = np.argmax(sky_hist)
        sky_peak_value = sky_bin_edges[sky_peak_bin] + bin_width/2
        
        # border = (veg_peak_value + sky_peak_value) / 2

        gap = (sky_peak_value - veg_peak_value)
        if type == "q1":
            border = veg_peak_value + 0.25 * gap
        if type == "q2":
            border = veg_peak_value + 0.5 * gap
        if type == "q3":
            border = veg_peak_value + 0.75 * gap

        # Binarize based on the threshold
        binarized = b_value > border
        # Convert the binarized array back to a PIL image and convert to mode '1' for 1-bit pixels
        img_binary = Image.fromarray(binarized).convert('1')

        # Create binary 2d array
        array_binary = np.array(img_binary)
        # Convert binary image array into 1's (color) and 0's (white)
        array_binary = np.where(array_binary == True, 1, 0)
        # Create a new array of zeros with the same shape as array_binary for storing shaded image
        array_transmission = np.zeros_like(array_binary, dtype=float)
        array_seg_binary = np.zeros_like(array_binary, dtype=float)

        img_arr = image_to_2d_array(seg_path)
        sky_indices = img_arr[:, :] == 4620980
        veg_indices = img_arr[:, :] == 7048739
        img_arr[:, :] = 0
        img_arr[sky_indices] = 2
        img_arr[veg_indices] = 1

        half_height = array_binary.shape[0] // 2  # Calculate half the width of the array

        for i in range(half_height):
            for j in range(array_binary.shape[1]):  # Loop only up to half width
                if img_arr[i, j] == 1:
                    # Define boundaries for the neighborhood
                    i_min = max(0, i - kernel_size // 2)
                    i_max = min(half_height, i + kernel_size // 2 + 1)
                    j_min = max(0, j - kernel_size // 2)
                    j_max = min(array_binary.shape[1], j + kernel_size // 2 + 1)

                    # Extract the neighborhood
                    neighborhood = array_binary[i_min:i_max, j_min:j_max]
                    # Compute the mean of the neighborhood
                    array_transmission[i, j] = neighborhood.mean()

                    array_seg_binary[i, j] = array_binary[i, j]
                elif img_arr[i, j] == 2:
                    array_transmission[i, j] = 1
                    array_seg_binary[i, j] = 1
                else:
                    array_transmission[i, j] = 0
                    array_seg_binary[i, j] = 0

        # Set the second half of each row in array_seg_binary to 0
        array_seg_binary[half_height:, :] = 0
        
        # Convert 1 to black and 0 to white
        array_seg_binary = np.where(array_seg_binary == 0, 0, 1)
        
        return array_transmission, array_seg_binary
    elif model == "scm":
        img = Image.open(img_path)
        # Convert to grayscale
        gray_img = img.convert('L')
        # Convert the image to a numpy array
        array_2d = np.array(gray_img)
        # Create a new array of zeros with the same shape as array_binary for storing shaded image
        array_transmission = np.zeros_like(array_2d, dtype=float)
        array_seg_binary = np.zeros_like(array_2d, dtype=float)

        img_arr = image_to_2d_array(seg_path)
        sky_indices = img_arr[:, :] == 4620980
        veg_indices = img_arr[:, :] == 7048739
        img_arr[:, :] = 0
        img_arr[sky_indices] = 2
        img_arr[veg_indices] = 1

        # Iterate over the image, and calculate average value in kernel_size x kernel_size neighborhood
        for i in range(array_2d.shape[0]):
            for j in range(array_2d.shape[1]):
                if img_arr[i, j] == 1:
                    array_transmission[i, j] = 0
                    array_seg_binary[i, j] = 0
                elif img_arr[i, j] == 2:
                    array_transmission[i, j] = 1
                    array_seg_binary[i, j] = 1
                else:
                    array_transmission[i, j] = 0
                    array_seg_binary[i, j] = 0
        
        # Convert 1 to black and 0 to white
        array_seg_binary = np.where(array_seg_binary == 0, 0, 1)
        
        return array_transmission, array_seg_binary

def get_transmittance_center_of_modes_upper(img_path, seg_path, kernel_size, binary_type = "brightness", model = "tcm", type = "q2"):
    if model == "tcm":
        img = Image.open(img_path)
        # Split the image into R, G, B components
        r, g, b = img.split()
        # Convert R, G, B components to float numpy arrays
        r = np.array(r, dtype=float)
        g = np.array(g, dtype=float)
        b = np.array(b, dtype=float)

        if binary_type == "rgb":
            # Calculate 'b' value for each pixel
            b_value = (r + g + b) / 3
        elif binary_type == "brightness":
            # Calculate 'b' value for each pixel
            b_value = (0.5 * r + g + 1.5 * b) / 3
        elif binary_type == "2bg":
            # Calculate 'b' value for each pixel
            b_value = 2 * b - g    
        elif binary_type == "gla":
            # Calculate 'b' value for each pixel
            gla = (2 * g - r - b) / (2 * g + r + b + 1e-10)   
            b_value = (1 - (gla + 1) / 2.0) * 255

        img_arr = image_to_2d_array(seg_path)

        # Adjust indices to only consider the upper half of the image
        height = img_arr.shape[0]
        upper_half = img_arr[:height // 2, :]

        # Update indices for sky and vegetation in the upper half
        sky_indices = upper_half == 4620980
        veg_indices = upper_half == 7048739

        # Calculate 'b' values for vegetation and sky in the upper half
        veg_b_value = b_value[:height // 2, :][veg_indices]
        sky_b_value = b_value[:height // 2, :][sky_indices]

        bin_width = 10
        bins = np.arange(-200, 340 + bin_width, bin_width)

        bins_veg = np.arange(-200, 100 + bin_width, bin_width)
        # Compute the histogram
        veg_hist, veg_bin_edges = np.histogram(veg_b_value, bins=bins_veg)
        # Find the peak of the histogram (mode)
        # The peak will correspond to the bin with the maximum count
        veg_peak_bin = np.argmax(veg_hist)
        veg_peak_value = veg_bin_edges[veg_peak_bin] + bin_width/2

        # Compute the histogram
        sky_hist, sky_bin_edges = np.histogram(sky_b_value, bins=bins)
        # Find the peak of the histogram (mode)
        # The peak will correspond to the bin with the maximum count
        sky_peak_bin = np.argmax(sky_hist)
        sky_peak_value = sky_bin_edges[sky_peak_bin] + bin_width/2
        
        # border = (veg_peak_value + sky_peak_value) / 2

        gap = (sky_peak_value - veg_peak_value)
        if type == "q1":
            border = veg_peak_value + 0.25 * gap
        if type == "q2":
            border = veg_peak_value + 0.5 * gap
        if type == "q3":
            border = veg_peak_value + 0.75 * gap

        # Binarize based on the threshold
        binarized = b_value > border
        # Convert the binarized array back to a PIL image and convert to mode '1' for 1-bit pixels
        img_binary = Image.fromarray(binarized).convert('1')

        # Create binary 2d array
        array_binary = np.array(img_binary)
        # Convert binary image array into 1's (color) and 0's (white)
        array_binary = np.where(array_binary == True, 1, 0)
        # Create a new array of zeros with the same shape as array_binary for storing shaded image
        array_transmission = np.zeros_like(array_binary, dtype=float)
        array_seg_binary = np.zeros_like(array_binary, dtype=float)

        img_arr = image_to_2d_array(seg_path)
        sky_indices = img_arr[:, :] == 4620980
        veg_indices = img_arr[:, :] == 7048739
        img_arr[:, :] = 0
        img_arr[sky_indices] = 2
        img_arr[veg_indices] = 1

        half_height = array_binary.shape[0] // 2  # Calculate half the width of the array

        for i in range(half_height):
            for j in range(array_binary.shape[1]):  # Loop only up to half width
                if img_arr[i, j] == 1:
                    # Define boundaries for the neighborhood
                    i_min = max(0, i - kernel_size // 2)
                    i_max = min(half_height, i + kernel_size // 2 + 1)
                    j_min = max(0, j - kernel_size // 2)
                    j_max = min(array_binary.shape[1], j + kernel_size // 2 + 1)

                    # Extract the neighborhood
                    neighborhood = array_binary[i_min:i_max, j_min:j_max]
                    # Compute the mean of the neighborhood
                    array_transmission[i, j] = neighborhood.mean()

                    array_seg_binary[i, j] = array_binary[i, j]
                elif img_arr[i, j] == 2:
                    array_transmission[i, j] = 1
                    array_seg_binary[i, j] = 1
                else:
                    array_transmission[i, j] = 0
                    array_seg_binary[i, j] = 0

        # Set the second half of each row in array_seg_binary to 0
        array_seg_binary[half_height:, :] = 0
        
        # Convert 1 to black and 0 to white
        array_seg_binary = np.where(array_seg_binary == 0, 0, 1)
        
        return array_transmission, array_seg_binary
    elif model == "scm":
        img = Image.open(img_path)
        # Convert to grayscale
        gray_img = img.convert('L')
        # Convert the image to a numpy array
        array_2d = np.array(gray_img)
        # Create a new array of zeros with the same shape as array_binary for storing shaded image
        array_transmission = np.zeros_like(array_2d, dtype=float)
        array_seg_binary = np.zeros_like(array_2d, dtype=float)

        img_arr = image_to_2d_array(seg_path)
        sky_indices = img_arr[:, :] == 4620980
        veg_indices = img_arr[:, :] == 7048739
        img_arr[:, :] = 0
        img_arr[sky_indices] = 2
        img_arr[veg_indices] = 1

        # Iterate over the image, and calculate average value in kernel_size x kernel_size neighborhood
        for i in range(array_2d.shape[0]):
            for j in range(array_2d.shape[1]):
                if img_arr[i, j] == 1:
                    array_transmission[i, j] = 0
                    array_seg_binary[i, j] = 0
                elif img_arr[i, j] == 2:
                    array_transmission[i, j] = 1
                    array_seg_binary[i, j] = 1
                else:
                    array_transmission[i, j] = 0
                    array_seg_binary[i, j] = 0
        
        # Convert 1 to black and 0 to white
        array_seg_binary = np.where(array_seg_binary == 0, 0, 1)
        
        return array_transmission, array_seg_binary
# def get_transmittance_seg(img_path, seg_path):
#     img = Image.open(img_path)
#     # Convert to grayscale
#     gray_img = img.convert('L')
#     # Convert the image to a numpy array
#     array_2d = np.array(gray_img)
#     # Create a new array of zeros with the same shape as array_binary for storing shaded image
#     array_transmission = np.zeros_like(array_2d, dtype=float)
#     array_seg_binary = np.zeros_like(array_2d, dtype=float)

#     img_arr = image_to_2d_array(seg_path)
#     sky_indices = img_arr[:, :] == 4620980
#     veg_indices = img_arr[:, :] == 7048739
#     img_arr[:, :] = 0
#     img_arr[sky_indices] = 2
#     img_arr[veg_indices] = 1

#     # Iterate over the image, and calculate average value in kernel_size x kernel_size neighborhood
#     for i in range(array_2d.shape[0]):
#         for j in range(array_2d.shape[1]):
#             if img_arr[i, j] == 1:
#                 array_transmission[i, j] = 0
#                 array_seg_binary[i, j] = 0
#             elif img_arr[i, j] == 2:
#                 array_transmission[i, j] = 1
#                 array_seg_binary[i, j] = 1
#             else:
#                 array_transmission[i, j] = 0
#                 array_seg_binary[i, j] = 0
    
#     # Convert 1 to black and 0 to white
#     array_seg_binary = np.where(array_seg_binary == 0, 0, 1)
    
#     return array_transmission, array_seg_binary

# def get_transmittance_bin(img_path, kernel_size, border, binary_type = "rgb"):
#     img = Image.open(img_path)
#     # Split the image into R, G, B components
#     r, g, b = img.split()
#     # Convert R, G, B components to float numpy arrays
#     r = np.array(r, dtype=float)
#     g = np.array(g, dtype=float)
#     b = np.array(b, dtype=float)

#     if binary_type == "rgb":
#         # Calculate 'b' value for each pixel
#         b_value = (r + g + b) / 3
#     elif binary_type == "brightness":
#         # Calculate 'b' value for each pixel
#         b_value = (0.5 * r + g + 1.5 * b) / 3
#     elif binary_type == "2bg":
#         # Calculate 'b' value for each pixel
#         b_value = 2 * b - g    
#     elif binary_type == "gla":
#         # Calculate 'b' value for each pixel
#         gla = (2 * g - r - b) / (2 * g + r + b + 1e-10)   
#         b_value = (1 - (gla + 1) / 2.0) * 255

#     # Binarize based on the threshold
#     binarized = b_value > border
#     # Convert the binarized array back to a PIL image and convert to mode '1' for 1-bit pixels
#     img_binary = Image.fromarray(binarized).convert('1')

#     # Create binary 2d array
#     array_binary = np.array(img_binary)
#     # Convert binary image array into 1's (color) and 0's (white)
#     array_binary = np.where(array_binary == True, 1, 0)
#     # Create a new array of zeros with the same shape as array_binary for storing shaded image
#     array_transmission = np.zeros_like(array_binary, dtype=float)
#     array_seg_binary = np.zeros_like(array_binary, dtype=float)

#     # Iterate over the image, and calculate average value in kernel_size x kernel_size neighborhood
#     for i in range(array_binary.shape[0]):
#         for j in range(array_binary.shape[1]):
#             # Define boundaries for the neighborhood
#             i_min = max(0, i - kernel_size // 2)
#             i_max = min(array_binary.shape[0], i + kernel_size // 2 + 1)
#             j_min = max(0, j - kernel_size // 2)
#             j_max = min(array_binary.shape[1], j + kernel_size // 2 + 1)

#             # Extract the neighborhood
#             neighborhood = array_binary[i_min:i_max, j_min:j_max]
#             # Compute the mean of the neighborhood
#             array_transmission[i, j] = neighborhood.mean()

#             array_seg_binary[i, j] = array_binary[i, j]
    
#     # Convert 1 to black and 0 to white
#     array_seg_binary = np.where(array_seg_binary == 0, 0, 1)
    
#     return array_transmission, array_seg_binary

def orthographic_fisheye(img):
    rows, cols, c = img.shape
    R = cols / (2 * math.pi)
    D = int(2 * R)
    cx, cy = R, R

    x, y = np.meshgrid(np.arange(D), np.arange(D))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    theta = np.arctan2(y - cy, x - cx)

    xp = np.floor((theta + np.pi) * cols / (2 * np.pi)).astype(int)
    yp = np.floor((2 / np.pi) * np.arcsin(r / R) * rows).astype(int)
    
    mask = r < R

    new_img = np.zeros((D, D, c), dtype=np.uint8)
    new_img.fill(255)
    new_img[y[mask], x[mask]] = img[yp[mask], xp[mask]]

    return new_img

def orthographic_fisheye_dir(input_dir, output_dir):
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image_path = os.path.join(input_dir, file_name)
            # image = Image.open(image_path)
            image = cv2.imread(image_path)
            ortho_fisheye = orthographic_fisheye(image)
            # ortho_fisheye.save(os.path.join(output_dir, file_name))
            cv2.imwrite(os.path.join(output_dir, file_name), ortho_fisheye)

def orthographic_fisheye_binary(img):
    rows, cols = img.shape
    R = cols / (2 * math.pi)
    D = int(2 * R)
    cx, cy = R, R

    x, y = np.meshgrid(np.arange(D), np.arange(D))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    theta = np.arctan2(y - cy, x - cx)

    xp = np.floor((theta + np.pi) * cols / (2 * np.pi)).astype(int)
    yp = np.floor((2 / np.pi) * np.arcsin(r / R) * rows).astype(int)
    
    mask = r < R

    new_img = np.zeros((D, D), dtype=np.uint8)
    new_img.fill(255)
    new_img[y[mask], x[mask]] = img[yp[mask], xp[mask]]

    return new_img


def get_sky_view_factor_from_binary(array_seg_binary):
    height, width = array_seg_binary.shape
    # Calculate the middle row
    middle_row = height // 2
    img_upper = array_seg_binary[:middle_row, :]
    img_upper_ortho = orthographic_fisheye_binary(img_upper*255)

    # plt.imshow(img_upper_ortho, cmap='gray')
    # plt.axis('off')

    black_pixels = np.sum(img_upper_ortho == 0)

    dia, dia = img_upper_ortho.shape

    radius = dia/2
    area = radius**2*np.pi

    return 1-black_pixels/area, img_upper_ortho

def pixel_value_hist(img_path, seg_path, binary_type = "rgb"):
    img = Image.open(img_path)
    # Split the image into R, G, B components
    r, g, b = img.split()
    # Convert R, G, B components to float numpy arrays
    r = np.array(r, dtype=float)
    g = np.array(g, dtype=float)
    b = np.array(b, dtype=float)

    if binary_type == "rgb":
        # Calculate 'b' value for each pixel
        b_value = (r + g + b) / 3
    elif binary_type == "brightness":
        # Calculate 'b' value for each pixel
        b_value = (0.5 * r + g + 1.5 * b) / 3
    elif binary_type == "2bg":
        # Calculate 'b' value for each pixel
        b_value = 2 * b - g    
    elif binary_type == "gla":
        # Calculate 'b' value for each pixel
        gla = (2 * g - r - b) / (2 * g + r + b + 1e-10)   
        b_value = (1 - (gla + 1) / 2.0) * 255

    img_arr = image_to_2d_array(seg_path)
    sky_indices = img_arr[:, :] == 4620980
    veg_indices = img_arr[:, :] == 7048739
    bld_indices = img_arr[:, :] == 4605510

    # Count the occurrences of each unique value in the array
    unique, counts = np.unique(img_arr, return_counts=True)
    count_by_values = dict(zip(unique, counts))
    count_by_values = dict(sorted(count_by_values.items(), key=lambda item: item[1], reverse=True))
    # print(count_by_values)

    # img_arr[:, :] = 0
    # img_arr[bld_indices] = 3

    # display_image(img_arr, 'gray')

    # sky_ave = np.mean(b_value[sky_indices])
    plt.figure(figsize=(6,2))
    plt.hist(b_value[veg_indices], bins=30, alpha=0.5, label='vegetation', color='g')
    plt.hist(b_value[sky_indices], bins=30, alpha=0.5, label='sky', color='b')
    plt.hist(b_value[bld_indices], bins=30, alpha=0.5, label='buildings', color='grey')
    # plt.hist(b_value[veg_indices], bins=25, edgecolor='black')
    # plt.axvline(sky_ave, color='red', linestyle='dashed', linewidth=2)

    # Add titles and labels
    # plt.title('Histogram of Array Data')
    plt.xlabel(binary_type)
    # plt.ylabel('Frequency')

    # Show plot
    plt.show()

def pixel_value_hist_mode(img_path, seg_path, binary_type = "rgb", focus_type='upper'):

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    print(img_name)

    # # Set the style of the plots
    plt.style.use('seaborn-v0_8-darkgrid')

    img = Image.open(img_path)
    # Split the image into R, G, B components
    r, g, b = img.split()
    # Convert R, G, B components to float numpy arrays
    r = np.array(r, dtype=float)
    g = np.array(g, dtype=float)
    b = np.array(b, dtype=float)

    if binary_type == "rgb":
        # Calculate 'b' value for each pixel
        b_value = (r + g + b) / 3
    elif binary_type == "brightness":
        # Calculate 'b' value for each pixel
        b_value = (0.5 * r + g + 1.5 * b) / 3
    elif binary_type == "2bg":
        # Calculate 'b' value for each pixel
        b_value = 2 * b - g    
    elif binary_type == "gla":
        # Calculate 'b' value for each pixel
        gla = (2 * g - r - b) / (2 * g + r + b + 1e-10)   
        b_value = (1 - (gla + 1) / 2.0) * 255

    img_arr = image_to_2d_array(seg_path)

    if focus_type == 'entire':        
        sky_indices = img_arr[:, :] == 4620980
        veg_indices = img_arr[:, :] == 7048739
        # bld_indices = img_arr[:, :] == 4605510
        
        # # sky_ave = np.mean(b_value[sky_indices])
        veg_b_value = b_value[veg_indices]
        sky_b_value = b_value[sky_indices]
        # bld_b_value = b_value[bld_indices]
    elif focus_type == 'upper':
        # Adjust indices to only consider the upper half of the image
        height = img_arr.shape[0]
        upper_half = img_arr[:height // 2, :]

        # Update indices for sky and vegetation in the upper half
        sky_indices = upper_half == 4620980
        veg_indices = upper_half == 7048739

        # Calculate 'b' values for vegetation and sky in the upper half
        veg_b_value = b_value[:height // 2, :][veg_indices]
        sky_b_value = b_value[:height // 2, :][sky_indices]    

    # Count the occurrences of each unique value in the array
    # unique, counts = np.unique(img_arr, return_counts=True)
    # count_by_values = dict(zip(unique, counts))
    # count_by_values = dict(sorted(count_by_values.items(), key=lambda item: item[1], reverse=True))
    # print(count_by_values)

    # img_arr[:, :] = 0
    # img_arr[bld_indices] = 3

    # display_image(img_arr, 'gray')



    bin_width = 10
    bins = np.arange(0, 250 + bin_width, bin_width)

    plt.figure(figsize=(1.6, 1.6))
    plt.hist(sky_b_value, bins=bins, density=True, alpha=0.5, label='sky', color='b')
    plt.hist(veg_b_value, bins=bins, density=True, alpha=0.5, label='vegetation', color='g')
    # plt.hist(bld_b_value, bins=100, alpha=0.5, label='buildings', color='grey')

    # veg_mode_value, veg_count = mode(veg_b_value[veg_b_value<100])
    # plt.axvline(veg_mode_value, color='g', linestyle='dashed', linewidth=0.25, label=f'mode at {veg_mode_value:.2f}')
    # sky_mode_value, sky_count = mode(sky_b_value)
    # plt.axvline(sky_mode_value, color='b', linestyle='dashed', linewidth=0.25, label=f'mode at {sky_mode_value:.2f}')
    # bld_mode_value, bld_count = mode(bld_b_value)
    # plt.axvline(bld_mode_value, color='grey', linestyle='dashed', linewidth=2, label=f'mode at {bld_mode_value:.2f}')

    bins_veg = np.arange(-200, 100 + bin_width, bin_width)
    # Compute the histogram
    if focus_type == 'entire':
        veg_hist, veg_bin_edges = np.histogram(b_value[veg_indices], bins=bins_veg)
    elif focus_type == 'upper':
        veg_hist, veg_bin_edges = np.histogram(b_value[:height // 2, :][veg_indices], bins=bins_veg)
        
    # Find the peak of the histogram (mode)
    # The peak will correspond to the bin with the maximum count
    veg_peak_bin = np.argmax(veg_hist)
    veg_peak_value = veg_bin_edges[veg_peak_bin] + bin_width/2

    # Compute the histogram
    if focus_type == 'entire':
        sky_hist, sky_bin_edges = np.histogram(b_value[sky_indices], bins=bins)
    elif focus_type == 'upper':
        sky_hist, sky_bin_edges = np.histogram(b_value[:height // 2, :][sky_indices], bins=bins)

    # Find the peak of the histogram (mode)
    # The peak will correspond to the bin with the maximum count
    sky_peak_bin = np.argmax(sky_hist)
    sky_peak_value = sky_bin_edges[sky_peak_bin] + bin_width/2

    plt.axvline(veg_peak_value, color='g', linestyle='dashed', linewidth=1, label=f'mode at {veg_peak_value:.2f}')
    plt.axvline(sky_peak_value, color='b', linestyle='dashed', linewidth=1, label=f'mode at {sky_peak_value:.2f}')

    # # Compute the histogram
    # bld_hist, bld_bin_edges = np.histogram(b_value[bld_indices], bins=30)
    # # Find the peak of the histogram (mode)
    # # The peak will correspond to the bin with the maximum count
    # bld_peak_bin = np.argmax(bld_hist)
    # bld_peak_value = bld_bin_edges[bld_peak_bin]

    # Visualizing the histogram
    # plt.hist(data, bins=30, alpha=0.75, label='Data histogram')
    # plt.axvline(peak_value, color='r', linestyle='dashed', linewidth=1, label=f'Peak at {peak_value:.2f}')
    # plt.title("Histogram and its Peak")
    # plt.xlabel("Data values")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.show()

    # plt.hist(b_value[veg_indices], bins=25, edgecolor='black')
    # plt.axvline(sky_ave, color='red', linestyle='dashed', linewidth=2)

    # Add titles and labels
    # plt.title('Histogram of Array Data')
    plt.xlabel("Brightness, $b$")
    plt.ylabel('Percentage')
    plt.tight_layout()
    
    svg_file_path = os.path.join("figure", f"pixel_hist_{img_name}.svg")
    plt.savefig(svg_file_path, format='svg')
    # Show plot
    plt.show()

def shift_image_by_degree(image, degree):
    """
    Shift an image horizontally by a given degree, wrapping around the image.

    Parameters:
    - image: A PIL Image object.
    - degree: The number of degrees to shift the image.

    Returns: The shifted image.
    """
    shift = int(image.width * (degree / 360.0))
    image_array = np.array(image)
    image_shifted = np.roll(image_array, shift, axis=1)
    return Image.fromarray(image_shifted)

def split_image(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        upper_half = img.crop((0, 0, width, height//2))
        lower_half = img.crop((0, height//2, width, height))
    return upper_half, lower_half

def split_images_dir(input_dir, output_dir_upper, output_dir_lower):
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image_path = os.path.join(input_dir, file_name)
            upper_half, lower_half = split_image(image_path)
            upper_half.save(os.path.join(output_dir_upper, file_name))
            lower_half.rotate(180, expand=True).save(os.path.join(output_dir_lower, file_name))

def rotate_images_dir(input_dir, output_dir, angle):
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.jpg') or file_name.endswith('.png'):
            image_path = os.path.join(input_dir, file_name)
            image = Image.open(image_path)
            # image = cv2.imread(image_path)
            rotated = image.rotate(angle)
            rotated.save(os.path.join(output_dir, file_name))
            # cv2.imwrite(os.path.join(output_dir, file_name), ortho_fisheye)

def calculate_black_pixel_ratio(image_path):
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert image to grayscale

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Binarize the image, assuming a threshold (128 used here)
    threshold = 128
    binary_image = img_array < threshold  # This will create a binary image

    # Calculate the total number of pixels
    total_pixels = binary_image.size

    # Calculate the number of black pixels
    black_pixels = np.sum(binary_image)  # Since True is 1 and False is 0

    # Calculate the ratio of black pixels
    black_pixel_ratio = black_pixels / total_pixels

    return black_pixel_ratio