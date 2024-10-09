import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import sys
from scipy.ndimage import zoom
from skimage.restoration import denoise_nl_means, estimate_sigma
from colorama import init, Fore, Style
import gc

# Initialize colorama
init(autoreset=True)

def print_title():
    title = f"""
{Fore.CYAN}{Style.BRIGHT}
  █████╗ ███████╗████████╗██████╗  ██████╗  ██████╗ ███╗   ███╗██╗   ██╗
 ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔════╝ ██╔═══██╗████╗ ████║██║   ██║
 ███████║███████╗   ██║   ██████╔╝██║  ███╗██║   ██║██╔████╔██║██║   ██║
 ██╔══██║╚════██║   ██║   ██╔══██╗██║   ██║██║   ██║██║╚██╔╝██║██║   ██║
 ██║  ██║███████║   ██║   ██║  ██║╚██████╔╝╚██████╔╝██║ ╚═╝ ██║╚██████╔╝
 ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝     ╚═╝ ╚═════╝ 
{Style.RESET_ALL}
{Fore.GREEN}{Style.BRIGHT}
AstroImageCombiner - Combine FITS files into a color image with customizable settings.
{Style.RESET_ALL}
"""
    print(title)

# Function to normalize and stretch the data
def normalize(data, stretch='linear', min_percent=0, max_percent=100):
    # Convert data to float
    data = data.astype(np.float32)
    # Clip data based on percentiles
    min_val = np.percentile(data, min_percent)
    max_val = np.percentile(data, max_percent)
    data_clipped = np.clip(data, min_val, max_val)
    # Normalize data to 0-1
    data_norm = (data_clipped - min_val) / (max_val - min_val + 1e-10)
    # Apply stretch function
    if stretch == 'log':
        data_norm = np.log10(data_norm * 9 + 1)
    elif stretch == 'sqrt':
        data_norm = np.sqrt(data_norm)
    return data_norm

# Function to denoise the data
def denoise_data(data, level='medium'):
    # Estimate the noise standard deviation from the data
    sigma_est = np.mean(estimate_sigma(data, channel_axis=None))
    # Set parameters based on the chosen level
    if level == 'low':
        patch_size = 5
        patch_distance = 6
        h = 0.8 * sigma_est
    elif level == 'medium':
        patch_size = 7
        patch_distance = 11
        h = 1.0 * sigma_est
    elif level == 'high':
        patch_size = 9
        patch_distance = 17
        h = 1.2 * sigma_est
    else:
        print("Invalid denoising level selected. Proceeding without denoising.")
        return data
    # Perform Non-Local Means denoising
    denoised = denoise_nl_means(
        data,
        h=h,
        fast_mode=True,
        patch_size=patch_size,
        patch_distance=patch_distance,
        preserve_range=True,
        channel_axis=None
    )
    return denoised

def main():
    print_title()
    print(f"{Fore.GREEN}{Style.BRIGHT}Welcome to AstroImageCombiner!{Style.RESET_ALL}\n")

    # Ask for the folder path containing FITS files
    print(f"{Fore.CYAN}{Style.BRIGHT}Enter the folder path containing the FITS files:{Style.RESET_ALL}")
    folder_path = input("> ").strip()

    # Check if the folder exists
    if not os.path.isdir(folder_path):
        print(f"{Fore.RED}The folder does not exist. Please check the path.")
        sys.exit()

    # List FITS files in the folder
    fits_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.fits', '.fit'))]

    # Check if there are at least three FITS files
    if len(fits_files) < 3:
        print(f"{Fore.RED}There are fewer than three FITS files in the folder.")
        sys.exit()

    print(f"{Fore.CYAN}{Style.BRIGHT}\nFITS files found in the folder:{Style.RESET_ALL}")
    for idx, filename in enumerate(fits_files, 1):
        print(f"{idx}: {filename}")

    # Ask if the user wants to autodetect files
    print(f"{Fore.CYAN}{Style.BRIGHT}\nWould you like to autodetect ", end="")
    print(f"{Fore.RED}red{Fore.CYAN}, {Fore.GREEN}green{Fore.CYAN}, and {Fore.BLUE}blue{Fore.CYAN} files based on their filenames?{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Files containing '{Fore.RED}red{Fore.CYAN}', '{Fore.GREEN}green{Fore.CYAN}', or '{Fore.BLUE}blue{Fore.CYAN}' (case-insensitive) will be assigned accordingly.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}Enter 'yes' to autodetect or 'no' to select files manually:{Style.RESET_ALL}")
    autodetect = input("> ").strip().lower()

    if autodetect == 'yes':
        red_file = green_file = blue_file = None
        for filename in fits_files:
            lower_filename = filename.lower()
            if 'red' in lower_filename and red_file is None:
                red_file = filename
            elif 'green' in lower_filename and green_file is None:
                green_file = filename
            elif 'blue' in lower_filename and blue_file is None:
                blue_file = filename

        if None in [red_file, green_file, blue_file]:
            print(f"{Fore.YELLOW}Could not autodetect all files. Proceeding to manual selection.")
            autodetect = 'no'
        else:
            print(f"{Fore.CYAN}{Style.BRIGHT}\nAutodetected files:{Style.RESET_ALL}")
            print(f"Red channel: {red_file}")
            print(f"Green channel: {green_file}")
            print(f"Blue channel: {blue_file}")
            print(f"{Fore.CYAN}{Style.BRIGHT}Is this correct? Enter 'yes' to proceed or 'no' to select files manually:{Style.RESET_ALL}")
            confirm = input("> ").strip().lower()
            if confirm != 'yes':
                autodetect = 'no'
    else:
        autodetect = 'no'

    if autodetect == 'no':
        # Ask the user to select files for each color channel
        print(f"{Fore.CYAN}{Style.BRIGHT}\nSelect the FITS file for each color channel by entering the corresponding number:{Style.RESET_ALL}")
        try:
            red_index = int(input(f"{Fore.RED}{Style.BRIGHT}Enter the number for the Red channel file:\n> {Style.RESET_ALL}")) - 1
            green_index = int(input(f"{Fore.GREEN}{Style.BRIGHT}Enter the number for the Green channel file:\n> {Style.RESET_ALL}")) - 1
            blue_index = int(input(f"{Fore.BLUE}{Style.BRIGHT}Enter the number for the Blue channel file:\n> {Style.RESET_ALL}")) - 1
        except ValueError:
            print(f"{Fore.RED}Invalid input. Please enter numeric values.")
            sys.exit()

        # Validate the selected indices
        if not all(0 <= idx < len(fits_files) for idx in [red_index, green_index, blue_index]):
            print(f"{Fore.RED}One or more selected numbers are out of range.")
            sys.exit()

        # Get the filenames
        red_file = fits_files[red_index]
        green_file = fits_files[green_index]
        blue_file = fits_files[blue_index]

    # Get the full file paths
    red_path = os.path.join(folder_path, red_file)
    green_path = os.path.join(folder_path, green_file)
    blue_path = os.path.join(folder_path, blue_file)

    # Load the FITS files
    try:
        red_data = fits.getdata(red_path).astype(np.float32)
        green_data = fits.getdata(green_path).astype(np.float32)
        blue_data = fits.getdata(blue_path).astype(np.float32)
    except Exception as e:
        print(f"{Fore.RED}Error loading FITS files: {e}")
        sys.exit()

    # Ask for stretch functions
    print(f"{Fore.CYAN}{Style.BRIGHT}\nChoose stretch function for each channel (linear, log, sqrt). Press Enter to accept default 'linear':{Style.RESET_ALL}")
    print(f"{Fore.CYAN}This affects the contrast of the image.{Style.RESET_ALL}")
    red_stretch = input(f"{Fore.RED}{Style.BRIGHT}Red channel stretch [default 'linear']:\n> {Style.RESET_ALL}").strip().lower() or 'linear'
    green_stretch = input(f"{Fore.GREEN}{Style.BRIGHT}Green channel stretch [default 'linear']:\n> {Style.RESET_ALL}").strip().lower() or 'linear'
    blue_stretch = input(f"{Fore.BLUE}{Style.BRIGHT}Blue channel stretch [default 'linear']:\n> {Style.RESET_ALL}").strip().lower() or 'linear'

    # Validate stretch functions
    valid_stretches = {'linear', 'log', 'sqrt'}
    if not all(stretch in valid_stretches for stretch in [red_stretch, green_stretch, blue_stretch]):
        print(f"{Fore.RED}Invalid stretch function entered. Please use 'linear', 'log', or 'sqrt'.")
        sys.exit()

    # Ask for percentile clipping values
    print(f"{Fore.CYAN}{Style.BRIGHT}\nEnter min and max percentiles for clipping (0-100). Press Enter to accept default values (min:1, max:99):{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Clipping enhances contrast by removing extreme pixel values.{Style.RESET_ALL}")
    try:
        red_min_percent_input = input(f"{Fore.RED}{Style.BRIGHT}Red channel min percentile [default 1]:\n> {Style.RESET_ALL}").strip()
        red_min_percent = float(red_min_percent_input) if red_min_percent_input else 1.0

        red_max_percent_input = input(f"{Fore.RED}{Style.BRIGHT}Red channel max percentile [default 99]:\n> {Style.RESET_ALL}").strip()
        red_max_percent = float(red_max_percent_input) if red_max_percent_input else 99.0

        green_min_percent_input = input(f"{Fore.GREEN}{Style.BRIGHT}Green channel min percentile [default 1]:\n> {Style.RESET_ALL}").strip()
        green_min_percent = float(green_min_percent_input) if green_min_percent_input else 1.0

        green_max_percent_input = input(f"{Fore.GREEN}{Style.BRIGHT}Green channel max percentile [default 99]:\n> {Style.RESET_ALL}").strip()
        green_max_percent = float(green_max_percent_input) if green_max_percent_input else 99.0

        blue_min_percent_input = input(f"{Fore.BLUE}{Style.BRIGHT}Blue channel min percentile [default 1]:\n> {Style.RESET_ALL}").strip()
        blue_min_percent = float(blue_min_percent_input) if blue_min_percent_input else 1.0

        blue_max_percent_input = input(f"{Fore.BLUE}{Style.BRIGHT}Blue channel max percentile [default 99]:\n> {Style.RESET_ALL}").strip()
        blue_max_percent = float(blue_max_percent_input) if blue_max_percent_input else 99.0

    except ValueError:
        print(f"{Fore.RED}Invalid input. Please enter numeric values.")
        sys.exit()

    # Validate percentile values
    if not all(0 <= p <= 100 for p in [
        red_min_percent, red_max_percent,
        green_min_percent, green_max_percent,
        blue_min_percent, blue_max_percent]):
        print(f"{Fore.RED}Percentile values must be between 0 and 100.")
        sys.exit()

    # Normalize each channel
    print(f"{Fore.GREEN}{Style.BRIGHT}\nNormalizing and stretching the data...{Style.RESET_ALL}")
    red_norm = normalize(red_data, stretch=red_stretch,
                         min_percent=red_min_percent, max_percent=red_max_percent)
    green_norm = normalize(green_data, stretch=green_stretch,
                           min_percent=green_min_percent, max_percent=green_max_percent)
    blue_norm = normalize(blue_data, stretch=blue_stretch,
                          min_percent=blue_min_percent, max_percent=blue_max_percent)

    # Apply scaling factors
    print(f"{Fore.CYAN}{Style.BRIGHT}\nEnter scaling factors for each channel (e.g., 1.0 for no change):{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Scaling factors adjust the brightness of each color channel.{Style.RESET_ALL}")
    try:
        red_scale = float(input(f"{Fore.RED}{Style.BRIGHT}Red channel scaling factor [default 1.0]:\n> {Style.RESET_ALL}").strip() or 1.0)
        green_scale = float(input(f"{Fore.GREEN}{Style.BRIGHT}Green channel scaling factor [default 1.0]:\n> {Style.RESET_ALL}").strip() or 1.0)
        blue_scale = float(input(f"{Fore.BLUE}{Style.BRIGHT}Blue channel scaling factor [default 1.0]:\n> {Style.RESET_ALL}").strip() or 1.0)
    except ValueError:
        print(f"{Fore.RED}Invalid input. Please enter numeric values.")
        sys.exit()

    red_norm *= red_scale
    green_norm *= green_scale
    blue_norm *= blue_scale

    # Ensure the data is in the range 0-1 after scaling
    red_norm = np.clip(red_norm, 0, 1)
    green_norm = np.clip(green_norm, 0, 1)
    blue_norm = np.clip(blue_norm, 0, 1)

    # Ask if the user wants to upscale the image
    print(f"{Fore.CYAN}{Style.BRIGHT}\nDo you want to upscale the image? This increases the number of pixels.{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Upscaling can enhance details but may increase processing time.{Style.RESET_ALL}")
    upscale_choice = input(f"{Fore.CYAN}{Style.BRIGHT}Enter 'yes' to upscale or 'no' to keep original resolution:\n> {Style.RESET_ALL}").strip().lower()

    MAX_ZOOM_FACTOR = 2  # Set maximum allowed zoom factor to prevent excessive upscaling

    if upscale_choice == 'yes':
        try:
            zoom_factor = float(input(f"{Fore.CYAN}{Style.BRIGHT}Enter the zoom factor (e.g., 2 for doubling the resolution):\n> {Style.RESET_ALL}"))
            if zoom_factor > MAX_ZOOM_FACTOR:
                print(f"{Fore.YELLOW}Zoom factor too high. Limiting to {MAX_ZOOM_FACTOR}.")
                zoom_factor = MAX_ZOOM_FACTOR
            # Upscale the normalized data
            order = 3  # Cubic interpolation for better quality
            print(f"{Fore.GREEN}{Style.BRIGHT}Upscaling images...{Style.RESET_ALL}")
            red_norm = zoom(red_norm, zoom_factor, order=order)
            green_norm = zoom(green_norm, zoom_factor, order=order)
            blue_norm = zoom(blue_norm, zoom_factor, order=order)
        except ValueError:
            print(f"{Fore.YELLOW}Invalid zoom factor. Proceeding without upscaling.")
            zoom_factor = 1
    else:
        zoom_factor = 1

    # Ask if the user wants to apply denoising
    print(f"{Fore.CYAN}{Style.BRIGHT}\nDo you want to apply denoising to the images?{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Denoising can reduce noise but may smooth out fine details.{Style.RESET_ALL}")
    denoise_choice = input(f"{Fore.CYAN}{Style.BRIGHT}Enter 'yes' to denoise or 'no' to skip:\n> {Style.RESET_ALL}").strip().lower()

    if denoise_choice == 'yes':
        print(f"{Fore.CYAN}{Style.BRIGHT}\nChoose the denoising level (low, medium, high):{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Higher levels remove more noise but may reduce detail.{Style.RESET_ALL}")
        denoise_level = input(f"{Fore.CYAN}{Style.BRIGHT}Enter the denoising level:\n> {Style.RESET_ALL}").strip().lower()
        # Apply denoising to each channel
        print(f"{Fore.GREEN}{Style.BRIGHT}Denoising red channel...{Style.RESET_ALL}")
        red_norm = denoise_data(red_norm, level=denoise_level)
        print(f"{Fore.GREEN}{Style.BRIGHT}Denoising green channel...{Style.RESET_ALL}")
        green_norm = denoise_data(green_norm, level=denoise_level)
        print(f"{Fore.GREEN}{Style.BRIGHT}Denoising blue channel...{Style.RESET_ALL}")
        blue_norm = denoise_data(blue_norm, level=denoise_level)
    else:
        print(f"{Fore.GREEN}Skipping denoising.")
        denoise_level = None  # Not used, but defined for completeness

    # Combine channels into RGB image
    print(f"{Fore.GREEN}{Style.BRIGHT}\nCombining channels into an RGB image...{Style.RESET_ALL}")
    rgb_image = np.dstack((red_norm, green_norm, blue_norm))

    # Adjust image size based on user's choice
    print(f"{Fore.CYAN}{Style.BRIGHT}\nChoose the size of the output image:{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Options: small, medium, large{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Larger sizes have higher resolution but may take longer to process.{Style.RESET_ALL}")
    size_options = ['small', 'medium', 'large']
    size_choice = input(f"{Fore.CYAN}{Style.BRIGHT}Enter your choice [default 'medium']:\n> {Style.RESET_ALL}").strip().lower() or 'medium'

    if size_choice not in size_options:
        print(f"{Fore.YELLOW}Invalid size option selected. Defaulting to 'medium'.")
        size_choice = 'medium'

    # Resize image according to size choice, limiting maximum dimensions
    size_factors = {'small': 1, 'medium': 1.5, 'large': 2}
    size_factor = size_factors.get(size_choice, 1.5)
    max_dimension = 8000  # Limit maximum dimension to prevent memory errors

    new_height = int(rgb_image.shape[0] * size_factor)
    new_width = int(rgb_image.shape[1] * size_factor)

    # Limit dimensions to max_dimension
    if new_height > max_dimension or new_width > max_dimension:
        print(f"{Fore.YELLOW}Image size too large. Limiting dimensions to {max_dimension}px.")
        scaling_factor = max_dimension / max(rgb_image.shape[0], rgb_image.shape[1])
        new_height = int(rgb_image.shape[0] * scaling_factor)
        new_width = int(rgb_image.shape[1] * scaling_factor)
        size_factor = scaling_factor

    # Resize the image
    print(f"{Fore.GREEN}{Style.BRIGHT}\nResizing image to '{size_choice}' size...{Style.RESET_ALL}")
    rgb_image = zoom(rgb_image, (size_factor, size_factor, 1), order=3)

    # Ensure the data is in the range 0-1 after resizing
    rgb_image = np.clip(rgb_image, 0, 1)

    # Display the image
    print(f"{Fore.GREEN}{Style.BRIGHT}\nDisplaying the combined image...{Style.RESET_ALL}")
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_image, origin='lower')
    plt.axis('off')
    plt.show()

    # Automatically set the output folder to the folder containing the FITS files
    output_folder = folder_path

    # Generate the output filename with resolution
    height, width, _ = rgb_image.shape
    resolution_str = f"{width}x{height}"
    output_filename = f"combined_image_{resolution_str}.png"

    # Full path to the output file
    output_file = os.path.join(output_folder, output_filename)

    # Save the image using imsave to preserve resolution
    print(f"{Fore.GREEN}{Style.BRIGHT}\nSaving the image to {output_file}...{Style.RESET_ALL}")
    try:
        plt.imsave(output_file, rgb_image, origin='lower')
        print(f"{Fore.GREEN}Image saved as {output_file}")
    except Exception as e:
        print(f"{Fore.RED}Error saving image: {e}")

    # Clear unused variables to free memory
    del red_data, green_data, blue_data
    del red_norm, green_norm, blue_norm
    gc.collect()

if __name__ == "__main__":
    main()
