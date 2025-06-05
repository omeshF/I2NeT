#version working with variable column counts
import numpy as np
from PIL import Image
import struct
import os
import json
import csv

# Configuration
TYPES_FILE = "data_types.json"
OUTPUT_CSV = "from_image.csv"

def load_type_information():
    try:
        if os.path.exists(TYPES_FILE):
            with open(TYPES_FILE, 'r') as f:
                type_info = json.load(f)
            return type_info
        else:
            print(f"  Type file '{TYPES_FILE}' not found. Using fallback detection.")
            return None
    except Exception as e:
        print(f" Error loading type information: {e}")
        return None

def detect_data_structure_from_rgb_count(rgb_count, type_info=None):
    
    # Each float uses 2 RGB pixels
    float_count = rgb_count // 2
    
    if type_info and 'original_types' in type_info:
        # Try to match the available data to the expected structure
        original_types = type_info['original_types']
        expected_floats = calculate_expected_float_count(original_types)       
        
        if float_count < expected_floats:
            # Truncate the expected types to match available data
            truncated_types = truncate_types_to_match_data(original_types, float_count)
            return {
                'original_types': truncated_types,
                'final_types': ['Float'] * float_count,
                'truncated': True,
                'available_floats': float_count,
                'expected_floats': expected_floats
            }
        elif float_count > expected_floats:
            return type_info
        else:
            return type_info
    else:
        # No type info available - create generic structure
        return {
            'original_types': ['Float'] * float_count,
            'final_types': ['Float'] * float_count,
            'generic': True,
            'available_floats': float_count
        }

def calculate_expected_float_count(original_types):
    float_count = 0
    for data_type in original_types:
        if data_type == "IP Address":
            float_count += 4  # 4 octets
        elif data_type == "MAC Address":
            float_count += 2  # 2 chunks of 3 bytes each
        else:  # Integer, Float, String
            float_count += 1
    return float_count

def truncate_types_to_match_data(original_types, available_floats):
    truncated_types = []
    float_count = 0
    
    for data_type in original_types:
        if data_type == "IP Address":
            if float_count + 4 <= available_floats:
                truncated_types.append(data_type)
                float_count += 4
            else:
                # Partial IP - treat remaining floats as individual integers
                remaining = available_floats - float_count
                truncated_types.extend(['Integer'] * remaining)
                break
        elif data_type == "MAC Address":
            if float_count + 2 <= available_floats:
                truncated_types.append(data_type)
                float_count += 2
            else:
                # Partial MAC - treat remaining floats as individual integers
                remaining = available_floats - float_count
                truncated_types.extend(['Integer'] * remaining)
                break
        else:  # Integer, Float, String
            if float_count + 1 <= available_floats:
                truncated_types.append(data_type)
                float_count += 1
            else:
                break
    
    return truncated_types

def two_rgb_pixels_to_float(rgb_pixel1, rgb_pixel2):
    try:
        # Extract bytes from RGB pixels
        r1, g1, b1 = rgb_pixel1
        r2, g2, b2 = rgb_pixel2
        
        # Reconstruct the 4-byte float representation
        packed_bytes = bytes([r1, g1, b1, r2])
        
        # Unpack as float
        float_val = struct.unpack('!f', packed_bytes)[0]
        return float_val
    except Exception as e:
        print(f"Float conversion error: {e}")
        return 0.0

def extract_rgb_from_image(image_path):
    try:
        img = Image.open(image_path)
        array = np.array(img)
        print(f"Image shape: {array.shape}")
        
        # Get image dimensions
        height, width = array.shape[:2]
        
        # Detect stripes by sampling rows and finding color changes
        stripe_colors = []
        last_color = None
        
        # Sample every few rows to detect color changes
        sample_step = max(1, height // 100)  # Sample more frequently for better detection
        
        for row in range(0, height, sample_step):
            current_color = tuple(array[row, 0])  # Sample first pixel of row
            
            if last_color is None or current_color != last_color:
                stripe_colors.append(current_color)
                last_color = current_color
        
        return stripe_colors
        
    except Exception as e:
        print(f" Error extracting RGB from {image_path}: {e}")
        return []

def decode_rgb_to_floats(rgb_values):
    decoded_values = []
    i = 0
    
    
    while i + 1 < len(rgb_values):
        float_val = two_rgb_pixels_to_float(rgb_values[i], rgb_values[i + 1])
        decoded_values.append(float_val)
        i += 2
    
    # Handle odd number of RGB values
    if i < len(rgb_values):
        print(f"  Odd number of RGB values, last one ignored")
    return decoded_values

def reconstruct_with_adaptive_types(decoded_values, adaptive_type_info):
    if not adaptive_type_info or 'original_types' not in adaptive_type_info:
        return decoded_values
    
    original_types = adaptive_type_info['original_types']
    reconstructed = []
    val_idx = 0
    
    
    for orig_idx, orig_type in enumerate(original_types):
        if val_idx >= len(decoded_values):
            break
            
        try:
            if orig_type == "IP Address":
                # Reconstruct IP from 4 float values
                if val_idx + 3 < len(decoded_values):
                    octet1 = max(0, min(255, int(round(decoded_values[val_idx]))))
                    octet2 = max(0, min(255, int(round(decoded_values[val_idx + 1]))))
                    octet3 = max(0, min(255, int(round(decoded_values[val_idx + 2]))))
                    octet4 = max(0, min(255, int(round(decoded_values[val_idx + 3]))))
                    
                    ip = f"{octet1}.{octet2}.{octet3}.{octet4}"
                    reconstructed.append(ip)
                    print(f"  Reconstructed IP: {ip}")
                    val_idx += 4
                else:
                    # Not enough values for complete IP
                    remaining = len(decoded_values) - val_idx
                    print(f"  ⚠️  Incomplete IP data, treating {remaining} values as integers")
                    for _ in range(remaining):
                        if val_idx < len(decoded_values):
                            int_val = int(round(decoded_values[val_idx]))
                            reconstructed.append(int_val)
                            val_idx += 1
                    break
                    
            elif orig_type == "MAC Address":
                # Reconstruct MAC from 2 float values
                if val_idx + 1 < len(decoded_values):
                    mac1 = int(round(decoded_values[val_idx])) & 0xFFFFFF
                    mac2 = int(round(decoded_values[val_idx + 1])) & 0xFFFFFF
                    
                    # Convert to MAC format
                    mac1_hex = format(mac1, '06x')
                    mac2_hex = format(mac2, '06x')
                    
                    mac = f"{mac1_hex[:2]}:{mac1_hex[2:4]}:{mac1_hex[4:6]}:{mac2_hex[:2]}:{mac2_hex[2:4]}:{mac2_hex[4:6]}"
                    reconstructed.append(mac)
                    print(f"  Reconstructed MAC: {mac}")
                    val_idx += 2
                else:
                    # Not enough values for complete MAC
                    remaining = len(decoded_values) - val_idx
                    print(f"  ⚠️  Incomplete MAC data, treating {remaining} values as integers")
                    for _ in range(remaining):
                        if val_idx < len(decoded_values):
                            int_val = int(round(decoded_values[val_idx]))
                            reconstructed.append(int_val)
                            val_idx += 1
                    break
                    
            elif orig_type == "Float":
                # Keep as float
                reconstructed.append(decoded_values[val_idx])
                print(f"  Float: {decoded_values[val_idx]}")
                val_idx += 1
                
            elif orig_type == "Integer":
                # Convert to integer
                int_val = int(round(decoded_values[val_idx]))
                reconstructed.append(int_val)
                print(f"  Integer: {int_val}")
                val_idx += 1
                
            else:  # String or unknown
                # For strings, show hash value
                hash_val = int(round(decoded_values[val_idx]))
                reconstructed.append(f"str_{hash_val}")
                print(f"  String hash: str_{hash_val}")
                val_idx += 1
                
        except Exception as e:
            print(f" Error reconstructing {orig_type}: {e}")
            reconstructed.append("error")
            val_idx += 1
    
    # Handle any remaining values
    while val_idx < len(decoded_values):
        remaining_val = decoded_values[val_idx]
        reconstructed.append(remaining_val)
        print(f"  Extra float: {remaining_val}")
        val_idx += 1
    
    return reconstructed

def translateSingleImage(image_path, type_info=None):
    print(f"Processing image: {image_path}")
    
    try:
        # Extract RGB values from image
        rgb_values = extract_rgb_from_image(image_path)
        
        if not rgb_values:
            print(f"No RGB values extracted from {image_path}")
            return []
        
        # Detect actual data structure based on RGB count
        adaptive_type_info = detect_data_structure_from_rgb_count(len(rgb_values), type_info)
        
        # Decode RGB to floats
        decoded_values = decode_rgb_to_floats(rgb_values)
        
        # Reconstruct using adaptive types
        reconstructed = reconstruct_with_adaptive_types(decoded_values, adaptive_type_info)
        
        # Report results. Uncomment to help with debugging. 
        #if adaptive_type_info.get('truncated'):
        #    print(f"⚠️  Data was truncated: {adaptive_type_info['available_floats']}/{adaptive_type_info['expected_floats']} columns")
        #elif adaptive_type_info.get('generic'):
        #    print(f"  Using generic float structure for {adaptive_type_info['available_floats']} columns")
        #else:
        #    print(f" Full data structure reconstructed")
        
        #print(f" Final result: {reconstructed}")
        return reconstructed
        
    except Exception as e:
        print(f" Error processing image {image_path}: {e}")
        return []

def translateMultipleImages(images_dir, type_info=None):
    
    if not os.path.exists(images_dir):
        print(f"Directory {images_dir} does not exist!")
        return
    
    try:
        with open(OUTPUT_CSV, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Get and sort image files
            images = os.listdir(images_dir)
            image_files = [img for img in images if img.endswith('.png')]
            image_files = sorted(image_files, key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0)
            
            
            for image in image_files:
                image_path = os.path.join(images_dir, image)
                #print(f"\n{'='*50}")
                print(f"Processing: {image}")
                
                try:
                    reconstructed_line = translateSingleImage(image_path, type_info)
                    
                    if reconstructed_line:  # Only write if we got data
                        writer.writerow(reconstructed_line)
                    else:
                        print(f" No data extracted from {image}")
                        
                except Exception as e:
                    print(f" Error processing {image}: {e}")
            
            print(f"\n CSV file '{OUTPUT_CSV}' created successfully!")
            
    except Exception as e:
        print(f" Error creating CSV file: {e}")

# Main execution
if __name__ == "__main__":
    # Load type information
    type_info = load_type_information()
    
    # uncomment to help with debugging
    # Test with single image first
    test_image = "data/0.png"
    if os.path.exists(test_image):
        #print(f"\n{'='*30} SINGLE IMAGE TEST {'='*30}")
        result = translateSingleImage(test_image, type_info)
        #print(f"Single image result: {result}")
        #print(f"Columns detected: {len(result)}")
    else:
        print(f"Test image {test_image} not found")
    
    # Process all images
    data_dir = "data"
    if os.path.exists(data_dir):
        translateMultipleImages(data_dir, type_info)
    else:
        print(f"Data directory {data_dir} not found!")
        print("Available files/directories:")
        for item in os.listdir('.'):
            print(f"  - {item}")

# Direct function call
if __name__ == "__main__":
    type_info = load_type_information()
    translateMultipleImages("data", type_info)