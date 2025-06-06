"""
I2NeT Decoder Module
Main module for decoding data from images to CSV format.
"""

import numpy as np
from PIL import Image
import struct
import os
import json
import csv
from typing import List, Dict, Any, Optional, Tuple
import glob
import re


class I2NeT_Decoder:
    """Main decoder class for image to CSV transformation."""
    
    def __init__(self, types_file: str = "data_types.json"):
        """
        Initialize the I2NeT decoder.
        
        Args:
            types_file: Path to the JSON file containing type information
        """
        self.types_file = types_file
        self.type_info = None
        self.processed_images = 0
        self.total_images = 0
        
    def load_type_information(self) -> Optional[Dict[str, Any]]:
        """
        Load data type information from JSON file.
        
        Returns:
            Dictionary containing type information or None if not found
        """
        try:
            if os.path.exists(self.types_file):
                with open(self.types_file, 'r') as f:
                    self.type_info = json.load(f)
                #print(f"Type information loaded from '{self.types_file}'")
                return self.type_info
            else:
                print(f"Type file '{self.types_file}' not found. Using fallback detection.")
                return None
        except Exception as e:
            print(f"Error loading type information: {e}")
            return None
    
    def detect_data_structure_from_rgb_count(self, rgb_count: int, type_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Detect data structure based on RGB pixel count.
        
        Args:
            rgb_count: Number of RGB pixels available
            type_info: Optional type information dictionary
            
        Returns:
            Dictionary containing detected structure information
        """
        # Each float uses 2 RGB pixels
        float_count = rgb_count // 2
        
        if type_info and 'original_types' in type_info:
            # Try to match the available data to the expected structure
            original_types = type_info['original_types']
            expected_floats = self._calculate_expected_float_count(original_types)
            
            if float_count < expected_floats:
                # Truncate the expected types to match available data
                truncated_types = self._truncate_types_to_match_data(original_types, float_count)
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
    
    def _calculate_expected_float_count(self, original_types: List[str]) -> int:
        """
        Calculate expected number of floats based on data types.
        
        Args:
            original_types: List of original data types
            
        Returns:
            Expected number of float values
        """
        float_count = 0
        for data_type in original_types:
            if data_type == "IP Address":
                float_count += 4  # 4 octets
            elif data_type == "MAC Address":
                float_count += 2  # 2 chunks of 3 bytes each
            else:  # Integer, Float, String
                float_count += 1
        return float_count
    
    def _truncate_types_to_match_data(self, original_types: List[str], available_floats: int) -> List[str]:
        """
        Truncate type list to match available data.
        
        Args:
            original_types: List of original data types
            available_floats: Number of available float values
            
        Returns:
            Truncated list of data types
        """
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
    
    def _two_rgb_pixels_to_float(self, rgb_pixel1: Tuple[int, int, int], rgb_pixel2: Tuple[int, int, int]) -> float:
        """
        Convert two RGB pixels to a single float value.
        
        Args:
            rgb_pixel1: First RGB pixel as (r, g, b) tuple
            rgb_pixel2: Second RGB pixel as (r, g, b) tuple
            
        Returns:
            Reconstructed float value
        """
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
    
    def _extract_rgb_from_image(self, image_path: str) -> List[Tuple[int, int, int]]:
        """
        Extract RGB values from image stripes.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of RGB tuples representing the stripes
        """
        try:
            img = Image.open(image_path)
            array = np.array(img)
            
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
            print(f"Error extracting RGB from {image_path}: {e}")
            return []
    
    def _decode_rgb_to_floats(self, rgb_values: List[Tuple[int, int, int]]) -> List[float]:
        """
        Decode RGB values to float values.
        
        Args:
            rgb_values: List of RGB tuples
            
        Returns:
            List of decoded float values
        """
        decoded_values = []
        i = 0
        
        while i + 1 < len(rgb_values):
            float_val = self._two_rgb_pixels_to_float(rgb_values[i], rgb_values[i + 1])
            decoded_values.append(float_val)
            i += 2
        
        # Handle odd number of RGB values
        if i < len(rgb_values):
            print(f"Warning: Odd number of RGB values, last one ignored")
            
        return decoded_values
    
    def _reconstruct_with_adaptive_types(self, decoded_values: List[float], adaptive_type_info: Dict[str, Any]) -> List[Any]:
        """
        Reconstruct data using adaptive type information.
        
        Args:
            decoded_values: List of decoded float values
            adaptive_type_info: Dictionary containing type adaptation information
            
        Returns:
            List of reconstructed values with appropriate types
        """
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
                        val_idx += 4
                    else:
                        # Not enough values for complete IP
                        remaining = len(decoded_values) - val_idx
                        print(f"  Incomplete IP data, treating {remaining} values as integers")
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
                        val_idx += 2
                    else:
                        # Not enough values for complete MAC
                        remaining = len(decoded_values) - val_idx
                        print(f"  Incomplete MAC data, treating {remaining} values as integers")
                        for _ in range(remaining):
                            if val_idx < len(decoded_values):
                                int_val = int(round(decoded_values[val_idx]))
                                reconstructed.append(int_val)
                                val_idx += 1
                        break
                        
                elif orig_type == "Float":
                    # Keep as float
                    reconstructed.append(decoded_values[val_idx])
                    val_idx += 1
                    
                elif orig_type == "Integer":
                    # Convert to integer
                    int_val = int(round(decoded_values[val_idx]))
                    reconstructed.append(int_val)
                    val_idx += 1
                    
                else:  # String or unknown
                    # For strings, show hash value
                    hash_val = int(round(decoded_values[val_idx]))
                    reconstructed.append(f"str_{hash_val}")
                    val_idx += 1
                    
            except Exception as e:
                print(f"Error reconstructing {orig_type}: {e}")
                reconstructed.append("error")
                val_idx += 1
        
        # Handle any remaining values
        while val_idx < len(decoded_values):
            remaining_val = decoded_values[val_idx]
            reconstructed.append(remaining_val)
            val_idx += 1
        
        return reconstructed
    
    def decode_single_image(self, image_path: str) -> List[Any]:
        """
        Decode a single image to extract data.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of decoded values
        """
        try:
            # Extract RGB values from image
            rgb_values = self._extract_rgb_from_image(image_path)
            
            if not rgb_values:
                print(f"No RGB values extracted from {image_path}")
                return []
            
            # Detect actual data structure based on RGB count
            adaptive_type_info = self.detect_data_structure_from_rgb_count(len(rgb_values), self.type_info)
            
            # Decode RGB to floats
            decoded_values = self._decode_rgb_to_floats(rgb_values)
            
            # Reconstruct using adaptive types
            reconstructed = self._reconstruct_with_adaptive_types(decoded_values, adaptive_type_info)
            
            return reconstructed
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return []
    
    def _get_sorted_image_files(self, data_directory: str) -> List[str]:
        """
        Get sorted list of image files from directory.
        
        Args:
            data_directory: Directory containing image files
            
        Returns:
            Sorted list of image file paths
        """
        # Support multiple image formats
        image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        image_files = []
        
        for pattern in image_patterns:
            image_files.extend(glob.glob(os.path.join(data_directory, pattern)))
        
        # Sort by numeric value if filename is numeric, otherwise alphabetically
        def sort_key(filepath):
            filename = os.path.basename(filepath)
            name_without_ext = os.path.splitext(filename)[0]
            if name_without_ext.isdigit():
                return int(name_without_ext)
            else:
                return filename
        
        return sorted(image_files, key=sort_key)
    
    def load_data(self, data_directory: str, output_csv: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Main function to load data from images and save to CSV.
        
        Args:
            data_directory: Directory containing the image files
            output_csv: Path for the output CSV file
            verbose: Whether to print progress information
            
        Returns:
            Dictionary with decoding results and metadata
        """
        if not os.path.exists(data_directory):
            raise FileNotFoundError(f"Directory {data_directory} does not exist!")
        
        # Load type information
        self.load_type_information()
        
        # Get sorted image files
        image_files = self._get_sorted_image_files(data_directory)
        
        if not image_files:
            print(f"No image files found in {data_directory}")
            return {
                "input_directory": data_directory,
                "output_file": output_csv,
                "total_images": 0,
                "processed_images": 0,
                "success_rate": 0.0,
                "type_info": self.type_info
            }
        
        self.total_images = len(image_files)
        self.processed_images = 0
        successful_rows = 0
        
        if verbose:
            print(f"Found {self.total_images} image files in {data_directory}")
            #print(f"Decoding images to {output_csv}...")
        
        try:
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                for i, image_path in enumerate(image_files):
                    try:
                        reconstructed_line = self.decode_single_image(image_path)
                        
                        if reconstructed_line:  # Only write if we got data
                            writer.writerow(reconstructed_line)
                            successful_rows += 1
                        else:
                            if verbose:
                                print(f"No data extracted from {os.path.basename(image_path)}")
                        
                        self.processed_images += 1
                        
                        # Progress indicator
                        if verbose and (i + 1) % 10 == 0:
                            print(f"Processed {i + 1}/{self.total_images} images...")
                            
                    except Exception as e:
                        if verbose:
                            print(f"Error processing {os.path.basename(image_path)}: {e}")
                        self.processed_images += 1
                
                success_rate = (successful_rows / self.total_images) * 100 if self.total_images > 0 else 0
                
                if verbose:
                    print(f"\nDecoding completed!")
                    print(f"- CSV file '{output_csv}' created successfully")
                    #print(f"- Processed {self.processed_images}/{self.total_images} images")
                    #print(f"- Successfully decoded {successful_rows} rows")
                    #print(f"- Success rate: {success_rate:.1f}%")
                
                return {
                    "input_directory": data_directory,
                    "output_file": output_csv,
                    "total_images": self.total_images,
                    "processed_images": self.processed_images,
                    "successful_rows": successful_rows,
                    "success_rate": success_rate,
                    "type_info": self.type_info
                }
                
        except Exception as e:
            error_msg = f"Error creating CSV file: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)


# Convenience functions for direct use
_default_decoder = None

def load_data(data_directory: str, 
              output_csv: str, 
              types_file: str = "data_types.json",
              verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function to load data from images and save to CSV.
    
    Args:
        data_directory: Directory containing the image files
        output_csv: Path for the output CSV file
        types_file: Path to the JSON file containing type information
        verbose: Whether to print progress information
        
    Returns:
        Dictionary with decoding results and metadata
        
    Example:
        >>> import I2NeT.decoder as decoder
        >>> results = decoder.load_data('data', 'decoded_images.csv')
        >>> print(f"Decoded {results['successful_rows']} rows")
    """
    global _default_decoder
    
    _default_decoder = I2NeT_Decoder(types_file=types_file)
    return _default_decoder.load_data(data_directory, output_csv, verbose)


def decode_single_image(image_path: str, types_file: str = "data_types.json") -> List[Any]:
    """
    Convenience function to decode a single image.
    
    Args:
        image_path: Path to the image file
        types_file: Path to the JSON file containing type information
        
    Returns:
        List of decoded values
        
    Example:
        >>> import I2NeT.decoder as decoder
        >>> values = decoder.decode_single_image('data/0.png')
        >>> print(f"Decoded {len(values)} values")
    """
    decoder = I2NeT_Decoder(types_file=types_file)
    decoder.load_type_information()
    return decoder.decode_single_image(image_path)


def get_decoder() -> Optional[I2NeT_Decoder]:
    """Get the current default decoder instance."""
    return _default_decoder


# Legacy function names for backward compatibility
def translateSingleImage(image_path: str, type_info: Optional[Dict] = None) -> List[Any]:
    """Legacy function - use decode_single_image instead."""
    return decode_single_image(image_path)


def translateMultipleImages(images_dir: str, type_info: Optional[Dict] = None) -> None:
    """Legacy function - use load_data instead."""
    output_csv = "from_image.csv"
    load_data(images_dir, output_csv)


# Aliases for different naming conventions
def images_to_csv(data_directory: str, output_csv: str, **kwargs) -> Dict[str, Any]:
    """Alias for load_data function."""
    return load_data(data_directory, output_csv, **kwargs)


def decode_images(data_directory: str, output_csv: str, **kwargs) -> Dict[str, Any]:
    """Alias for load_data function."""
    return load_data(data_directory, output_csv, **kwargs)