"""
I2NeT Decoder Module with IPv4/IPv6 Support and Simple Interface
Main module for decoding data from images to CSV format.
Images generated from NeT2I is supported
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
import ipaddress


class EnhancedI2NeT_Decoder:
    
    def __init__(self, 
                 types_file_ipv4: str = "data_types.json",
                 types_file_ipv6: str = "data_types_ipv6.json"):
        self.types_file_ipv4 = types_file_ipv4
        self.types_file_ipv6 = types_file_ipv6
        self.ipv4_type_info = None
        self.ipv6_type_info = None
        self.processed_images = 0
        self.total_images = 0
        
    def load_type_information(self, is_ipv6: bool = False) -> Optional[Dict[str, Any]]:
        types_file = self.types_file_ipv6 if is_ipv6 else self.types_file_ipv4
        type_info_attr = 'ipv6_type_info' if is_ipv6 else 'ipv4_type_info'
        
        try:
            if os.path.exists(types_file):
                with open(types_file, 'r') as f:
                    type_info = json.load(f)
                setattr(self, type_info_attr, type_info)
                #print(f"Type information loaded from '{types_file}' ({'IPv6' if is_ipv6 else 'IPv4'})")
                return type_info
            else:
                print(f"Type file '{types_file}' not found. Using fallback detection.")
                return None
        except Exception as e:
            print(f"Error loading type information from {types_file}: {e}")
            return None
    
    def detect_data_structure_from_rgb_count(self, rgb_count: int, type_info: Optional[Dict] = None) -> Dict[str, Any]:
        
        if type_info and 'original_types' in type_info:
            # Try to match the available data to the expected structure
            original_types = type_info['original_types']
            expected_pixels = self._calculate_expected_pixel_count(original_types)
            
            if rgb_count < expected_pixels:
                # Truncate the expected types to match available data
                truncated_types = self._truncate_types_to_match_data(original_types, rgb_count)
                return {
                    'original_types': truncated_types,
                    'final_types': self._get_final_types_from_original(truncated_types),
                    'truncated': True,
                    'available_pixels': rgb_count,
                    'expected_pixels': expected_pixels
                }
            elif rgb_count >= expected_pixels:
                return type_info
        else:
            # No type info available - assume 2 pixels per float
            float_count = rgb_count // 2
            return {
                'original_types': ['Float'] * float_count,
                'final_types': ['Float'] * float_count,
                'generic': True,
                'available_pixels': rgb_count
            }
    
    def _calculate_expected_pixel_count(self, original_types: List[str]) -> int:
        
        pixel_count = 0
        for data_type in original_types:
            if data_type == "IPv6 Address":
                pixel_count += 6  # IPv6 uses 6 RGB pixels
            elif data_type == "IPv4 Address":
                pixel_count += 8  # IPv4 uses 8 RGB pixels (4 octets × 2 pixels each)
            elif data_type == "MAC Address":
                pixel_count += 4  # MAC uses 4 RGB pixels (2 chunks × 2 pixels each)
            else:  # Integer, Float, String
                pixel_count += 2  # Standard types use 2 RGB pixels
        return pixel_count
    
    def _get_final_types_from_original(self, original_types: List[str]) -> List[str]:

        final_types = []
        for orig_type in original_types:
            if orig_type == "IPv4 Address":
                final_types.extend(["IPv4 Address"] * 4)  # 4 octets
            elif orig_type == "MAC Address":
                final_types.extend(["MAC Address"] * 2)  # 2 chunks
            else:
                final_types.append(orig_type)  # IPv6, Float, Integer, String unchanged
        return final_types
    
    def _truncate_types_to_match_data(self, original_types: List[str], available_pixels: int) -> List[str]:

        truncated_types = []
        pixel_count = 0
        
        for data_type in original_types:
            if data_type == "IPv6 Address":
                if pixel_count + 6 <= available_pixels:
                    truncated_types.append(data_type)
                    pixel_count += 6
                else:
                    # Partial IPv6 - treat remaining pixels as pairs for floats
                    remaining = available_pixels - pixel_count
                    float_count = remaining // 2
                    truncated_types.extend(['Float'] * float_count)
                    break
            elif data_type == "IPv4 Address":
                if pixel_count + 8 <= available_pixels:
                    truncated_types.append(data_type)
                    pixel_count += 8
                else:
                    # Partial IPv4 - treat remaining pixels as pairs for floats
                    remaining = available_pixels - pixel_count
                    float_count = remaining // 2
                    truncated_types.extend(['Float'] * float_count)
                    break
            elif data_type == "MAC Address":
                if pixel_count + 4 <= available_pixels:
                    truncated_types.append(data_type)
                    pixel_count += 4
                else:
                    # Partial MAC - treat remaining pixels as pairs for floats
                    remaining = available_pixels - pixel_count
                    float_count = remaining // 2
                    truncated_types.extend(['Float'] * float_count)
                    break
            else:  # Integer, Float, String
                if pixel_count + 2 <= available_pixels:
                    truncated_types.append(data_type)
                    pixel_count += 2
                else:
                    break
        
        return truncated_types
    
    def _two_rgb_pixels_to_float(self, rgb_pixel1: Tuple[int, int, int], rgb_pixel2: Tuple[int, int, int]) -> float:

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
    
    def _six_rgb_pixels_to_ipv6(self, rgb_pixels: List[Tuple[int, int, int]]) -> str:

        try:
            # Extract 18 bytes from 6 RGB pixels
            data_bytes = []
            for r, g, b in rgb_pixels:
                data_bytes.extend([r, g, b])
            
            # Take first 16 bytes for IPv6 (ignore 2 padding bytes)
            ipv6_bytes = bytes(data_bytes[:16])
            
            # Create IPv6 address
            ipv6_addr = ipaddress.IPv6Address(ipv6_bytes)
            return str(ipv6_addr)
        except Exception as e:
            print(f"IPv6 conversion error: {e}")
            return "::1"  # Fallback to localhost
    
    def _extract_rgb_from_image(self, image_path: str) -> List[Tuple[int, int, int]]:
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
    
    def _reconstruct_with_adaptive_types(self, rgb_values: List[Tuple[int, int, int]], adaptive_type_info: Dict[str, Any]) -> List[Any]:

        if not adaptive_type_info or 'original_types' not in adaptive_type_info:
            # Fallback: decode as floats (2 pixels per float)
            decoded_values = []
            i = 0
            while i + 1 < len(rgb_values):
                float_val = self._two_rgb_pixels_to_float(rgb_values[i], rgb_values[i + 1])
                decoded_values.append(float_val)
                i += 2
            return decoded_values
        
        original_types = adaptive_type_info['original_types']
        reconstructed = []
        pixel_idx = 0
        
        for orig_type in original_types:
            if pixel_idx >= len(rgb_values):
                break
                
            try:
                if orig_type == "IPv6 Address":
                    # Reconstruct IPv6 from 6 RGB pixels
                    if pixel_idx + 5 < len(rgb_values):
                        ipv6_pixels = rgb_values[pixel_idx:pixel_idx + 6]
                        ipv6_addr = self._six_rgb_pixels_to_ipv6(ipv6_pixels)
                        reconstructed.append(ipv6_addr)
                        pixel_idx += 6
                    else:
                        # Not enough pixels for complete IPv6
                        remaining = len(rgb_values) - pixel_idx
                        print(f"  Incomplete IPv6 data, treating {remaining} pixels as floats")
                        while pixel_idx + 1 < len(rgb_values):
                            float_val = self._two_rgb_pixels_to_float(rgb_values[pixel_idx], rgb_values[pixel_idx + 1])
                            reconstructed.append(float_val)
                            pixel_idx += 2
                        break
                        
                elif orig_type == "IPv4 Address":
                    # Reconstruct IPv4 from 8 RGB pixels (4 octets × 2 pixels each)
                    if pixel_idx + 7 < len(rgb_values):
                        octets = []
                        for i in range(4):
                            octet_float = self._two_rgb_pixels_to_float(
                                rgb_values[pixel_idx + i*2], 
                                rgb_values[pixel_idx + i*2 + 1]
                            )
                            octet = max(0, min(255, int(round(octet_float))))
                            octets.append(str(octet))
                        
                        ipv4_addr = ".".join(octets)
                        reconstructed.append(ipv4_addr)
                        pixel_idx += 8
                    else:
                        # Not enough pixels for complete IPv4
                        remaining = len(rgb_values) - pixel_idx
                        print(f"  Incomplete IPv4 data, treating {remaining} pixels as floats")
                        while pixel_idx + 1 < len(rgb_values):
                            float_val = self._two_rgb_pixels_to_float(rgb_values[pixel_idx], rgb_values[pixel_idx + 1])
                            reconstructed.append(float_val)
                            pixel_idx += 2
                        break
                        
                elif orig_type == "MAC Address":
                    # Reconstruct MAC from 4 RGB pixels (2 chunks × 2 pixels each)
                    if pixel_idx + 3 < len(rgb_values):
                        chunk1_float = self._two_rgb_pixels_to_float(rgb_values[pixel_idx], rgb_values[pixel_idx + 1])
                        chunk2_float = self._two_rgb_pixels_to_float(rgb_values[pixel_idx + 2], rgb_values[pixel_idx + 3])
                        
                        chunk1 = int(round(chunk1_float)) & 0xFFFFFF
                        chunk2 = int(round(chunk2_float)) & 0xFFFFFF
                        
                        chunk1_hex = format(chunk1, '06x')
                        chunk2_hex = format(chunk2, '06x')
                        
                        mac = f"{chunk1_hex[:2]}:{chunk1_hex[2:4]}:{chunk1_hex[4:6]}:{chunk2_hex[:2]}:{chunk2_hex[2:4]}:{chunk2_hex[4:6]}"
                        reconstructed.append(mac)
                        pixel_idx += 4
                    else:
                        # Not enough pixels for complete MAC
                        remaining = len(rgb_values) - pixel_idx
                        print(f"  Incomplete MAC data, treating {remaining} pixels as floats")
                        while pixel_idx + 1 < len(rgb_values):
                            float_val = self._two_rgb_pixels_to_float(rgb_values[pixel_idx], rgb_values[pixel_idx + 1])
                            reconstructed.append(float_val)
                            pixel_idx += 2
                        break
                        
                elif orig_type == "Float":
                    # Keep as float (2 pixels)
                    if pixel_idx + 1 < len(rgb_values):
                        float_val = self._two_rgb_pixels_to_float(rgb_values[pixel_idx], rgb_values[pixel_idx + 1])
                        reconstructed.append(float_val)
                        pixel_idx += 2
                    else:
                        break
                        
                elif orig_type == "Integer":
                    # Convert to integer (2 pixels)
                    if pixel_idx + 1 < len(rgb_values):
                        float_val = self._two_rgb_pixels_to_float(rgb_values[pixel_idx], rgb_values[pixel_idx + 1])
                        int_val = int(round(float_val))
                        reconstructed.append(int_val)
                        pixel_idx += 2
                    else:
                        break
                        
                else:  # String or unknown
                    # For strings, show hash value (2 pixels)
                    if pixel_idx + 1 < len(rgb_values):
                        float_val = self._two_rgb_pixels_to_float(rgb_values[pixel_idx], rgb_values[pixel_idx + 1])
                        hash_val = int(round(float_val))
                        reconstructed.append(f"str_{hash_val}")
                        pixel_idx += 2
                    else:
                        break
                        
            except Exception as e:
                print(f"Error reconstructing {orig_type}: {e}")
                reconstructed.append("error")
                # Skip appropriate number of pixels
                if orig_type == "IPv6 Address":
                    pixel_idx += 6
                elif orig_type == "IPv4 Address":
                    pixel_idx += 8
                elif orig_type == "MAC Address":
                    pixel_idx += 4
                else:
                    pixel_idx += 2
        
        return reconstructed
    
    def decode_single_image(self, image_path: str, is_ipv6: bool = None) -> List[Any]:
        try:
            # Auto-detect IP version from filename if not specified
            if is_ipv6 is None:
                filename = os.path.basename(image_path)
                is_ipv6 = filename.startswith('ipv6_')
            
            # Load appropriate type information
            type_info = self.ipv6_type_info if is_ipv6 else self.ipv4_type_info
            if type_info is None:
                type_info = self.load_type_information(is_ipv6)
            
            # Extract RGB values from image
            rgb_values = self._extract_rgb_from_image(image_path)
            
            if not rgb_values:
                print(f"No RGB values extracted from {image_path}")
                return []
            
            # Detect actual data structure based on RGB count
            adaptive_type_info = self.detect_data_structure_from_rgb_count(len(rgb_values), type_info)
            
            # Reconstruct using adaptive types
            reconstructed = self._reconstruct_with_adaptive_types(rgb_values, adaptive_type_info)
            
            return reconstructed
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return []
    
    def _get_sorted_image_files(self, data_directory: str, prefix: str = None) -> List[str]:

        # Support multiple image formats
        image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        image_files = []
        
        for pattern in image_patterns:
            files = glob.glob(os.path.join(data_directory, pattern))
            if prefix:
                files = [f for f in files if os.path.basename(f).startswith(prefix)]
            image_files.extend(files)
        
        # Sort by numeric value if filename is numeric, otherwise alphabetically
        def sort_key(filepath):
            filename = os.path.basename(filepath)
            if prefix:
                name_without_prefix = filename[len(prefix):]
            else:
                name_without_prefix = filename
            name_without_ext = os.path.splitext(name_without_prefix)[0]
            if name_without_ext.isdigit():
                return int(name_without_ext)
            else:
                return filename
        
        return sorted(image_files, key=sort_key)
    
    def load_data(self, data_directory: str, verbose: bool = True) -> Dict[str, Any]:

        if not os.path.exists(data_directory):
            raise FileNotFoundError(f"Directory {data_directory} does not exist!")
        
        # Load both type information files
        self.load_type_information(is_ipv6=False)  # Load IPv4 types
        self.load_type_information(is_ipv6=True)   # Load IPv6 types
        
        results = {
            "input_directory": data_directory,
            "ipv4_results": None,
            "ipv6_results": None,
            "total_images_processed": 0
        }
        
        # Process IPv4 images
        ipv4_files = self._get_sorted_image_files(data_directory, 'ipv4_')
        if ipv4_files:
            if verbose:
                print(f"Found {len(ipv4_files)} IPv4 images")
            ipv4_results = self._process_image_set(ipv4_files, 'decoded_ipv4.csv', is_ipv6=False, verbose=verbose)
            results["ipv4_results"] = ipv4_results
            results["total_images_processed"] += ipv4_results["processed_images"]
        
        # Process IPv6 images
        ipv6_files = self._get_sorted_image_files(data_directory, 'ipv6_')
        if ipv6_files:
            if verbose:
                print(f"Found {len(ipv6_files)} IPv6 images")
            ipv6_results = self._process_image_set(ipv6_files, 'decoded_ipv6.csv', is_ipv6=True, verbose=verbose)
            results["ipv6_results"] = ipv6_results
            results["total_images_processed"] += ipv6_results["processed_images"]
        
        # Process any remaining images without prefix
        other_files = []
        all_files = self._get_sorted_image_files(data_directory)
        for f in all_files:
            filename = os.path.basename(f)
            if not filename.startswith('ipv4_') and not filename.startswith('ipv6_'):
                other_files.append(f)
        
        if other_files:
            if verbose:
                print(f"Found {len(other_files)} images without prefix, processing as IPv4")
            other_results = self._process_image_set(other_files, 'decoded_other.csv', is_ipv6=False, verbose=verbose)
            results["other_results"] = other_results
            results["total_images_processed"] += other_results["processed_images"]
        
        if verbose:
            print(f"\n Decoding completed!")
            #print(f"Total images processed: {results['total_images_processed']}")
            #if results["ipv4_results"]:
            #    print(f"IPv4 CSV: decoded_ipv4.csv ({results['ipv4_results']['successful_rows']} rows)")
            #if results["ipv6_results"]:
            #    print(f"IPv6 CSV: decoded_ipv6.csv ({results['ipv6_results']['successful_rows']} rows)")
            #if results.get("other_results"):
            #    print(f"Other CSV: decoded_other.csv ({results['other_results']['successful_rows']} rows)")
        
        return results
    
    def _process_image_set(self, image_files: List[str], output_csv: str, is_ipv6: bool, verbose: bool) -> Dict[str, Any]:

        processed_images = 0
        successful_rows = 0
        
        try:
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                for i, image_path in enumerate(image_files):
                    try:
                        reconstructed_line = self.decode_single_image(image_path, is_ipv6)
                        
                        if reconstructed_line:  # Only write if we got data
                            writer.writerow(reconstructed_line)
                            successful_rows += 1
                        else:
                            if verbose:
                                print(f"No data extracted from {os.path.basename(image_path)}")
                        
                        processed_images += 1
                        
                        # Progress indicator
                        if verbose and (i + 1) % 10 == 0:
                            print(f"Processed {i + 1}/{len(image_files)} {'IPv6' if is_ipv6 else 'IPv4'} images...")
                            
                    except Exception as e:
                        if verbose:
                            print(f"Error processing {os.path.basename(image_path)}: {e}")
                        processed_images += 1
                
                success_rate = (successful_rows / len(image_files)) * 100 if len(image_files) > 0 else 0
                
                return {
                    "output_file": output_csv,
                    "ip_version": "IPv6" if is_ipv6 else "IPv4",
                    "total_images": len(image_files),
                    "processed_images": processed_images,
                    "successful_rows": successful_rows,
                    "success_rate": success_rate,
                    "type_info": self.ipv6_type_info if is_ipv6 else self.ipv4_type_info
                }
                
        except Exception as e:
            error_msg = f"Error creating CSV file {output_csv}: {e}"
            print(error_msg)
            raise RuntimeError(error_msg)




def decode(data_directory: str, 
          output_csv: str,
          types_file_ipv4: str = "data_types.json",
          types_file_ipv6: str = "data_types_ipv6.json",
          verbose: bool = True) -> Dict[str, Any]:
    
    if verbose:
        print(" Starting I2NeT decoding...")
    
    # Use the enhanced decoder to process images
    results = load_data_with_ipv4_ipv6_support(
        data_directory=data_directory,
        types_file_ipv4=types_file_ipv4,
        types_file_ipv6=types_file_ipv6,
        verbose=verbose
    )
    
    # Merge the separate CSV files into one
    total_rows = _merge_csv_files(results, output_csv, verbose)
    
    # Clean up temporary files
    _cleanup_temp_files(results, verbose)
    
    # Prepare final results
    final_results = {
        "input_directory": data_directory,
        "output_file": output_csv,
        "total_images_processed": results["total_images_processed"],
        "total_rows": total_rows,
        "ipv4_rows": results["ipv4_results"]["successful_rows"] if results.get("ipv4_results") else 0,
        "ipv6_rows": results["ipv6_results"]["successful_rows"] if results.get("ipv6_results") else 0,
        "other_rows": results["other_results"]["successful_rows"] if results.get("other_results") else 0,
        "success": total_rows > 0
    }
    
    if verbose:
        #print(f"\nDecoding completed!")
        #print(f"📁 Input directory: {data_directory}")
        print(f" Output file: {output_csv}")
        #print(f"🖼️  Total images processed: {final_results['total_images_processed']}")
        #print(f"📊 Total rows decoded: {final_results['total_rows']}")
        #if final_results['ipv4_rows'] > 0:
        #    print(f"   └─ IPv4 rows: {final_results['ipv4_rows']}")
        #if final_results['ipv6_rows'] > 0:
        #    print(f"   └─ IPv6 rows: {final_results['ipv6_rows']}")
        #if final_results['other_rows'] > 0:
        #    print(f"   └─ Other rows: {final_results['other_rows']}")
    
    return final_results



def load_data_with_ipv4_ipv6_support(data_directory: str,
                                    types_file_ipv4: str = "data_types.json",
                                    types_file_ipv6: str = "data_types_ipv6.json",
                                    verbose: bool = True) -> Dict[str, Any]:

    decoder = EnhancedI2NeT_Decoder(types_file_ipv4, types_file_ipv6)
    return decoder.load_data(data_directory, verbose)


def _merge_csv_files(results: Dict[str, Any], output_csv: str, verbose: bool) -> int:

    total_rows = 0
    
    try:
        with open(output_csv, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            
            # Process IPv4 results
            if results.get("ipv4_results") and os.path.exists("decoded_ipv4.csv"):
                #if verbose:
                #    print("📄 Merging IPv4 data...")
                with open("decoded_ipv4.csv", 'r') as infile:
                    reader = csv.reader(infile)
                    for row in reader:
                        writer.writerow(row)
                        total_rows += 1
            
            # Process IPv6 results
            if results.get("ipv6_results") and os.path.exists("decoded_ipv6.csv"):
                #if verbose:
                #    print("📄 Merging IPv6 data...")
                with open("decoded_ipv6.csv", 'r') as infile:
                    reader = csv.reader(infile)
                    for row in reader:
                        writer.writerow(row)
                        total_rows += 1
            
            # Process other results
            if results.get("other_results") and os.path.exists("decoded_other.csv"):
                #if verbose:
                #    print("📄 Merging other data...")
                with open("decoded_other.csv", 'r') as infile:
                    reader = csv.reader(infile)
                    for row in reader:
                        writer.writerow(row)
                        total_rows += 1
        
        #if verbose:
        #    print(f"📊 Merged {total_rows} total rows into {output_csv}")
        
        return total_rows
        
    except Exception as e:
        print(f"Error merging CSV files: {e}")
        return 0


def _cleanup_temp_files(results: Dict[str, Any], verbose: bool) -> None:

    temp_files = []
    
    if results.get("ipv4_results"):
        temp_files.append("decoded_ipv4.csv")
    if results.get("ipv6_results"):
        temp_files.append("decoded_ipv6.csv")
    if results.get("other_results"):
        temp_files.append("decoded_other.csv")
    
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                #if verbose:
                #    print(f"🗑️  Cleaned up temporary file: {temp_file}")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not remove temporary file {temp_file}: {e}")



def decode_single(image_path: str, 
                 types_file: str = "data_types.json",
                 is_ipv6: bool = None,
                 verbose: bool = False) -> List[Any]:

    # Determine IP version and appropriate types file
    if is_ipv6 is None:
        filename = os.path.basename(image_path)
        is_ipv6 = filename.startswith('ipv6_')
    
    # Use appropriate types file
    if is_ipv6 and types_file == "data_types.json":
        types_file = "data_types_ipv6.json"
    
    decoder = EnhancedI2NeT_Decoder(
        types_file_ipv4="data_types.json" if not is_ipv6 else types_file,
        types_file_ipv6="data_types_ipv6.json" if is_ipv6 else types_file
    )
    
    return decoder.decode_single_image(image_path, is_ipv6)


def get_decoder(types_file_ipv4: str = "data_types.json",
               types_file_ipv6: str = "data_types_ipv6.json") -> EnhancedI2NeT_Decoder:

    return EnhancedI2NeT_Decoder(types_file_ipv4, types_file_ipv6)



__all__ = [
    'decode',
    'decode_single', 
    'get_decoder',
    'load_data_with_ipv4_ipv6_support',
    'EnhancedI2NeT_Decoder'
]

