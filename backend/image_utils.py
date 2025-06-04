# image_utils.py - Enhanced Version with OpenAI Integration
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import logging
import base64
import json
import os
import requests
from typing import Dict, Any, Tuple, Optional, Union

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration Constants
class Config:
    SSIM_DEFAULT_WINDOW_SIZE = 7
    SSIM_MIN_WINDOW_SIZE = 3
    OPENAI_MAX_TOKENS = 800
    OPENAI_TEMPERATURE = 0.4
    OPENAI_DEFAULT_MODEL = "gpt-4o"
    HISTOGRAM_BINS = [8, 8, 8]
    HISTOGRAM_RANGES = [0, 180, 0, 256, 0, 256]
    OPENAI_API_KEY = "your_api_key_here"
# Custom Exceptions
class ImageSimilarityError(Exception):
    """Custom exception for image similarity calculation errors."""
    pass

class OpenAIAPIError(Exception):
    """Custom exception for OpenAI API errors."""
    pass

# Utility Functions
def validate_image_input(image: np.ndarray, name: str = "image") -> None:
    """Validate that the input is a proper image array."""
    if image is None:
        raise ValueError(f"{name} is None")
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be numpy array, got {type(image)}")
    if image.size == 0:
        raise ValueError(f"{name} is empty")
    if len(image.shape) not in [2, 3]:
        raise ValueError(f"{name} must be 2D or 3D array, got shape {image.shape}")

def is_grayscale_or_single_channel(image: np.ndarray) -> bool:
    """Check if image is grayscale or single channel."""
    return len(image.shape) < 3 or (len(image.shape) == 3 and image.shape[2] == 1)

def ensure_compatible_shapes(imageA: np.ndarray, imageB: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure two images have compatible shapes by resizing if necessary."""
    if imageA.shape != imageB.shape:
        logging.warning(f"Image shapes differ: {imageA.shape} vs {imageB.shape}. Resizing second image.")
        try:
            imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))
        except Exception as e:
            raise ImageSimilarityError(f"Error resizing image: {e}")
    return imageA, imageB

def get_openai_api_key() -> str:
    """Get OpenAI API key from config or environment variable."""
    api_key = Config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIAPIError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or Config.OPENAI_API_KEY")
    return api_key

# --- CV-based Similarity Functions ---
def calculate_mse(imageA: np.ndarray, imageB: np.ndarray) -> float:
    """
    Calculate the Mean Squared Error (MSE) between two images.
    Lower MSE indicates higher similarity.
    
    Args:
        imageA, imageB: Input images as numpy arrays
        
    Returns:
        float: MSE value (lower = more similar)
        
    Raises:
        ImageSimilarityError: If calculation fails
    """
    try:
        validate_image_input(imageA, "imageA")
        validate_image_input(imageB, "imageB")
        
        imageA, imageB = ensure_compatible_shapes(imageA, imageB)
        
        # Convert to float for calculation
        imageA = imageA.astype("float64")
        imageB = imageB.astype("float64")
        
        err = np.sum((imageA - imageB) ** 2)
        
        # Calculate total number of pixels
        num_pixels = imageA.shape[0] * imageA.shape[1]
        if len(imageA.shape) == 3:
            num_pixels *= imageA.shape[2]
        
        mse = err / float(num_pixels)
        logging.info(f"Calculated MSE: {mse}")
        return mse
        
    except Exception as e:
        logging.error(f"Error calculating MSE: {e}")
        raise ImageSimilarityError(f"MSE calculation failed: {e}")

def calculate_ssim(imageA: np.ndarray, imageB: np.ndarray) -> float:
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    SSIM values are between -1 and 1, where 1 is perfect similarity.
    
    Args:
        imageA, imageB: Input images as numpy arrays
        
    Returns:
        float: SSIM value (1 = identical, -1 = completely different)
        
    Raises:
        ImageSimilarityError: If calculation fails
    """
    try:
        validate_image_input(imageA, "imageA")
        validate_image_input(imageB, "imageB")
        
        imageA, imageB = ensure_compatible_shapes(imageA, imageB)
        
        # Convert to grayscale if needed
        if len(imageA.shape) == 3:
            grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        else:
            grayA = imageA.copy()
            
        if len(imageB.shape) == 3:
            grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        else:
            grayB = imageB.copy()
        
        # Ensure proper data type
        if grayA.dtype != np.uint8:
            grayA = np.clip(grayA, 0, 255).astype(np.uint8)
        if grayB.dtype != np.uint8:
            grayB = np.clip(grayB, 0, 255).astype(np.uint8)
        
        # Handle small images
        min_dim = min(grayA.shape[0], grayA.shape[1])
        if min_dim < Config.SSIM_MIN_WINDOW_SIZE:
            logging.warning(f"Image too small for SSIM (min_dim: {min_dim})")
            # For very small images, use simple pixel comparison
            if np.array_equal(grayA, grayB):
                return 1.0
            else:
                return 0.0
        
        # Calculate appropriate window size (must be odd)
        win_size = min(Config.SSIM_DEFAULT_WINDOW_SIZE, min_dim)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < Config.SSIM_MIN_WINDOW_SIZE:
            win_size = Config.SSIM_MIN_WINDOW_SIZE
        
        # Calculate SSIM
        score, _ = ssim(grayA, grayB, full=True, data_range=255.0, win_size=win_size)
        logging.info(f"Calculated SSIM: {score}")
        return float(score)
        
    except Exception as e:
        logging.error(f"Error calculating SSIM: {e}")
        raise ImageSimilarityError(f"SSIM calculation failed: {e}")

def compare_histograms(imageA: np.ndarray, imageB: np.ndarray, 
                      method: int = cv2.HISTCMP_CORREL) -> float:
    """
    Compare the histograms of two images using a specified method.
    
    Args:
        imageA, imageB: Input images as numpy arrays
        method: OpenCV histogram comparison method
        
    Returns:
        float: Similarity score (interpretation depends on method)
        
    Raises:
        ImageSimilarityError: If calculation fails
    """
    try:
        validate_image_input(imageA, "imageA")
        validate_image_input(imageB, "imageB")
        
        imageA, imageB = ensure_compatible_shapes(imageA, imageB)
        
        # Convert to BGR if grayscale for HSV conversion
        if is_grayscale_or_single_channel(imageA):
            imageA = cv2.cvtColor(imageA, cv2.COLOR_GRAY2BGR)
        if is_grayscale_or_single_channel(imageB):
            imageB = cv2.cvtColor(imageB, cv2.COLOR_GRAY2BGR)
        
        # Convert to HSV
        hsvA = cv2.cvtColor(imageA, cv2.COLOR_BGR2HSV)
        hsvB = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms
        histA = cv2.calcHist([hsvA], [0, 1, 2], None, Config.HISTOGRAM_BINS, Config.HISTOGRAM_RANGES)
        histB = cv2.calcHist([hsvB], [0, 1, 2], None, Config.HISTOGRAM_BINS, Config.HISTOGRAM_RANGES)
        
        # Normalize histograms
        cv2.normalize(histA, histA, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(histB, histB, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        # Compare histograms
        similarity = cv2.compareHist(histA, histB, method)
        logging.info(f"Calculated Histogram Comparison (method {method}): {similarity}")
        return float(similarity)
        
    except Exception as e:
        logging.error(f"Error comparing histograms: {e}")
        raise ImageSimilarityError(f"Histogram comparison failed: {e}")

# --- LLM-based Semantic Similarity ---
def call_openai_api(payload: Dict[str, Any], timeout: int = 30) -> Dict[str, Any]:
    """
    Make actual API call to OpenAI.
    
    Args:
        payload: Request payload for OpenAI API
        timeout: Request timeout in seconds
        
    Returns:
        Dict containing API response or error information
    """
    try:
        api_key = get_openai_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "response": response.json()
            }
        else:
            return {
                "success": False,
                "error": f"API call failed with status {response.status_code}: {response.text}",
                "status_code": response.status_code
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"Request timed out after {timeout} seconds"
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }

def get_llm_semantic_comparison(image_cv_1: np.ndarray, image_cv_2: np.ndarray, 
                               image1_filename: str = "image1.png", 
                               image2_filename: str = "image2.png",
                               model: str = Config.OPENAI_DEFAULT_MODEL,
                               make_api_call: bool = False) -> Dict[str, Any]:
    """
    Compare two images using OpenAI's multimodal LLM.
    
    Args:
        image_cv_1, image_cv_2: OpenCV image objects (NumPy arrays)
        image1_filename, image2_filename: Original filenames for context
        model: OpenAI model to use
        make_api_call: If True, makes actual API call; if False, returns request payload
        
    Returns:
        Dict containing API response, request details, or error information
    """
    try:
        validate_image_input(image_cv_1, "image_cv_1")
        validate_image_input(image_cv_2, "image_cv_2")
        
        # Encode images to base64
        _, buffer1 = cv2.imencode('.png', image_cv_1)
        base64_image1 = base64.b64encode(buffer1).decode('utf-8')
        
        _, buffer2 = cv2.imencode('.png', image_cv_2)
        base64_image2 = base64.b64encode(buffer2).decode('utf-8')
        
        logging.info("Images successfully encoded to base64 for OpenAI LLM.")
        
        # Construct prompt
        prompt = (
            f"You are an expert image analyst. Compare the two provided images, "
            f"'{image1_filename}' and '{image2_filename}'. "
            "Describe their key visual similarities and differences. "
            "Focus on content, style, objects, and overall composition. "
            "After your description, provide a conceptual similarity score on a new line in the format: "
            "'Conceptual Similarity Score: X/10', where X is a number from 0 (completely different) "
            "to 10 (conceptually identical). Be concise but informative."
        )
        
        # OpenAI API payload
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image1}"}
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image2}"}
                        }
                    ]
                }
            ],
            "max_tokens": Config.OPENAI_MAX_TOKENS,
            "temperature": Config.OPENAI_TEMPERATURE
        }
        
        if make_api_call:
            # Make actual API call
            logging.info("Making OpenAI API call...")
            api_response = call_openai_api(payload)
            
            if api_response["success"]:
                # Extract response text
                response_content = api_response["response"]["choices"][0]["message"]["content"]
                parsed_response = parse_llm_response(response_content)
                
                return {
                    "status": "completed",
                    "api_response": api_response["response"],
                    "parsed_response": parsed_response,
                    "usage": api_response["response"].get("usage", {})
                }
            else:
                return {
                    "status": "api_error",
                    "error": api_response["error"],
                    "payload_for_retry": payload
                }
        else:
            # Return request details for external execution
            return {
                "status": "pending_openai_api_call",
                "payload_for_api": payload,
                "api_url_for_api": "https://api.openai.com/v1/chat/completions",
                "comment": "LLM call to be executed by the calling service using OpenAI API."
            }
        
    except Exception as e:
        logging.error(f"Error in LLM comparison: {e}")
        return {"error": f"Failed to prepare/execute LLM comparison: {str(e)}"}

def get_all_similarity_scores(source_image_cv: np.ndarray, target_image_cv: np.ndarray, 
                             source_filename: str = "source.png", 
                             target_filename: str = "target.png",
                             include_llm_call: bool = False) -> Dict[str, Any]:
    """
    Calculate all similarity scores between two images.
    
    Args:
        source_image_cv, target_image_cv: OpenCV image objects
        source_filename, target_filename: Original filenames for context
        include_llm_call: If True, makes actual OpenAI API call
        
    Returns:
        Dict containing CV scores and LLM results/request details
    """
    result = {
        "cv_scores": {},
        "llm_comparison": {}
    }
    
    try:
        validate_image_input(source_image_cv, "source_image_cv")
        validate_image_input(target_image_cv, "target_image_cv")
        
        # Calculate CV-based scores
        cv_scores = {}
        try:
            cv_scores['mse'] = calculate_mse(source_image_cv, target_image_cv)
            cv_scores['ssim'] = calculate_ssim(source_image_cv, target_image_cv)
            cv_scores['histogram_correlation'] = compare_histograms(
                source_image_cv, target_image_cv, method=cv2.HISTCMP_CORREL
            )
            cv_scores['histogram_chi_squared'] = compare_histograms(
                source_image_cv, target_image_cv, method=cv2.HISTCMP_CHISQR
            )
            result["cv_scores"] = cv_scores
            
        except ImageSimilarityError as e:
            logging.error(f"Error calculating CV scores: {e}")
            result["cv_scores"] = {"error": str(e)}
        
        # Handle LLM comparison
        try:
            llm_result = get_llm_semantic_comparison(
                source_image_cv, target_image_cv, 
                source_filename, target_filename,
                make_api_call=include_llm_call
            )
            result["llm_comparison"] = llm_result
            
        except Exception as e:
            logging.error(f"Error in LLM comparison: {e}")
            result["llm_comparison"] = {"error": str(e)}
    
    except Exception as e:
        logging.error(f"Error in get_all_similarity_scores: {e}")
        result["cv_scores"] = {"error": f"Input validation failed: {str(e)}"}
        result["llm_comparison"] = {"error": f"Input validation failed: {str(e)}"}
    
    return result

# Utility function for parsing LLM responses
def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM response to extract description and similarity score.
    
    Args:
        response_text: Raw response text from LLM
        
    Returns:
        Dict containing parsed description and score
    """
    try:
        lines = response_text.strip().split('\n')
        score_line_prefix = "Conceptual Similarity Score:"
        
        description_lines = []
        parsed_score_str = "N/A"
        parsed_score_val = 0
        
        for line in lines:
            line = line.strip()
            if line.startswith(score_line_prefix):
                parsed_score_str = line.replace(score_line_prefix, "").strip()
                try:
                    # Extract number before '/'
                    score_num_str = parsed_score_str.split('/')[0].strip()
                    parsed_score_val = int(score_num_str)
                    parsed_score_val = max(0, min(10, parsed_score_val))  # Clamp to 0-10
                except (ValueError, IndexError):
                    logging.warning(f"Could not parse LLM score from: {parsed_score_str}")
                    parsed_score_val = 0
            else:
                if line:  # Skip empty lines
                    description_lines.append(line)
        
        parsed_description = " ".join(description_lines).strip()
        
        return {
            "description": parsed_description,
            "conceptual_score_str": parsed_score_str,
            "conceptual_score_value": parsed_score_val,
            "raw_llm_output": response_text
        }
        
    except Exception as e:
        logging.error(f"Error parsing LLM response: {e}")
        return {
            "error": f"Failed to parse LLM response: {str(e)}",
            "raw_llm_output": response_text
        }

if __name__ == "__main__":
    logging.info("Running image_utils.py directly for testing.")
    
    # Create test images
    try:
        # Black square
        source_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite("test_source.png", source_img)
        
        # Black square with white rectangle
        target_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(target_img, (10, 10), (30, 30), (255, 255, 255), -1)
        cv2.imwrite("test_target.png", target_img)
        
        # Load images
        img_source = cv2.imread("test_source.png")
        img_target = cv2.imread("test_target.png")
        
        if img_source is None or img_target is None:
            raise ValueError("Failed to load test images")
        
        print("Testing image similarity calculations...")
        
        # Test without OpenAI API call (default behavior)
        print("\n=== Testing without OpenAI API call ===")
        result = get_all_similarity_scores(img_source, img_target, "test_source.png", "test_target.png")
        
        print("\nCV Scores:")
        cv_scores = result["cv_scores"]
        if "error" in cv_scores:
            print(f"  Error: {cv_scores['error']}")
        else:
            for metric, score in cv_scores.items():
                print(f"  {metric}: {score:.6f}")
        
        print("\nLLM Request Status:")
        llm_result = result["llm_comparison"]
        if "error" in llm_result:
            print(f"  Error: {llm_result['error']}")
        else:
            print(f"  Status: {llm_result.get('status', 'Unknown')}")
            if 'payload_for_api' in llm_result:
                print(f"  Model: {llm_result['payload_for_api'].get('model', 'Unknown')}")
        
        # Test with OpenAI API call (only if API key is available)
        print("\n=== Testing with OpenAI API call ===")
        try:
            get_openai_api_key()  # This will raise an exception if no API key
            print("OpenAI API key found. Making API call...")
            
            result_with_api = get_all_similarity_scores(
                img_source, img_target, 
                "test_source.png", "test_target.png",
                include_llm_call=True
            )
            
            llm_api_result = result_with_api["llm_comparison"]
            if llm_api_result.get("status") == "completed":
                parsed = llm_api_result["parsed_response"]
                print(f"  LLM Description: {parsed['description'][:100]}...")
                print(f"  Conceptual Score: {parsed['conceptual_score_str']}")
                print(f"  Usage: {llm_api_result.get('usage', {})}")
            elif llm_api_result.get("status") == "api_error":
                print(f"  API Error: {llm_api_result['error']}")
            else:
                print(f"  Unexpected status: {llm_api_result}")
                
        except OpenAIAPIError as e:
            print(f"  Skipping API test: {e}")
        except Exception as e:
            print(f"  API test failed: {e}")
        
        # Clean up test files
        for filename in ["test_source.png", "test_target.png"]:
            if os.path.exists(filename):
                os.remove(filename)
                
        print("\nTest completed successfully!")
        
    except Exception as e:
        logging.error(f"Test failed: {e}")
        print(f"Test failed: {e}")