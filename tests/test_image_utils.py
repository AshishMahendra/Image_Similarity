#!/usr/bin/env python3
"""
Comprehensive test suite for image_utils.py
Tests all functions with various scenarios including edge cases.
"""

import unittest
import numpy as np
import cv2
import os
import tempfile
import logging
from unittest.mock import patch, MagicMock
import sys

# Import the module to test
try:
    from image_utils import (
        calculate_mse, calculate_ssim, compare_histograms,
        get_llm_semantic_comparison, get_all_similarity_scores,
        parse_llm_response, validate_image_input, 
        is_grayscale_or_single_channel, ensure_compatible_shapes,
        ImageSimilarityError, Config
    )
except ImportError as e:
    print(f"Error importing image_utils: {e}")
    print("Make sure image_utils.py is in the same directory or Python path")
    sys.exit(1)

class TestImageUtils(unittest.TestCase):
    """Test cases for image similarity functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test images
        self.identical_img1 = np.ones((50, 50, 3), dtype=np.uint8) * 128
        self.identical_img2 = np.ones((50, 50, 3), dtype=np.uint8) * 128
        
        self.different_img1 = np.zeros((50, 50, 3), dtype=np.uint8)
        self.different_img2 = np.ones((50, 50, 3), dtype=np.uint8) * 255
        
        self.similar_img1 = np.ones((50, 50, 3), dtype=np.uint8) * 100
        self.similar_img2 = np.ones((50, 50, 3), dtype=np.uint8) * 110
        
        # Small images for edge case testing
        self.tiny_img1 = np.ones((2, 2, 3), dtype=np.uint8) * 128
        self.tiny_img2 = np.ones((2, 2, 3), dtype=np.uint8) * 128
        
        # Grayscale images
        self.gray_img1 = np.ones((50, 50), dtype=np.uint8) * 128
        self.gray_img2 = np.ones((50, 50), dtype=np.uint8) * 128
        
        # Different sized images
        self.large_img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        self.small_img = np.ones((25, 25, 3), dtype=np.uint8) * 128
        
        # Temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Suppress logging during tests
        logging.getLogger().setLevel(logging.CRITICAL)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)
    
    # --- Utility Function Tests ---
    
    def test_validate_image_input_valid(self):
        """Test validate_image_input with valid inputs."""
        # Should not raise any exception
        validate_image_input(self.identical_img1)
        validate_image_input(self.gray_img1)
    
    def test_validate_image_input_invalid(self):
        """Test validate_image_input with invalid inputs."""
        with self.assertRaises(ValueError):
            validate_image_input(None)
        
        with self.assertRaises(TypeError):
            validate_image_input("not an array")
        
        with self.assertRaises(ValueError):
            validate_image_input(np.array([]))
        
        with self.assertRaises(ValueError):
            validate_image_input(np.ones((50, 50, 3, 4)))  # 4D array
    
    def test_is_grayscale_or_single_channel(self):
        """Test grayscale detection function."""
        self.assertTrue(is_grayscale_or_single_channel(self.gray_img1))
        self.assertFalse(is_grayscale_or_single_channel(self.identical_img1))
        
        # Test single channel 3D array
        single_channel = np.ones((50, 50, 1), dtype=np.uint8)
        self.assertTrue(is_grayscale_or_single_channel(single_channel))
    
    def test_ensure_compatible_shapes(self):
        """Test shape compatibility function."""
        # Same shapes - should return unchanged
        img1, img2 = ensure_compatible_shapes(self.identical_img1, self.identical_img2)
        self.assertTrue(np.array_equal(img1, self.identical_img1))
        self.assertTrue(np.array_equal(img2, self.identical_img2))
        
        # Different shapes - should resize second image
        img1, img2 = ensure_compatible_shapes(self.large_img, self.small_img)
        self.assertEqual(img1.shape, self.large_img.shape)
        self.assertEqual(img2.shape, self.large_img.shape)
    
    # --- MSE Tests ---
    
    def test_calculate_mse_identical(self):
        """Test MSE calculation with identical images."""
        mse = calculate_mse(self.identical_img1, self.identical_img2)
        self.assertEqual(mse, 0.0)
    
    def test_calculate_mse_different(self):
        """Test MSE calculation with different images."""
        mse = calculate_mse(self.different_img1, self.different_img2)
        self.assertGreater(mse, 0)
        self.assertAlmostEqual(mse, 255**2, places=2)  # Should be close to max difference
    
    def test_calculate_mse_similar(self):
        """Test MSE calculation with similar images."""
        mse = calculate_mse(self.similar_img1, self.similar_img2)
        self.assertGreater(mse, 0)
        self.assertLess(mse, 255**2)
    
    def test_calculate_mse_different_shapes(self):
        """Test MSE calculation with different shaped images."""
        mse = calculate_mse(self.large_img, self.small_img)
        self.assertIsInstance(mse, float)
        self.assertGreaterEqual(mse, 0)
    
    def test_calculate_mse_invalid_input(self):
        """Test MSE calculation with invalid inputs."""
        with self.assertRaises(ImageSimilarityError):
            calculate_mse(None, self.identical_img1)
        
        with self.assertRaises(ImageSimilarityError):
            calculate_mse(self.identical_img1, None)
    
    # --- SSIM Tests ---
    
    def test_calculate_ssim_identical(self):
        """Test SSIM calculation with identical images."""
        ssim_score = calculate_ssim(self.identical_img1, self.identical_img2)
        self.assertAlmostEqual(ssim_score, 1.0, places=5)
    
    def test_calculate_ssim_different(self):
        """Test SSIM calculation with different images."""
        ssim_score = calculate_ssim(self.different_img1, self.different_img2)
        self.assertLess(ssim_score, 1.0)
    
    def test_calculate_ssim_grayscale(self):
        """Test SSIM calculation with grayscale images."""
        ssim_score = calculate_ssim(self.gray_img1, self.gray_img2)
        self.assertAlmostEqual(ssim_score, 1.0, places=5)
    
    def test_calculate_ssim_tiny_images(self):
        """Test SSIM calculation with very small images."""
        ssim_score = calculate_ssim(self.tiny_img1, self.tiny_img2)
        # Should handle small images gracefully
        self.assertIsInstance(ssim_score, float)
        self.assertGreaterEqual(ssim_score, -1.0)
        self.assertLessEqual(ssim_score, 1.0)
    
    def test_calculate_ssim_invalid_input(self):
        """Test SSIM calculation with invalid inputs."""
        with self.assertRaises(ImageSimilarityError):
            calculate_ssim(None, self.identical_img1)
    
    # --- Histogram Tests ---
    
    def test_compare_histograms_identical(self):
        """Test histogram comparison with identical images."""
        similarity = compare_histograms(self.identical_img1, self.identical_img2, cv2.HISTCMP_CORREL)
        self.assertAlmostEqual(similarity, 1.0, places=5)
    
    def test_compare_histograms_different(self):
        """Test histogram comparison with different images."""
        similarity = compare_histograms(self.different_img1, self.different_img2, cv2.HISTCMP_CORREL)
        self.assertLess(similarity, 1.0)
    
    def test_compare_histograms_grayscale(self):
        """Test histogram comparison with grayscale images."""
        similarity = compare_histograms(self.gray_img1, self.gray_img2, cv2.HISTCMP_CORREL)
        self.assertIsInstance(similarity, float)
    
    def test_compare_histograms_chi_squared(self):
        """Test histogram comparison with chi-squared method."""
        similarity = compare_histograms(self.identical_img1, self.identical_img2, cv2.HISTCMP_CHISQR)
        self.assertAlmostEqual(similarity, 0.0, places=5)  # Chi-squared: 0 = identical
    
    def test_compare_histograms_invalid_input(self):
        """Test histogram comparison with invalid inputs."""
        with self.assertRaises(ImageSimilarityError):
            compare_histograms(None, self.identical_img1)
    
    # --- LLM Semantic Comparison Tests ---
    
    def test_get_llm_semantic_comparison_valid(self):
        """Test LLM semantic comparison with valid inputs."""
        result = get_llm_semantic_comparison(self.identical_img1, self.identical_img2)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("status"), "pending_openai_api_call")
        self.assertIn("payload_for_api", result)
        self.assertIn("api_url_for_api", result)
        
        # Check payload structure
        payload = result["payload_for_api"]
        self.assertIn("model", payload)
        self.assertIn("messages", payload)
        self.assertIn("max_tokens", payload)
        self.assertIn("temperature", payload)
        
        # Check message structure
        messages = payload["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("content", messages[0])
        
        content = messages[0]["content"]