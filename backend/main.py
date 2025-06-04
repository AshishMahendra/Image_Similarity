# main.py
import os
import shutil
import uuid
import logging
import json
from typing import Optional

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx

# Import utility functions
import image_utils 

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Image Similarity API with LLM",
    description="Compare images using computer vision metrics and OpenAI's multimodal LLM",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

@app.on_event("startup")
async def startup_event():
    logging.info("FastAPI application startup...")
    if not os.path.exists(UPLOADS_DIR):
        os.makedirs(UPLOADS_DIR)
    logging.info(f"Uploads directory is set to: {os.path.abspath(UPLOADS_DIR)}")
    
    # Check if OpenAI API key is available
    try:
        image_utils.get_openai_api_key()
        logging.info("OpenAI API key found - LLM comparisons will be available")
    except image_utils.OpenAIAPIError as e:
        logging.warning(f"OpenAI API key not found: {e}")
        logging.warning("LLM comparisons will return request payloads only")

async def make_openai_api_call(payload: dict, timeout: int = 60) -> dict:
    """
    Make asynchronous call to OpenAI API.
    
    Args:
        payload: OpenAI API request payload
        timeout: Request timeout in seconds
        
    Returns:
        Dict containing API response or error information
    """
    try:
        api_key = image_utils.get_openai_api_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            logging.info("Making OpenAI API call...")
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                api_response = response.json()
                # Extract the response content
                response_content = api_response["choices"][0]["message"]["content"]
                parsed_response = image_utils.parse_llm_response(response_content)
                
                return {
                    "success": True,
                    "status": "completed",
                    "api_response": api_response,
                    "parsed_response": parsed_response,
                    "usage": api_response.get("usage", {})
                }
            else:
                error_text = response.text
                return {
                    "success": False,
                    "status": "api_error",
                    "error": f"OpenAI API call failed with status {response.status_code}: {error_text}",
                    "status_code": response.status_code
                }
                
    except httpx.TimeoutException:
        return {
            "success": False,
            "status": "timeout_error",
            "error": f"Request timed out after {timeout} seconds"
        }
    except httpx.RequestError as e:
        return {
            "success": False,
            "status": "request_error",
            "error": f"Request failed: {str(e)}"
        }
    except image_utils.OpenAIAPIError as e:
        return {
            "success": False,
            "status": "config_error",
            "error": str(e)
        }
    except Exception as e:
        return {
            "success": False,
            "status": "unexpected_error",
            "error": f"Unexpected error: {str(e)}"
        }

@app.post("/compare-images/")
async def compare_images_endpoint(
    source_image_file: UploadFile = File(..., description="Source image file"),
    target_image_file: UploadFile = File(..., description="Target image file"),
    include_llm: bool = Query(default=True, description="Whether to include LLM-based semantic comparison"),
    llm_timeout: int = Query(default=60, ge=10, le=300, description="Timeout for LLM API call in seconds")
):
    """
    Compare two images using computer vision metrics and optionally OpenAI's multimodal LLM.
    
    Args:
        source_image_file: First image to compare
        target_image_file: Second image to compare  
        include_llm: Whether to make actual OpenAI API call for semantic comparison
        llm_timeout: Timeout for OpenAI API call in seconds (10-300)
        
    Returns:
        JSON response with CV scores and LLM comparison results
    """
    logging.info(f"Received request to /compare-images/ for {source_image_file.filename} and {target_image_file.filename}")
    logging.info(f"Include LLM: {include_llm}, LLM Timeout: {llm_timeout}s")

    source_id = str(uuid.uuid4())
    target_id = str(uuid.uuid4())
    source_ext = os.path.splitext(source_image_file.filename)[1] if source_image_file.filename else ".png"
    target_ext = os.path.splitext(target_image_file.filename)[1] if target_image_file.filename else ".png"
    source_path = os.path.join(UPLOADS_DIR, f"{source_id}{source_ext}")
    target_path = os.path.join(UPLOADS_DIR, f"{target_id}{target_ext}")

    final_response = {
        "request_info": {
            "source_filename": source_image_file.filename or "source_image",
            "target_filename": target_image_file.filename or "target_image",
            "include_llm": include_llm,
            "llm_timeout": llm_timeout
        },
        "cv_scores": {},
        "llm_comparison": {}
    }

    try:
        # Save uploaded files
        with open(source_path, "wb") as buffer:
            shutil.copyfileobj(source_image_file.file, buffer)
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(target_image_file.file, buffer)

        # Load images with OpenCV
        img_source_cv = cv2.imread(source_path)
        img_target_cv = cv2.imread(target_path)

        if img_source_cv is None:
            raise HTTPException(
                status_code=400, 
                detail=f"Could not read source image: {source_image_file.filename or 'source'}. Please ensure it's a valid image file."
            )
        if img_target_cv is None:
            raise HTTPException(
                status_code=400, 
                detail=f"Could not read target image: {target_image_file.filename or 'target'}. Please ensure it's a valid image file."
            )
        
        logging.info(f"Successfully loaded images: source {img_source_cv.shape}, target {img_target_cv.shape}")
        
        # Get all similarity scores using the enhanced image_utils
        all_scores_info = image_utils.get_all_similarity_scores(
            img_source_cv, 
            img_target_cv,
            source_filename=source_image_file.filename or "source_image",
            target_filename=target_image_file.filename or "target_image",
            include_llm_call=False  # We'll handle the API call ourselves for better control
        )
        
        # Handle CV scores
        final_response["cv_scores"] = all_scores_info.get("cv_scores", {})
        
        # Handle LLM comparison
        if include_llm:
            llm_info = all_scores_info.get("llm_comparison", {})
            
            if llm_info.get("status") == "pending_openai_api_call":
                # We have a valid request payload, make the API call
                payload = llm_info.get("payload_for_api")
                if payload:
                    logging.info("Making OpenAI API call for semantic comparison...")
                    llm_result = await make_openai_api_call(payload, timeout=llm_timeout)
                    
                    if llm_result["success"]:
                        final_response["llm_comparison"] = {
                            "status": llm_result["status"],
                            "description": llm_result["parsed_response"]["description"],
                            "conceptual_score_str": llm_result["parsed_response"]["conceptual_score_str"],
                            "conceptual_score_value": llm_result["parsed_response"]["conceptual_score_value"],
                            "usage": llm_result.get("usage", {}),
                            "model_used": payload.get("model", "unknown")
                        }
                        logging.info(f"LLM comparison completed. Score: {llm_result['parsed_response']['conceptual_score_str']}")
                    else:
                        final_response["llm_comparison"] = {
                            "status": llm_result["status"],
                            "error": llm_result["error"],
                            "request_payload_available": True
                        }
                        logging.error(f"LLM API call failed: {llm_result['error']}")
                else:
                    final_response["llm_comparison"] = {
                        "status": "payload_error",
                        "error": "Missing API payload for LLM call"
                    }
            elif "error" in llm_info:
                # Error from image_utils preparing the request
                final_response["llm_comparison"] = {
                    "status": "preparation_error",
                    "error": f"LLM request preparation failed: {llm_info['error']}"
                }
            else:
                final_response["llm_comparison"] = {
                    "status": "unknown_error",
                    "error": "Unknown state for LLM comparison request",
                    "raw_info": llm_info
                }
        else:
            # LLM comparison was disabled by request
            final_response["llm_comparison"] = {
                "status": "disabled",
                "message": "LLM comparison was disabled for this request"
            }
        
        # Log warnings for CV score errors without failing the entire request
        if 'error' in final_response.get("cv_scores", {}):
            logging.warning(f"Error in CV scores: {final_response['cv_scores']['error']}")
        
        # Add summary information
        final_response["summary"] = {
            "cv_scores_available": "error" not in final_response.get("cv_scores", {}),
            "llm_comparison_completed": final_response.get("llm_comparison", {}).get("status") == "completed",
            "total_metrics": len([k for k in final_response.get("cv_scores", {}).keys() if k != "error"])
        }
        
        logging.info("Image comparison completed successfully")
        return final_response

    except HTTPException as http_exc:
        logging.error(f"HTTP Exception in /compare-images/: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logging.error(f"An unexpected error occurred in /compare-images/: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")
    finally:
        # Clean up temporary files
        for path in [source_path, target_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logging.debug(f"Removed temporary file: {path}")
                except Exception as e_remove:
                    logging.error(f"Error removing temporary file {path}: {e_remove}")
        
        # Close uploaded file handles
        try:
            await source_image_file.close()
            await target_image_file.close()
        except Exception as e_close:
            logging.error(f"Error closing uploaded files: {e_close}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if OpenAI API key is available
        image_utils.get_openai_api_key()
        openai_status = "available"
    except image_utils.OpenAIAPIError:
        openai_status = "unavailable"
    
    return {
        "status": "healthy",
        "uploads_dir": os.path.abspath(UPLOADS_DIR),
        "uploads_dir_exists": os.path.exists(UPLOADS_DIR),
        "openai_api_key": openai_status,
        "supported_cv_metrics": ["mse", "ssim", "histogram_correlation", "histogram_chi_squared"]
    }

@app.get("/config")
async def get_config():
    """Get current configuration."""
    try:
        openai_available = True
        image_utils.get_openai_api_key()
    except image_utils.OpenAIAPIError:
        openai_available = False
    
    return {
        "openai_available": openai_available,
        "openai_model": image_utils.Config.OPENAI_DEFAULT_MODEL,
        "openai_max_tokens": image_utils.Config.OPENAI_MAX_TOKENS,
        "openai_temperature": image_utils.Config.OPENAI_TEMPERATURE,
        "uploads_directory": os.path.abspath(UPLOADS_DIR)
    }

@app.post("/test-openai/")
async def test_openai_connection():
    """Test OpenAI API connection without processing images."""
    try:
        api_key = image_utils.get_openai_api_key()
        
        # Simple test payload
        test_payload = {
            "model": image_utils.Config.OPENAI_DEFAULT_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! Please respond with 'API connection successful'."
                }
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        result = await make_openai_api_call(test_payload, timeout=30)
        
        if result["success"]:
            return {
                "status": "success",
                "message": "OpenAI API connection successful",
                "model_used": test_payload["model"],
                "response_preview": result["api_response"]["choices"][0]["message"]["content"][:100],
                "usage": result.get("usage", {})
            }
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "failed",
                    "error": result["error"],
                    "error_type": result.get("status", "unknown")
                }
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e)
            }
        )

@app.get("/")
async def root():
    """Root endpoint with API information."""
    try:
        image_utils.get_openai_api_key()
        llm_status = "✅ Available"
    except image_utils.OpenAIAPIError:
        llm_status = "❌ Unavailable (set OPENAI_API_KEY)"
    
    return {
        "message": "Welcome to the Image Similarity API with LLM Integration",
        "version": "1.0.0",
        "endpoints": {
            "compare_images": "/compare-images/",
            "health_check": "/health",
            "configuration": "/config",
            "test_openai": "/test-openai/",
            "documentation": "/docs"
        },
        "features": {
            "computer_vision_metrics": ["MSE", "SSIM", "Histogram Correlation", "Histogram Chi-Squared"],
            "llm_semantic_comparison": llm_status
        },
        "usage": "Upload two images to /compare-images/ for similarity analysis"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Set up logging
    logging.info("Starting FastAPI server with Uvicorn directly from main.py...")
    
    # Check environment
    try:
        image_utils.get_openai_api_key()
        logging.info("✅ OpenAI API key found - full functionality available")
    except image_utils.OpenAIAPIError as e:
        logging.warning(f"⚠️  OpenAI API key not found: {e}")
        logging.warning("🔧 LLM comparisons will return request payloads only")
        logging.info("💡 Set OPENAI_API_KEY environment variable to enable LLM functionality")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        reload=False  # Set to True for development
    )