# app.py
import streamlit as st
import requests
from PIL import Image
import io
import logging

# Configure basic logging for Streamlit app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - Streamlit - %(levelname)s - %(message)s')

FASTAPI_URL = "http://127.0.0.1:8000/compare-images/" # Ensure this matches your FastAPI port

st.set_page_config(layout="wide", page_title="Advanced Image Similarity Checker")

st.title("🖼️ Advanced Image Similarity Checker")
st.markdown("""
Upload a **source image** and a **target image**. We'll compare them using:
1.  **Computer Vision Metrics:** MSE, SSIM, and Histogram comparisons.
2.  **AI Semantic Analysis:** A description of similarities/differences and a conceptual score from a multimodal LLM.
""")

col1, col2 = st.columns(2)

with col1:
    st.header("Source Image")
    source_image_file = st.file_uploader("Upload Source Image", type=["png", "jpg", "jpeg", "bmp", "webp"], key="source")
    if source_image_file:
        try:
            img_source_pil = Image.open(source_image_file)
            st.image(img_source_pil, caption=f"Source: {source_image_file.name}", use_column_width="always")
        except Exception as e:
            st.error(f"Error displaying source image: {e}")
            source_image_file = None 

with col2:
    st.header("Target Image")
    target_image_file = st.file_uploader("Upload Target Image", type=["png", "jpg", "jpeg", "bmp", "webp"], key="target")
    if target_image_file:
        try:
            img_target_pil = Image.open(target_image_file)
            st.image(img_target_pil, caption=f"Target: {target_image_file.name}", use_column_width="always")
        except Exception as e:
            st.error(f"Error displaying target image: {e}")
            target_image_file = None


if st.button("🔬 Analyze Similarity", use_container_width=True, type="primary"):
    if source_image_file is not None and target_image_file is not None:
        
        progress_bar = st.progress(0, text="Starting analysis...")
        
        # Prepare files for the POST request
        files_to_upload = {
            "source_image_file": (source_image_file.name, source_image_file.getvalue(), source_image_file.type),
            "target_image_file": (target_image_file.name, target_image_file.getvalue(), target_image_file.type)
        }
        
        try:
            progress_bar.progress(10, text="📤 Uploading images to backend...")
            logging.info(f"Sending request to FastAPI: {FASTAPI_URL}")
            
            response = requests.post(FASTAPI_URL, files=files_to_upload, timeout=180) # Increased timeout for LLM
            progress_bar.progress(30, text="⚙️ Backend processing (CV metrics)...") # Approximate
            
            response.raise_for_status() 
            
            results = response.json()
            logging.info(f"Received response from FastAPI: {results}")
            progress_bar.progress(70, text="🧠 Waiting for LLM analysis (can take a moment)...")


            st.success("✅ Analysis Complete!")
            
            # --- Display CV Scores ---
            st.subheader("📊 Computer Vision Metrics")
            cv_scores = results.get("cv_scores", {})
            if 'error' in cv_scores:
                st.error(f"Error in CV Metrics: {cv_scores['error']}")
            else:
                interpretations_cv = {
                    "mse": "Lower is better (0 = identical). Pixel-wise difference.",
                    "ssim": "Closer to 1 is better (-1 to 1). Structural similarity.",
                    "histogram_correlation": "Closer to 1 is better (0 to 1). Color distribution similarity.",
                    "histogram_chi_squared": "Closer to 0 is better (0+). Color distribution dissimilarity."
                }
                cv_col1, cv_col2 = st.columns(2)
                with cv_col1:
                    mse_val = cv_scores.get('mse', float('inf'))
                    st.metric(label="Mean Squared Error (MSE)", value=f"{mse_val:.4f}" if mse_val != float('inf') else "N/A")
                    st.caption(interpretations_cv["mse"])
                    
                    hist_corr_val = cv_scores.get('histogram_correlation', 0.0)
                    st.metric(label="Histogram Correlation", value=f"{hist_corr_val:.4f}")
                    st.caption(interpretations_cv["histogram_correlation"])
                with cv_col2:
                    ssim_val = cv_scores.get('ssim', -1.0)
                    st.metric(label="Structural Similarity (SSIM)", value=f"{ssim_val:.4f}")
                    st.caption(interpretations_cv["ssim"])

                    hist_chi_val = cv_scores.get('histogram_chi_squared', float('inf'))
                    st.metric(label="Histogram Chi-Squared", value=f"{hist_chi_val:.4f}" if hist_chi_val != float('inf') else "N/A")
                    st.caption(interpretations_cv["histogram_chi_squared"])
            
            st.divider()

            # --- Display LLM Comparison ---
            st.subheader("🧠 AI Semantic Comparison (via LLM)")
            llm_comparison = results.get("llm_comparison", {})
            progress_bar.progress(100, text="Displaying results...")


            if 'error' in llm_comparison:
                st.error(f"LLM Analysis Error: {llm_comparison['error']}")
            elif not llm_comparison.get("description"): # Check if description is empty or missing
                st.warning("LLM analysis did not return a description. Raw output might be available.")
                if llm_comparison.get("raw_llm_output"):
                    with st.expander("Show Raw LLM Output"):
                        st.text_area("", llm_comparison["raw_llm_output"], height=150, disabled=True)
            else:
                st.markdown(f"**LLM's Description:**")
                st.info(llm_comparison.get("description", "No description provided."))
                
                score_str = llm_comparison.get("conceptual_score_str", "N/A")
                score_val = llm_comparison.get("conceptual_score_value", 0)

                # Display conceptual score using a custom progress bar like representation or metric
                if score_str != "N/A":
                    st.markdown(f"**Conceptual Similarity Score:** `{score_str}`")
                    try:
                        # Create a simple visual for the 0-10 score
                        # Max score is 10, so value is score_val / 10
                        st.progress(score_val / 10.0, text=f"{score_val}/10")
                    except Exception: # Catch any error if score_val is not as expected
                        st.markdown(f"(Could not visualize score: {score_str})")
                else:
                    st.markdown(f"**Conceptual Similarity Score:** Not Available")

                if llm_comparison.get("raw_llm_output"):
                    with st.expander("Show Raw LLM Output (for debugging or details)"):
                        st.text_area("", llm_comparison["raw_llm_output"], height=200, disabled=True, help="The full text response from the LLM.")
            
            progress_bar.empty() # Remove progress bar after completion

        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err} - {http_err.response.text}")
            st.error(f"API Error: {http_err.response.status_code} - {http_err.response.text}")
            progress_bar.empty()
        except requests.exceptions.ConnectionError as conn_err:
            logging.error(f"Connection error occurred: {conn_err}")
            st.error(f"Connection Error: Could not connect to the backend at {FASTAPI_URL}. Is it running?")
            progress_bar.empty()
        except requests.exceptions.Timeout as timeout_err:
            logging.error(f"Timeout error occurred: {timeout_err}")
            st.error("Request Timed Out: The backend took too long to respond. This can happen with large images or slow LLM responses.")
            progress_bar.empty()
        except Exception as e:
            logging.error(f"An unexpected error occurred in Streamlit app: {e}", exc_info=True)
            st.error(f"An unexpected error occurred: {e}")
            progress_bar.empty()

    elif source_image_file is None and target_image_file is None:
        st.warning("👈 Please upload a source image and a target image. 👆")
    elif source_image_file is None:
        st.warning("👈 Please upload a source image.")
    else: # target_image_file is None
        st.warning("👆 Please upload a target image.")

st.markdown("---")
st.caption("Powered by FastAPI, Streamlit, OpenCV, scikit-image, and a Multimodal LLM.")
