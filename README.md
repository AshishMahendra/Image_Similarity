# Image Similarity Platform

A comprehensive full-stack application for comparing images using both computer vision metrics and OpenAI's multimodal LLM for semantic analysis. The platform consists of a FastAPI backend service and an intuitive Streamlit web interface.

## 🚀 Features

### Backend API (FastAPI)
- **Computer Vision Metrics**: MSE, SSIM, Histogram Correlation, Histogram Chi-Squared
- **LLM Semantic Analysis**: OpenAI GPT-4 Vision for conceptual similarity scoring
- **Async Processing**: Fast, non-blocking image comparison
- **RESTful API**: Clean, documented endpoints with automatic OpenAPI docs
- **Health Monitoring**: Built-in health checks and configuration endpoints

### Frontend Interface (Streamlit)
- **Drag & Drop Upload**: Easy image uploading with preview
- **Real-time Progress**: Visual progress tracking during analysis
- **Interactive Dashboard**: Clean, responsive UI for viewing results
- **Detailed Metrics**: Visual representation of similarity scores
- **Error Handling**: User-friendly error messages and troubleshooting

## 📋 Requirements

- Python 3.8+
- OpenAI API key (optional, for LLM features)
- 2GB+ RAM recommended for image processing

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/image-similarity-platform.git
cd image-similarity-platform
```

### 2. Create Project Structure

```bash
mkdir -p backend frontend scripts tests
# Move your files to appropriate directories
mv main.py backend/
mv image_utils.py backend/
mv app.py frontend/
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Run the Applications

#### Option 1: Manual Start (Recommended for Development)

**Terminal 1 - Backend:**
```bash
cd backend
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend  
streamlit run app.py
```

#### Option 2: Using Scripts
```bash
chmod +x scripts/*.sh
./scripts/start_both.sh
```

#### Option 3: Docker Compose
```bash
docker-compose up -d
```

**Access the applications:**
- **Web Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Endpoints**: http://localhost:8000

## 🌐 Web Interface Usage

### Streamlit Dashboard

1. **Open your browser** to http://localhost:8501
2. **Upload images** using the drag-and-drop interface:
   - Upload a source image on the left
   - Upload a target image on the right
3. **Click "🔬 Analyze Similarity"** to start the comparison
4. **View results** in an organized dashboard:
   - Computer Vision metrics with explanations
   - LLM semantic analysis with conceptual scoring
   - Progress tracking during analysis

### Features:
- **Visual Preview**: See your uploaded images before analysis
- **Progress Tracking**: Real-time updates during processing
- **Error Handling**: Clear error messages if something goes wrong
- **Responsive Design**: Works on desktop and mobile devices

## 🔗 API Endpoints

### Compare Images
**POST** `/compare-images/`

Upload two images for similarity comparison.

**Parameters:**
- `source_image_file` (file): First image to compare
- `target_image_file` (file): Second image to compare
- `include_llm` (bool, optional): Enable LLM semantic comparison (default: true)
- `llm_timeout` (int, optional): Timeout for LLM API call in seconds (10-300, default: 60)

**Response:**
```json
{
  "request_info": {
    "source_filename": "image1.jpg",
    "target_filename": "image2.jpg",
    "include_llm": true,
    "llm_timeout": 60
  },
  "cv_scores": {
    "mse": 1250.5,
    "ssim": 0.85,
    "histogram_correlation": 0.92,
    "histogram_chi_squared": 15.3
  },
  "llm_comparison": {
    "status": "completed",
    "description": "Both images show similar outdoor landscapes...",
    "conceptual_score_str": "High similarity (8/10)",
    "conceptual_score_value": 8,
    "model_used": "gpt-4-vision-preview"
  },
  "summary": {
    "cv_scores_available": true,
    "llm_comparison_completed": true,
    "total_metrics": 4
  }
}
```

### Health Check
**GET** `/health`

Check API health and configuration status.

### Configuration
**GET** `/config`

Get current API configuration settings.

### Test OpenAI Connection
**POST** `/test-openai/`

Test OpenAI API connectivity without processing images.

### Root Information
**GET** `/`

Get API information and available endpoints.

## 📊 Understanding the Scores

### Computer Vision Metrics

- **MSE (Mean Squared Error)**: Lower values indicate higher similarity (0 = identical)
- **SSIM (Structural Similarity Index)**: Higher values indicate higher similarity (1 = identical)
- **Histogram Correlation**: Higher values indicate higher similarity (1 = identical)
- **Histogram Chi-Squared**: Lower values indicate higher similarity (0 = identical)

### LLM Semantic Score

- **Conceptual Score**: 1-10 scale where 10 represents highest semantic similarity
- **Description**: Natural language explanation of the comparison

## 🐳 Docker Usage

### Build and Run

```bash
# Backend only
docker build -f Dockerfile.backend -t similarity-backend .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key_here similarity-backend

# Frontend only  
docker build -f Dockerfile.frontend -t similarity-frontend .
docker run -p 8501:8501 similarity-frontend

# Full stack
docker-compose up -d
```

### Using Docker Compose

```bash
docker-compose up -d
```

## 📝 Usage Examples

### Python Client

```python
import requests

url = "http://localhost:8000/compare-images/"

with open("image1.jpg", "rb") as f1, open("image2.jpg", "rb") as f2:
    files = {
        "source_image_file": f1,
        "target_image_file": f2
    }
    params = {
        "include_llm": True,
        "llm_timeout": 60
    }
    response = requests.post(url, files=files, params=params)
    result = response.json()
    print(f"SSIM Score: {result['cv_scores']['ssim']}")
    print(f"LLM Score: {result['llm_comparison']['conceptual_score_value']}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/compare-images/" \
  -F "source_image_file=@image1.jpg" \
  -F "target_image_file=@image2.jpg" \
  -G -d "include_llm=true" -d "llm_timeout=60"
```

### JavaScript/Fetch

```javascript
const formData = new FormData();
formData.append('source_image_file', sourceFile);
formData.append('target_image_file', targetFile);

const response = await fetch('http://localhost:8000/compare-images/?include_llm=true', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log('Similarity scores:', result.cv_scores);
console.log('LLM analysis:', result.llm_comparison);
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM features | None |
| `HOST` | Server host address | 0.0.0.0 |
| `PORT` | Server port | 8000 |
| `LOG_LEVEL` | Logging level | info |
| `UPLOADS_DIR` | Temporary upload directory | uploads |

### Supported Image Formats

- JPEG/JPG
- PNG
- BMP
- TIFF
- WebP
- All formats supported by OpenCV

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=. tests/
```

## 📊 Performance

- **Typical response time**: 1-3 seconds (CV only), 5-15 seconds (with LLM)
- **Memory usage**: ~200MB base + image size
- **Concurrent requests**: Supports async processing
- **Rate limits**: Dependent on OpenAI API limits

## 🛡️ Security Considerations

- Temporary files are automatically cleaned up
- No persistent storage of uploaded images
- CORS is enabled for all origins (configure for production)
- API key is required for LLM features
- Input validation for file types and sizes

## 🔧 Troubleshooting

### Common Issues

1. **OpenAI API Key Not Found**
   ```
   Solution: Set OPENAI_API_KEY environment variable
   ```

2. **Image Loading Errors**
   ```
   Solution: Ensure images are valid and in supported formats
   ```

3. **Timeout Errors**
   ```
   Solution: Increase llm_timeout parameter or check network connectivity
   ```

4. **Memory Issues**
   ```
   Solution: Resize large images before upload, increase server memory
   ```

### Debug Mode

Enable debug logging:
```bash
LOG_LEVEL=debug python main.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- Create an issue on GitHub for bug reports
- Check the `/health` endpoint for system status
- Review logs for detailed error information
- Consult OpenAI documentation for LLM-related issues

## 🔮 Roadmap

- [ ] Batch image processing
- [ ] Additional CV metrics (PSNR, LPIPS)
- [ ] Support for other LLM providers
- [ ] Image preprocessing options
- [ ] Results caching
- [ ] API rate limiting
- [ ] Database integration for history

---

**Made with ❤️ using FastAPI and OpenAI**
