# PARTSOL.COM INTERVIEW QUESTIONS
Hello, human.

Your goal is to demonstrate your coding skills by creating a video recording of your answers to some general knowledge questions, writing an ML API demo using Docker, python3, and a bit of magic. Please spend minimal effort on graphics and UI, as this is not a test of your UI coding skills. Just don't stress on frontend stuff.

# 1) GENERAL KNOWLEDGE VIDEO DEMONSTRATION
- Please make a video recording of your answers to the questions in the notebook: https://github.com/daniel-partsol/fizzbuzz/blob/main/Interview_Questions.ipynb
- Please send the video file to brian.rau@parsol.com and ed.galarza@partsol.com
- After you submit the video, proceed to step 2, below.

# 2) MODEL DEPLOYMENT DEMONSTRATION

- Create a container to process inference requests from any pretrained model in the huggingface model hub: https://huggingface.co/models
- Your solution should include server components to support multiple parallel incoming requests (e.g., NGINX/gunicorn)
- Create a notebook to demonstrate requests that POST to the container endpoint and print out the response
- Please explain why you have chosen this model as your demonstration

---

## Setup and Usage Instructions

### Prerequisites

- Docker and Docker Compose installed
- Python 3.11+ (for running the demo notebook locally, if needed)
- At least 4GB RAM available for Docker

### Quick Start

1. **Build and start the containers:**
   ```bash
   docker-compose up -d
   ```

2. **Check that services are running:**
   ```bash
   docker-compose ps
   ```

3. **View logs (optional):**
   ```bash
   docker-compose logs -f
   ```

4. **Test the API:**
   ```bash
   curl http://localhost/health
   ```

5. **Make a prediction:**
   ```bash
   curl -X POST http://localhost/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I love this product!"}'
   ```

6. **Run the demo notebook:**
   - Open `demo_notebook.ipynb` in Jupyter
   - Execute all cells to see comprehensive API demonstrations
   - The notebook includes:
     - Health checks
     - Single and multiple requests
     - Parallel request handling demonstration
     - Performance comparisons
     - Error handling examples

7. **Stop the containers:**
   ```bash
   docker-compose down
   ```

### Architecture

The deployment uses a three-tier architecture:

```
Client → NGINX (Port 80) → Gunicorn (4 workers) → Flask App → DistilBERT Model
```

- **NGINX**: Reverse proxy and load balancer
- **Gunicorn**: WSGI server with 4 worker processes for parallel request handling
- **Flask**: Web framework handling API logic
- **DistilBERT**: Sentiment analysis model (loaded once, reused for all requests)

### API Endpoints

#### Health Check
```bash
GET http://localhost/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu"
}
```

#### Predict Sentiment
```bash
POST http://localhost/predict
Content-Type: application/json

{
  "text": "Your text here"
}
```

Response:
```json
{
  "sentiment": "POSITIVE",
  "score": 0.9998,
  "positive_score": 0.9998,
  "negative_score": 0.0002,
  "text": "Your text here"
}
```

### Model Information

**Model:** `distilbert-base-uncased-finetuned-sst-2-english`

See [model_explanation.md](model_explanation.md) for detailed rationale on model selection.

### Files Structure

```
.
├── README.md                 # This file
├── answers.md                # Comprehensive answers to general knowledge questions
├── app.py                    # Flask application with model inference
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container definition
├── docker-compose.yml       # Service orchestration
├── nginx.conf               # NGINX configuration
├── .dockerignore            # Docker build exclusions
├── demo_notebook.ipynb      # API demonstration notebook
├── model_explanation.md     # Model selection rationale
└── Interview_Questions.ipynb # Original interview questions
```

### Troubleshooting

**Issue: Containers won't start**
- Check Docker is running: `docker ps`
- Check ports 80 is not in use: `lsof -i :80`
- View logs: `docker-compose logs`

**Issue: Model download is slow**
- The model (~260MB) downloads on first run
- Subsequent starts use cached model
- Check network connection if download fails

**Issue: Out of memory errors**
- Reduce Gunicorn workers in Dockerfile: `--workers 2`
- Ensure Docker has sufficient memory allocated

**Issue: API returns 503 errors**
- Model may still be loading (check logs)
- Wait 30-60 seconds after container start
- Check health endpoint: `curl http://localhost/health`

### Development

To modify the application:

1. Edit `app.py` for API changes
2. Rebuild container: `docker-compose build flask`
3. Restart: `docker-compose up -d`

### Production Considerations

For production deployment:

- Add SSL/TLS certificates to NGINX
- Use environment variables for configuration
- Implement proper logging and monitoring
- Set up health check monitoring
- Consider using a GPU-enabled base image for faster inference
- Add authentication/API keys if needed
- Configure resource limits in docker-compose.yml

### Performance

- **Single request latency**: ~50-150ms (CPU)
- **Parallel requests**: Handles 16+ concurrent requests efficiently
- **Throughput**: ~10-20 requests/second (depending on hardware)
- **Model size**: ~260MB
- **Container memory**: ~2-3GB (including model)
