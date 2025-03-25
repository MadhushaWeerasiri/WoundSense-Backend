# WoundSense - Backend

## Overview

WoundSense is a web-based application designed to assist non-expert users in classifying wound types, providing care suggestions, and offering educational content on wound care. The backend, built with FastAPI, handles API requests, image processing, and model inference for wound classification. It integrates a pre-trained Convolutional Neural Network (CNN) model to classify ten wound types (abrasions, bruises, burns, cuts, diabetic wounds, lacerations, normal skin, pressure wounds, surgical wounds, venous wounds) using TensorFlow. The backend communicates with the React frontend running on `http://localhost:3000`, ensuring seamless cross-origin requests via CORS middleware. It is optimized for performance (predictions within 1.5 seconds, under the 5-second requirement) and includes error handling for reliability, supporting the project’s goal of providing an accessible, user-friendly tool for non-expert users.

## Features

- **API Endpoints**:
  - `/ping`: Health check endpoint to confirm the API is running.
  - `/predict`: Accepts image uploads, processes them, and returns the predicted wound type, confidence score, and care suggestions.
- **Image Processing**: Resizes images to 640x640 pixels, converts to RGB format, and prepares them for model inference.
- **Model Inference**: Uses a pre-trained CNN model (loaded from `1.h5`) to classify wounds with a confidence score.
- **Care Suggestions**: Maps predicted wound types to predefined care advice (e.g., "Clean with water and monitor for infection").
- **Error Handling**: Robust error handling with try-except blocks and `HTTPException` for invalid inputs (e.g., unsupported file formats).
- **Logging**: Logs key events (e.g., model loading, prediction results) for debugging and monitoring.

## Prerequisites

- **Python**: Version 3.12.7 or higher (download from [python.org](https://www.python.org)).
- **pip**: Package manager for Python, typically included with Python installation.
- **Frontend Server**: Ensure the React frontend is running on `http://localhost:3000` (refer to the frontend README for setup).
- **Pre-trained Model**: The CNN model file (`1.h5`) must be available in the project directory (`../model/1.h5` relative to the backend script).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MadhushaWeerasiri/WoundSense-Backend.git
   cd WoundSense-Backend
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   This installs FastAPI, Uvicorn, TensorFlow, Pillow (PIL), NumPy, and other dependencies listed in `requirements.txt`. If `requirements.txt` is not provided, install manually:
   ```bash
   pip install fastapi uvicorn tensorflow pillow numpy python-multipart
   ```

4. **Verify Model File**:
   - Ensure the pre-trained CNN model file (`1.h5`) is located at `../model/1.h5` relative to the backend script. If not, update the path in the code or place the file in the correct directory.

## Running the Application

1. **Start the Backend Server**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
   This runs the FastAPI application on `http://localhost:8000`.

2. **Verify the Server**:
   - Open your browser or use a tool like Postman to access `http://localhost:8000/ping`.
   - Expected response: `{"message": "Hello, FastAPI is running!"}`.

3. **Interact with the Frontend**:
   - Ensure the React frontend is running on `http://localhost:3000`.
   - Use the frontend to upload wound images, which will send requests to the `/predict` endpoint.

## Project Structure

```
WoundSense-Backend/
├── main.py               # Main FastAPI application script
├── requirements.txt      # List of Python dependencies
├── ../model/
│   └── 1.h5             # Pre-trained CNN model file
└── README.md            # This file
```

## Usage

- **Health Check**:
  - Endpoint: `GET /ping`
  - Response: `{"message": "Hello, FastAPI is running!"}`
  - Use this to confirm the backend is operational.

- **Wound Prediction**:
  - Endpoint: `POST /predict`
  - Request: Upload an image file (JPG/PNG) via the frontend or directly via tools like Postman.
  - Response: JSON object with the predicted wound type, confidence score, and care suggestions.
    ```json
    {
      "class": "cut",
      "confidence": 0.75,
      "suggestions": "Clean with water and monitor for infection."
    }
    ```
  - The backend processes the image, performs inference using the CNN model, and returns results within 1.5 seconds.

## Development Notes

- **FastAPI**: Chosen for its high performance and asynchronous request handling, ensuring quick predictions (1.5 seconds average).
- **CORS Middleware**: Configured to allow requests from `http://localhost:3000`, enabling frontend-backend communication.
- **Image Processing**: The `read_file_as_image` function uses Pillow (PIL) to resize images to 640x640 pixels and convert them to RGB format for model compatibility.
- **Model Integration**: The CNN model is loaded using TensorFlow from the `1.h5` file, with error handling for loading failures.
- **Care Suggestions**: The `get_suggestions` function maps predicted classes to predefined suggestions, defaulting to a generic message if no match is found.
- **Error Handling**: Uses try-except blocks and `HTTPException` to handle invalid inputs (e.g., unsupported file formats) and model failures, with logging for debugging.
- **Logging**: Logs key events like model loading and prediction results, aiding in troubleshooting.

## Troubleshooting

- **Model Fails to Load**:
  - Ensure the `1.h5` file is in the correct path (`../model/1.h5`).
  - Check the log output for errors (e.g., "Model failed to load").
  - Verify TensorFlow is installed correctly (`pip show tensorflow`).

- **CORS Issues**:
  - Confirm the frontend URL (`http://localhost:3000`) is allowed in the CORS middleware configuration.
  - If accessing from a different origin, update the `origins` list in `main.py`.

- **Invalid Image Format**:
  - The backend only accepts JPG/PNG files. Ensure the uploaded file is in a supported format.
  - Check the error response for details (e.g., HTTP 400 with "Invalid image file").

- **Slow Predictions**:
  - Predictions should take ~1.5 seconds. If slower, check system resources (CPU usage) or network latency.
  - Ensure the model file is not corrupted and TensorFlow is optimized for your hardware.

## Future Improvements

- **Cloud Deployment**: Transition to a cloud-based setup (e.g., AWS, Google Cloud) to improve scalability and support up to 1,000 concurrent users.
- **Enhanced Model**: Integrate transfer learning (e.g., MobileNetV2) to improve accuracy beyond the current 81% (target: 85%).
- **Real-Time Monitoring**: Add endpoints for continuous wound monitoring, enabling dynamic treatment adjustments.
- **Database Integration**: Replace the static `SUGGESTIONS` list with a database (e.g., PostgreSQL) for more flexible suggestion management.
- **Multilingual Support**: Extend care suggestions to support multiple languages for broader accessibility.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Open a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions or support, contact the developer at `mdweerasiri@gmail.com`.  
GitHub Repository: [https://github.com/MadhushaWeerasiri/WoundSense-Backend](https://github.com/MadhushaWeerasiri/WoundSense-Backend)

