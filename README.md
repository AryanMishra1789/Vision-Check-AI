# Vision Check AI

An accessible and accurate vision screening tool that uses your device's camera and computer vision to estimate visual acuity.

## Features

- Real-time face and eye tracking using MediaPipe
- Digital Snellen chart for vision testing
- Distance measurement to ensure proper testing conditions
- Interactive test interface
- Results with estimated visual acuity

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Webcam
- Modern web browser (Chrome, Firefox, Edge, or Safari)

## Installation

1. Clone this repository or download the source code
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Activate your virtual environment if not already activated:
   ```
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

2. Run the Flask application:
   ```
   python app.py
   ```

3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

4. Follow the on-screen instructions to complete the vision test.

## How It Works

1. **Distance Check**: The application uses computer vision to ensure you're at the correct distance (approximately 10 feet/3 meters) from the screen.

2. **Vision Test**: You'll be shown lines of letters in decreasing sizes. Read each line aloud or type what you see.

3. **Results**: Based on your responses, the application will estimate your visual acuity (e.g., 20/20, 20/40, etc.).

## Important Notes

- This is a screening tool only and not a substitute for a professional eye examination.
- Results may vary based on screen size, resolution, and lighting conditions.
- For accurate results, ensure good lighting and position yourself at the recommended distance.
- Consult an eye care professional for a comprehensive eye examination.

## Troubleshooting

- If the camera doesn't work, ensure no other application is using the webcam.
- Make sure you've granted camera permissions in your browser.
- For best results, use the application in a well-lit environment.

## License

This project is open source and available under the [MIT License](LICENSE).
