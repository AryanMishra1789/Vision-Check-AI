from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import math
import os

app = Flask(__name__)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Snellen chart lines with standard vision measurements (20/XX)
# Each line contains: (letters, size, required_correct, current_correct, current_attempts, level_name)
SNELLEN_LINES = [
    ("E", 200, 1, 0, 0, "Level 1/8: Largest"),      # 20/200 - 1 letter, need 1 correct
    ("FP", 100, 1, 0, 0, "Level 2/8: Very Large"),  # 20/100 - 2 letters, need 1 correct
    ("TOZ", 70, 2, 0, 0, "Level 3/8: Large"),       # 20/70 - 3 letters, need 2 correct
    ("LPED", 50, 2, 0, 0, "Level 4/8: Medium-Large"), # 20/50 - 4 letters, need 2 correct
    ("PECFD", 40, 3, 0, 0, "Level 5/8: Medium"),     # 20/40 - 5 letters, need 3 correct (driving standard)
    ("EDFCZP", 30, 3, 0, 0, "Level 6/8: Small"),     # 20/30 - 6 letters, need 3 correct
    ("DEFPOTEC", 20, 4, 0, 0, "Level 7/8: Very Small"), # 20/20 - 8 letters, need 4 correct (normal vision)
    ("FELOPZD", 15, 4, 0, 0, "Level 8/8: Smallest")   # 20/15 - 7 letters, need 4 correct (better than normal)
]

# Global variables to store test state
current_test_state = {
    'distance_ok': False,
    'current_line': 0,  # Current line being tested
    'responses': [],    # All responses
    'acuity': "",       # Final acuity result
    'test_complete': False,
    'current_letter_index': 0,  # For tracking which letter in the current line
    'max_line_reached': 0       # Track the smallest line the user could read
}

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def process_frame(frame):
    """Process video frame to detect face and measure distance."""
    # Convert the BGR image to RGB
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    
    # Process the image and detect face landmarks
    results = face_mesh.process(image)
    
    # Draw the face mesh annotations on the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get specific landmarks for distance calculation
            # Using the outer eye corners (landmarks 33 and 263)
            h, w, _ = image.shape
            left_eye_outer = face_landmarks.landmark[33]
            right_eye_outer = face_landmarks.landmark[263]
            
            # Convert to pixel coordinates
            left_eye_px = (int(left_eye_outer.x * w), int(left_eye_outer.y * h))
            right_eye_px = (int(right_eye_outer.x * w), int(right_eye_outer.y * h))
            
            # Calculate distance between eyes in pixels
            eye_distance_px = calculate_distance(left_eye_px, right_eye_px)
            
            # Estimate distance from camera (very rough estimation)
            # This is a simplified calculation and may need calibration
            distance = 600 / eye_distance_px  # 600 is an arbitrary scaling factor
            
            # Draw eye landmarks
            cv2.circle(image, left_eye_px, 5, (0, 255, 0), -1)
            cv2.circle(image, right_eye_px, 5, (0, 255, 0), -1)
            
            # Draw line between eyes
            cv2.line(image, left_eye_px, right_eye_px, (0, 255, 0), 2)
            
            # Calculate distance in feet (using a more lenient range)
            distance_feet = 600 / eye_distance_px  # Simplified distance calculation
            
            # Add debug text to the frame
            debug_text = f"Eye distance: {eye_distance_px:.1f}px, Est. distance: {distance_feet:.1f}ft"
            cv2.putText(image, debug_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # More lenient distance check (8-12 feet range)
            if 8.0 <= distance_feet <= 12.0:
                current_test_state['distance_ok'] = True
                cv2.putText(image, "Perfect distance! (8-12ft)", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                current_test_state['distance_ok'] = False
                if distance_feet < 8.0:
                    cv2.putText(image, f"Move further from screen ({distance_feet:.1f}ft)", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.putText(image, f"Move closer to screen ({distance_feet:.1f}ft)", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    return image

def generate_frames():
    """Generate video frames with face detection."""
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process the frame
        frame = process_frame(frame)
        
        # Convert the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_test', methods=['POST'])
def start_test():
    """Start the vision test."""
    global current_test_state
    current_test_state = {
        'distance_ok': False,
        'current_line': 0,
        'responses': [],
        'acuity': ""
    }
    return jsonify({"status": "Test started"})

@app.route('/check_distance', methods=['GET'])
def check_distance():
    """Check if user is at correct distance."""
    print(f"[DEBUG] Check distance called. Current state: {current_test_state}")
    response = {
        "distance_ok": current_test_state.get('distance_ok', False),
        "debug": {
            "current_state": current_test_state
        }
    }
    print(f"[DEBUG] Sending response: {response}")
    return jsonify(response)

@app.route('/get_chart_line', methods=['GET'])
def get_chart_line():
    """Get the current Snellen chart line to display."""
    try:
        line_index = int(request.args.get('line', 0))
        print(f"[DEBUG] Getting chart line {line_index}")  # Debug log
        
        if current_test_state['test_complete']:
            return jsonify({"test_complete": True, "acuity": current_test_state['acuity']})
            
        # Ensure line_index is within bounds
        if line_index < 0 or line_index >= len(SNELLEN_LINES):
            print(f"[ERROR] Line index {line_index} out of bounds")
            return jsonify({"error": "Invalid line index", "success": False}), 400
            
        if 0 <= line_index < len(SNELLEN_LINES):
            print(f"[DEBUG] Fetching line {line_index} from SNELLEN_LINES")
            print(f"[DEBUG] SNELLEN_LINES length: {len(SNELLEN_LINES)}")
            print(f"[DEBUG] SNELLEN_LINES content: {SNELLEN_LINES}")
            
            try:
                line_data = SNELLEN_LINES[line_index]
                print(f"[DEBUG] Line data: {line_data}")
                print(f"[DEBUG] Line data type: {type(line_data)}")
                print(f"[DEBUG] Line data length: {len(line_data) if hasattr(line_data, '__len__') else 'N/A'}")
                
                # Ensure we have at least 5 elements in the tuple
                if len(line_data) < 5:
                    print(f"[ERROR] Invalid SNELLEN_LINES format at index {line_index}. Got length: {len(line_data)}")
                    return jsonify({"error": "Invalid test configuration", "success": False}), 500
            except Exception as e:
                print(f"[ERROR] Error accessing SNELLEN_LINES[{line_index}]: {str(e)}")
                return jsonify({"error": f"Error accessing test data: {str(e)}", "success": False}), 500
                
            # Unpack the line data safely
            letters = line_data[0]
            size = line_data[1]
            required = line_data[2]
            correct = line_data[3] if len(line_data) > 3 else 0
            attempts = line_data[4] if len(line_data) > 4 else 0
            
            # Reset the counters for this line if it's a new test or moving to a new line
            if line_index != current_test_state.get('current_line', -1):
                current_test_state['current_line'] = line_index
                current_test_state['current_letter_index'] = 0
                current_test_state['max_line_reached'] = max(current_test_state['max_line_reached'], line_index)
                
            # Get the current letter to show (one at a time)
            try:
                letter_index = current_test_state['current_letter_index']
                if not letters or letter_index >= len(letters):
                    print(f"[ERROR] Invalid letter index {letter_index} for letters: {letters}")
                    return jsonify({"error": "Invalid test configuration", "success": False}), 500
                current_letter = letters[letter_index]
                print(f"[DEBUG] Showing letter {current_letter} (index {letter_index}) from {letters}")
            except Exception as e:
                print(f"[ERROR] Error accessing letter at index {current_test_state['current_letter_index']} from {letters}: {str(e)}")
                return jsonify({"error": "Error accessing test letters", "success": False}), 500
            
            # Get level information - ensure we don't go out of bounds
            level_info = f"Level {line_index + 1}/{len(SNELLEN_LINES)}"
            try:
                if len(SNELLEN_LINES[line_index]) > 5 and SNELLEN_LINES[line_index][5]:
                    level_info = SNELLEN_LINES[line_index][5]
            except Exception as e:
                print(f"[WARNING] Could not get level info: {str(e)}")
                # Continue with default level info
            
            return jsonify({
                "letters": current_letter,  # Show one letter at a time
                "all_letters": letters,     # For debugging
                "acuity": f"20/{size}",
                "is_last": line_index == len(SNELLEN_LINES) - 1,
                "success": True,
                "current_letter_index": current_test_state['current_letter_index'],
                "total_letters": len(letters),
                "level_info": level_info,
                "current_level": line_index + 1,
                "total_levels": len(SNELLEN_LINES)
            })
        return jsonify({"error": "Invalid line index", "success": False}), 400
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/submit_response', methods=['POST'])
def submit_response():
    """Handle user's response to the current letter."""
    try:
        data = request.get_json()
        user_response = data.get('response', '').strip().upper()
        
        if current_test_state['test_complete']:
            return jsonify({"test_complete": True, "acuity": current_test_state['acuity']})
            
        line_index = current_test_state.get('current_line', 0)
        if line_index < 0 or line_index >= len(SNELLEN_LINES):
            return jsonify({"error": "Invalid test state", "success": False}), 400
            
        # Get the current line data - handle both 5 and 6 value tuples
        line_data = SNELLEN_LINES[line_index]
        letters = line_data[0]
        size = line_data[1]
        required = line_data[2]
        correct = line_data[3] if len(line_data) > 3 else 0
        attempts = line_data[4] if len(line_data) > 4 else 0
        current_letter_index = current_test_state.get('current_letter_index', 0)
        
        # Check if the response is correct for the current letter
        correct_letter = letters[current_letter_index]
        is_correct = user_response == correct_letter
        
        # Update the current line's stats - maintain all 6 values if they exist
        if is_correct:
            correct_count = correct + 1
        else:
            correct_count = correct
            
        # Create a new tuple with all the original values, updating only what's needed
        updated_line = list(SNELLEN_LINES[line_index])
        if len(updated_line) > 3:
            updated_line[3] = correct_count  # Update correct count
        if len(updated_line) > 4:
            updated_line[4] = attempts + 1   # Update attempts
            
        SNELLEN_LINES[line_index] = tuple(updated_line)
        
        # Log the response
        current_test_state['responses'].append({
            'line_index': line_index,
            'letter_index': current_letter_index,
            'letter': correct_letter,
            'response': user_response,  # Fixed variable name from response to user_response
            'is_correct': is_correct,
            'acuity': f"20/{size}"
        })
        
        # Move to next letter or determine if we should move to next line
        current_letter_index += 1
        current_test_state['current_letter_index'] = current_letter_index
        
        # Check if we've completed all letters in this line
        if current_letter_index >= len(letters):
            # Check if we got enough correct for this line
            correct, required = SNELLEN_LINES[line_index][3:5]
            
            if correct >= required:
                # Passed this line, move to next line
                next_line = line_index + 1
                if next_line < len(SNELLEN_LINES):
                    current_test_state['current_line'] = next_line
                    current_test_state['current_letter_index'] = 0
                    current_test_state['max_line_reached'] = max(current_test_state['max_line_reached'], next_line)
                else:
                    # Completed all lines
                    current_test_state['test_complete'] = True
                    current_test_state['acuity'] = f"20/{SNELLEN_LINES[-1][1]}"  # Best possible
            else:
                # Failed this line, end test
                current_test_state['test_complete'] = True
                if line_index > 0:
                    current_test_state['acuity'] = f"20/{SNELLEN_LINES[line_index-1][1]}"
                else:
                    current_test_state['acuity'] = "Worse than 20/200"
        
        return jsonify({
            "status": "Response recorded",
            "is_correct": is_correct,
            "correct_letter": correct_letter,
            "test_complete": current_test_state['test_complete'],
            "acuity": current_test_state['acuity'],
            "move_to_next_letter": not current_test_state['test_complete'] and current_letter_index < len(letters),
            "move_to_next_line": not current_test_state['test_complete'] and current_letter_index >= len(letters) and SNELLEN_LINES[line_index][3] >= SNELLEN_LINES[line_index][2]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_results', methods=['GET'])
def get_results():
    """Get the test results."""
    if not current_test_state['test_complete'] and current_test_state['responses']:
        # If test isn't complete but we have responses, calculate current acuity
        max_line = current_test_state.get('max_line_reached', 0)
        if max_line > 0:
            current_test_state['acuity'] = f"20/{SNELLEN_LINES[max_line][1]}"
        else:
            current_test_state['acuity'] = "Worse than 20/200"
    
    return jsonify({
        "acuity": current_test_state['acuity'],
        "responses": current_test_state['responses'],
        "test_complete": current_test_state['test_complete'],
        "max_line_reached": current_test_state.get('max_line_reached', 0)
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
