# from datetime import datetime
# from flask import Flask, request, Response, jsonify, render_template
# import cv2
# import threading
# import os
# import time
# import pandas as pd
# from ultralytics import YOLO
# from tracker import Tracker
# import csv

# app = Flask(__name__)

# # Global Variables
# model = YOLO("latestptt.pt")
# tracker = Tracker()
# LINE_POSITION = 308
# LINE_OFFSET = 2
# counter_down = set()
# global_pothole_count = 0
# rolling_pothole_count = 0
# lock = threading.Lock()

# # Video variables
# uploaded_video_path = None
# cap = None
# output_frame = None
# csv_file = None

# # # Initialize CSV
# # csv_file = "pothole_detections.csv"
# # with open(csv_file, mode='w', newline='') as file:
# #     writer = csv.writer(file)
# #     writer.writerow(["ID", "Total Potholes Detected", "Potholes Per Frame"])

# @app.route('/')
# def index():
#     """Render the main HTML page."""
#     return render_template('index2.html')


# @app.route('/upload', methods=['POST'])
# def upload_video():
#     """Handle video upload."""
#     global uploaded_video_path, cap, csv_file
#     file = request.files.get('file')
#     if file:
#         # Generate a unique CSV file name
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         video_name = os.path.splitext(file.filename)[0]
#         csv_folder = os.path.join("csv")  # Directory for CSV files
#         os.makedirs(csv_folder, exist_ok=True)  # Create folder if it doesn't exist
#         csv_file = os.path.join(csv_folder, f"{video_name}_{timestamp}.csv")

#         # Save the uploaded video
#         uploaded_video_path = os.path.join("uploads", file.filename)
#         os.makedirs("uploads", exist_ok=True)
#         file.save(uploaded_video_path)
#         cap = cv2.VideoCapture(uploaded_video_path)

#         # Initialize the CSV file with headers
#         with open(csv_file, mode='w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(["ID", "Total Potholes Detected", "Potholes Per Frame"])

#         threading.Thread(target=detection_process).start()  # Start processing in a thread
#         return jsonify({"message": "Video uploaded successfully!", "csv_file": csv_file})
#     return jsonify({"message": "No file uploaded!"}), 400

    

# def detection_process():
#     """Process video for pothole detection."""
#     global output_frame, rolling_pothole_count
#     tracker = Tracker()
#     global_pothole_count = 0
#     frame_count = 0
#     previous_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break  

#         # Step 1: Detect potholes
#         results = model.predict(frame)
#         detections = results[0].boxes.data.detach().cpu().numpy()

#         # Process predictions
#         detected_boxes = []
#         for _, row in pd.DataFrame(detections).iterrows():
#             x1, y1, x2, y2 = map(int, row[:4])
#             detected_boxes.append([x1, y1, x2, y2])

#         # Update tracker with new bounding boxes
#         tracked_objects = tracker.update(detected_boxes)
#         for  x1, y1, x2, y2, obj_id1 in tracked_objects:
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#             cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)  # Draw center points
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
#             cv2.putText(frame, str(obj_id1), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            
#             if LINE_POSITION - LINE_OFFSET <= cy <= LINE_POSITION + LINE_OFFSET:
#                 counter_down.add(obj_id1)

#         with lock:
#             global_pothole_count = len(counter_down)
#             rolling_pothole_count = global_pothole_count - previous_count
#             previous_count = global_pothole_count

#         # Draw detections and counts
#         for x1, y1, x2, y2, obj_id1 in tracked_objects:
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, str(obj_id1), (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # Draw line and counts
#         cv2.line(frame, (0, LINE_POSITION), (frame.shape[1], LINE_POSITION), (0, 0, 255), 2)
#         cv2.putText(frame, f"Rolling: {rolling_pothole_count}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
#         cv2.putText(frame, f"Total: {global_pothole_count}", (10, 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

#         # Save to CSV
#         with open(csv_file, mode='a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([frame_count, global_pothole_count, rolling_pothole_count])

#         # Save the processed frame
#         with lock:
#             output_frame = frame.copy()
#             _, buffer = cv2.imencode('.jpg', output_frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#         frame_count += 1

# # @app.route('/get-data', methods=['GET'])
# # def get_data():
# #     try:
# #         # Check if the file exists and is not empty
# #         if not os.path.exists('D:\pothole\live_analysis\pothole_detections.csv') or os.path.getsize('D:\pothole\live_analysis\pothole_detections.csv') == 0:
# #             return jsonify({'status': 'success', 'data': []}), 200
        
# #         # Load the CSV file with encoding handling
# #         df = pd.read_csv('D:\pothole\live_analysis\pothole_detections.csv', encoding='utf-8')
        
# #         # Convert to JSON format
# #         data = df.to_dict(orient='records')
# #         return jsonify({'status': 'success', 'data': data}), 200
# #     except UnicodeDecodeError:
# #         return jsonify({'status': 'error', 'message': 'File contains invalid characters. Ensure it is UTF-8 encoded.'}), 500
# #     except Exception as e:
# #         return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/get-data', methods=['GET'])
# def get_data():
#     """Fetch data from the current CSV file."""
#     global csv_file
#     try:
#         if not csv_file or not os.path.exists(csv_file):
#             return jsonify({'status': 'error', 'message': 'CSV file not found.'}), 404
        
#         # Load the CSV file and convert it to JSON
#         df = pd.read_csv(csv_file)
#         data = df.to_dict(orient='records')
#         return jsonify({'status': 'success', 'data': data}), 200
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# @app.route('/video_feed')
# def video_feed():
#     """Stream video feed."""
#     return Response(detection_process(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/clear-data', methods=['POST'])
# def clear_data():
#     """Clear data for the current video."""
#     global csv_file, counter_down, global_pothole_count, rolling_pothole_count
#     try:
#         if csv_file and os.path.exists(csv_file):
#             os.remove(csv_file)  # Remove the specific CSV file
        
#         # Reset global variables
#         counter_down.clear()
#         global_pothole_count = 0
#         rolling_pothole_count = 0
#         csv_file = None

#         return jsonify({"message": "Data cleared successfully!"}), 200
#     except Exception as e:
#         return jsonify({"message": f"Error: {str(e)}"}), 500



# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)


# from datetime import datetime
# from flask import Flask, request, Response, jsonify, render_template
# import cv2
# import threading
# import os
# import time
# import pandas as pd
# from ultralytics import YOLO
# from tracker import Tracker
# import csv
# import gridfs
# from pymongo import MongoClient
# from bson import ObjectId
# import uuid  # For generating unique folder names

# app = Flask(__name__)

# # MongoDB setup
# client = MongoClient("mongodb+srv://rndpago:Wzu17hNDUD3rY6iM@pothole.sbzz8.mongodb.net/?retryWrites=true&w=majority&appName=Pothole")  # Replace with your MongoDB URI
# db = client["pothole_detection"]
# metadata_collection = client['media_db']['video_metadata']
# fs = gridfs.GridFS(db)
# # folders_collection = db["folders"]

# # Global Variables
# model = YOLO("latestptt.pt")
# tracker = Tracker()
# LINE_POSITION = 308
# LINE_OFFSET = 2
# counter_down = set()
# global_pothole_count = 0
# rolling_pothole_count = 0
# lock = threading.Lock()

# # Video variables
# uploaded_video_path = None
# cap = None
# output_frame = None
# csv_file = None
# unique_id_folder = None


# @app.route('/')
# def index():
#     """Render the main HTML page."""
#     return render_template('index2.html')


# @app.route('/upload', methods=['POST'])
# def upload_video():
#     """Handle video upload."""
#     global uploaded_video_path, cap, csv_file, unique_id_folder
#     try:
#         file = request.files.get('file')
#     except Exception as e:
#         print(e)
#     if file:
#         # Generate a unique folder using UUID
#         # unique_id= str(uuid.uuid4())
#         # unique_id_folder = os.path.join("data", unique_id)
#         # os.makedirs(unique_id_folder, exist_ok=True)

#         # # Save the uploaded video
#         # uploaded_video_path = os.path.join(unique_id_folder, file.filename)
#         # file.save(uploaded_video_path)
#         # print(uploaded_video_path)
#         video_id = fs.put(file, filename=file.filename, content_type=file.content_type)
#         print(f"video_id is {video_id}")
#         # Process files immediately after uploading
#         video_file = fs.get(ObjectId(video_id))
#         print(f"video_file is:{video_file}")
#         unique_id_folder = os.path.join("data", video_id)
#         os.makedirs(unique_id_folder, exist_ok=True)

#         # Save the uploaded video
#         uploaded_video_path = os.path.join(unique_id_folder, file.filename)
#         file.save(uploaded_video_path)
#         print(uploaded_video_path)
#         try:
#             with open(uploaded_video_path, 'wb') as f:
#                 f.write(video_file.read())
#         except Exception as e:
#             print(f"error : {e}")
#         # Generate a unique CSV file name
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         video_name = os.path.splitext(file.filename)[0]
#         csv_file = os.path.join(unique_id_folder, f"{video_name}_{timestamp}.csv")
#         try:
#         # Initialize the CSV file with headers
#             with open(csv_file, mode='w', newline='') as file:
#                 writer = csv.writer(file)
#                 writer.writerow(["ID", "Total Potholes Detected", "Potholes Per Frame"])
#         except Exception as e:
#             print(e)
#         # Open video for processing
#         cap = cv2.VideoCapture(uploaded_video_path)
        
#         # Start processing in a thread
#         threading.Thread(target=detection_process).start()
#         data = {
#         'video_id': str(video_id),
#         "csv_file": csv_file,
#         }
#         metadata_collection.insert_one(data)
#         # folder_data = {
#         #     "folder_id": unique_id,
#         #     "folder_path": unique_id_folder,
#         #     "csv_file": csv_file,
#         #     "created_at": datetime.now(),
#         #     "session_active": True
#         # }
#         # folders_collection.insert_one(folder_data)
#         return jsonify({"message": "Video uploaded successfully!", "csv_file": csv_file})
#     return jsonify({"message": "No file uploaded!"}), 400


# def detection_process():
#     """Process video for pothole detection."""
#     global output_frame, rolling_pothole_count, unique_id_folder
#     tracker = Tracker()
#     global_pothole_count = 0
#     frame_count = 0
#     previous_count = 0

#     # Path to save processed video
#     processed_video_path = os.path.join(unique_id_folder, "processed_output.avi")
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     video_writer = None

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if video_writer is None:
#             # Initialize video writer
#             height, width, _ = frame.shape
#             video_writer = cv2.VideoWriter(processed_video_path, fourcc, 20.0, (width, height))

#         # Step 1: Detect potholes
#         results = model.predict(frame)
#         detections = results[0].boxes.data.detach().cpu().numpy()

#         # Process predictions
#         detected_boxes = []
#         for _, row in pd.DataFrame(detections).iterrows():
#             x1, y1, x2, y2 = map(int, row[:4])
#             detected_boxes.append([x1, y1, x2, y2])

#         # Update tracker with new bounding boxes
#         tracked_objects = tracker.update(detected_boxes)
#         for x1, y1, x2, y2, obj_id1 in tracked_objects:
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#             cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)  # Draw center points
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
#             cv2.putText(frame, str(obj_id1), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

#             if LINE_POSITION - LINE_OFFSET <= cy <= LINE_POSITION + LINE_OFFSET:
#                 counter_down.add(obj_id1)

#         with lock:
#             global_pothole_count = len(counter_down)
#             rolling_pothole_count = global_pothole_count - previous_count
#             previous_count = global_pothole_count

#         # Draw line and counts
#         cv2.line(frame, (0, LINE_POSITION), (frame.shape[1], LINE_POSITION), (0, 0, 255), 2)
#         cv2.putText(frame, f"Rolling: {rolling_pothole_count}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
#         cv2.putText(frame, f"Total: {global_pothole_count}", (10, 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

#         # Save to CSV
#         with open(csv_file, mode='a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([frame_count, global_pothole_count, rolling_pothole_count])

#         # Save the processed frame to video
#         video_writer.write(frame)

#         # Save the processed frame for live streaming
#         with lock:
#             output_frame = frame.copy()

#         frame_count += 1

#     # Release resources
#     if video_writer:
#         video_writer.release()


# @app.route('/get-data', methods=['GET'])
# def get_data():
#     """Fetch data from the current CSV file."""
#     global csv_file
#     try:
#         if not csv_file or not os.path.exists(csv_file):
#             return jsonify({'status': 'error', 'message': 'CSV file not found.'}), 404

#         # Load the CSV file and convert it to JSON
#         df = pd.read_csv(csv_file)
#         data = df.to_dict(orient='records')
#         return jsonify({'status': 'success', 'data': data}), 200
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500


# @app.route('/video_feed')
# def video_feed():
#     """Stream video feed."""
#     return Response(detection_process(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


# # @app.route('/clear-data', methods=['POST'])
# # def clear_data():
# #     """Clear data for the current video."""
# #     global csv_file, counter_down, global_pothole_count, rolling_pothole_count, unique_id_folder
# #     try:
# #         if unique_id_folder and os.path.exists(unique_id_folder):
# #             # Remove the unique folder and all its contents
# #             for root, dirs, files in os.walk(unique_id_folder, topdown=False):
# #                 for name in files:
# #                     os.remove(os.path.join(root, name))
# #                 for name in dirs:
# #                     os.rmdir(os.path.join(root, name))
# #             os.rmdir(unique_id_folder)

# #         # Reset global variables
# #         counter_down.clear()
# #         global_pothole_count = 0
# #         rolling_pothole_count = 0
# #         csv_file = None
# #         unique_id_folder = None

# #         return jsonify({"message": "Data cleared successfully!"}), 200
# #     except Exception as e:
# #         return jsonify({"message": f"Error: {str(e)}"}), 500


# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)



from datetime import datetime
from flask import Flask, request, Response, jsonify, render_template
import cv2
import threading
import os
import time
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import csv
import uuid  # For generating unique folder names

app = Flask(__name__)

# Global Variables
model = YOLO("latestptt.pt")
tracker = Tracker()
LINE_POSITION = 308
LINE_OFFSET = 2
counter_down = set()
global_pothole_count = 0
rolling_pothole_count = 0
lock = threading.Lock()

# Video variables
uploaded_video_path = None
cap = None
output_frame = None
csv_file = None
unique_id_folder = None


@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index2.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload."""
    global uploaded_video_path, cap, csv_file, unique_id_folder
    file = request.files.get('file')
    if file:
        # Generate a unique folder using UUID
        unique_id_folder = os.path.join("data", str(uuid.uuid4()))
        os.makedirs(unique_id_folder, exist_ok=True)

        # Save the uploaded video
        uploaded_video_path = os.path.join(unique_id_folder, file.filename)
        file.save(uploaded_video_path)

        # Generate a unique CSV file name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.splitext(file.filename)[0]
        csv_file = os.path.join(unique_id_folder, f"{video_name}_{timestamp}.csv")

        # Initialize the CSV file with headers
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "Total Potholes Detected", "Potholes Per Frame"])

        # Open video for processing
        cap = cv2.VideoCapture(uploaded_video_path)

        # Start processing in a thread
        threading.Thread(target=detection_process).start()
        return jsonify({"message": "Video uploaded successfully!", "csv_file": csv_file})
    return jsonify({"message": "No file uploaded!"}), 400


def detection_process():
    """Process video for pothole detection."""
    global output_frame, rolling_pothole_count
    tracker = Tracker()
    global_pothole_count = 0
    frame_count = 0
    previous_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  

        # Step 1: Detect potholes
        results = model.predict(frame)
        detections = results[0].boxes.data.detach().cpu().numpy()

        # Process predictions
        detected_boxes = []
        for _, row in pd.DataFrame(detections).iterrows():
            x1, y1, x2, y2 = map(int, row[:4])
            detected_boxes.append([x1, y1, x2, y2])

        # Update tracker with new bounding boxes
        tracked_objects = tracker.update(detected_boxes)
        for  x1, y1, x2, y2, obj_id1 in tracked_objects:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)  # Draw center points
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
            cv2.putText(frame, str(obj_id1), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            
            if LINE_POSITION - LINE_OFFSET <= cy <= LINE_POSITION + LINE_OFFSET:
                counter_down.add(obj_id1)

        with lock:
            global_pothole_count = len(counter_down)
            rolling_pothole_count = global_pothole_count - previous_count
            previous_count = global_pothole_count

        # Draw detections and counts
        for x1, y1, x2, y2, obj_id1 in tracked_objects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(obj_id1), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw line and counts
        cv2.line(frame, (0, LINE_POSITION), (frame.shape[1], LINE_POSITION), (0, 0, 255), 2)
        cv2.putText(frame, f"Rolling: {rolling_pothole_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Total: {global_pothole_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Save to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frame_count, global_pothole_count, rolling_pothole_count])

        # Save the processed frame
        with lock:
            output_frame = frame.copy()
            _, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        frame_count += 1
   


@app.route('/get-data', methods=['GET'])
def get_data():
    """Fetch data from the current CSV file."""
    global csv_file
    try:
        if not csv_file or not os.path.exists(csv_file):
            return jsonify({'status': 'error', 'message': 'CSV file not found.'}), 404

        # Load the CSV file and convert it to JSON
        df = pd.read_csv(csv_file)
        data = df.to_dict(orient='records')
        return jsonify({'status': 'success', 'data': data}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/video_feed')
def video_feed():
    """Stream video feed."""
    return Response(detection_process(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/clear-data', methods=['POST'])
def clear_data():
    """Clear data for the current video."""
    global csv_file, counter_down, global_pothole_count, rolling_pothole_count, unique_id_folder
    try:
        if unique_id_folder and os.path.exists(unique_id_folder):
            # Remove the unique folder and all its contents
            for root, dirs, files in os.walk(unique_id_folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(unique_id_folder)

        # Reset global variables
        counter_down.clear()
        global_pothole_count = 0
        rolling_pothole_count = 0
        csv_file = None
        unique_id_folder = None

        return jsonify({"message": "Data cleared successfully!"}), 200
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)




# from datetime import datetime
# from flask import Flask, request, Response, jsonify, render_template
# import cv2
# import threading
# import os
# import time
# import pandas as pd
# from ultralytics import YOLO
# from tracker import Tracker
# import csv
# import uuid  # For generating unique folder names

# app = Flask(__name__)

# # Global Variables
# model = YOLO("latestptt.pt")
# tracker = Tracker()
# LINE_POSITION = 308
# LINE_OFFSET = 2
# counter_down = set()
# global_pothole_count = 0
# rolling_pothole_count = 0
# lock = threading.Lock()

# # Video variables
# uploaded_video_path = None
# cap = None
# output_frame = None
# csv_file = None
# unique_id_folder = None


# @app.route('/')
# def index():
#     """Render the main HTML page."""
#     return render_template('index2.html')


# @app.route('/upload', methods=['POST'])
# def upload_video():
#     """Handle video upload."""
#     global uploaded_video_path, cap, csv_file, unique_id_folder
#     file = request.files.get('file')
#     if file:
#         # Generate a unique folder using UUID
#         unique_id_folder = os.path.join("data", str(uuid.uuid4()))
#         os.makedirs(unique_id_folder, exist_ok=True)

#         # Save the uploaded video
#         uploaded_video_path = os.path.join(unique_id_folder, file.filename)
#         file.save(uploaded_video_path)

#         # Generate a unique CSV file name
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         video_name = os.path.splitext(file.filename)[0]
#         csv_file = os.path.join(unique_id_folder, f"{video_name}_{timestamp}.csv")

#         # Initialize the CSV file with headers
#         with open(csv_file, mode='w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(["ID", "Total Potholes Detected", "Potholes Per Frame"])

#         # Open video for processing
#         cap = cv2.VideoCapture(uploaded_video_path)

#         # Start processing in a thread
#         threading.Thread(target=detection_process).start()
#         return jsonify({"message": "Video uploaded successfully!", "csv_file": csv_file})
#     return jsonify({"message": "No file uploaded!"}), 400


# def detection_process():
#     """Process video for pothole detection."""
#     global output_frame, rolling_pothole_count
#     tracker = Tracker()
#     global_pothole_count = 0
#     frame_count = 0
#     previous_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break  

#         # Step 1: Detect potholes
#         results = model.predict(frame)
#         detections = results[0].boxes.data.detach().cpu().numpy()

#         # Process predictions
#         detected_boxes = []
#         for _, row in pd.DataFrame(detections).iterrows():
#             x1, y1, x2, y2 = map(int, row[:4])
#             detected_boxes.append([x1, y1, x2, y2])

#         # Update tracker with new bounding boxes
#         tracked_objects = tracker.update(detected_boxes)
#         for  x1, y1, x2, y2, obj_id1 in tracked_objects:
#             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
#             cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)  # Draw center points
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
#             cv2.putText(frame, str(obj_id1), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            
#             if LINE_POSITION - LINE_OFFSET <= cy <= LINE_POSITION + LINE_OFFSET:
#                 counter_down.add(obj_id1)

#         with lock:
#             global_pothole_count = len(counter_down)
#             rolling_pothole_count = global_pothole_count - previous_count
#             previous_count = global_pothole_count

#         # Draw detections and counts
#         for x1, y1, x2, y2, obj_id1 in tracked_objects:
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(frame, str(obj_id1), (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # Draw line and counts
#         cv2.line(frame, (0, LINE_POSITION), (frame.shape[1], LINE_POSITION), (0, 0, 255), 2)
#         cv2.putText(frame, f"Rolling: {rolling_pothole_count}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
#         cv2.putText(frame, f"Total: {global_pothole_count}", (10, 70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

#         # Save to CSV
#         with open(csv_file, mode='a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([frame_count, global_pothole_count, rolling_pothole_count])

#         # Save the processed frame
#         with lock:
#             output_frame = frame.copy()
#             _, buffer = cv2.imencode('.jpg', output_frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#         frame_count += 1
   


# @app.route('/get-data', methods=['GET'])
# def get_data():
#     """Fetch data from the current CSV file."""
#     global csv_file
#     try:
#         if not csv_file or not os.path.exists(csv_file):
#             return jsonify({'status': 'error', 'message': 'CSV file not found.'}), 404

#         # Load the CSV file and convert it to JSON
#         df = pd.read_csv(csv_file)
#         data = df.to_dict(orient='records')
#         return jsonify({'status': 'success', 'data': data}), 200
#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500


# @app.route('/video_feed')
# def video_feed():
#     """Stream video feed."""
#     return Response(detection_process(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/clear-data', methods=['POST'])
# def clear_data():
#     """Clear data for the current video."""
#     global csv_file, counter_down, global_pothole_count, rolling_pothole_count, unique_id_folder
#     try:
#         if unique_id_folder and os.path.exists(unique_id_folder):
#             # Remove the unique folder and all its contents
#             for root, dirs, files in os.walk(unique_id_folder, topdown=False):
#                 for name in files:
#                     os.remove(os.path.join(root, name))
#                 for name in dirs:
#                     os.rmdir(os.path.join(root, name))
#             os.rmdir(unique_id_folder)

#         # Reset global variables
#         counter_down.clear()
#         global_pothole_count = 0
#         rolling_pothole_count = 0
#         csv_file = None
#         unique_id_folder = None

#         return jsonify({"message": "Data cleared successfully!"}), 200
#     except Exception as e:
#         return jsonify({"message": f"Error: {str(e)}"}), 500


# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

