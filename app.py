import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from collections import defaultdict
import time
import threading
import queue
import logging
import traceback
from datetime import datetime
from PIL import Image, ImageTk
import psutil
import json
from datetime import datetime
from pathlib import Path
import torch
import sys
import os
import gc  # Import garbage collector module
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

class OptimizedFrameGrabber:
    """
    This class is responsible for capturing frames from an RTSP stream efficiently
    while handling reconnections, frame skipping, and memory management.
    """
    def __init__(self, rtsp_url, queue_size=2):
        """
        Initializes the frame grabber.
        
        :param rtsp_url: RTSP stream URL for video capture.
        :param queue_size: Maximum number of frames stored in the buffer.
        """
        self.rtsp_url = rtsp_url
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.reconnect_delay = 2
        self.max_reconnect_attempts = 5
        self.current_reconnect_attempts = 0
        self.frame_counter = 0
        self.last_frame = None
        self.lock = threading.Lock()
        
    def start(self):
        """
        Starts the frame grabbing thread.
        """
        threading.Thread(target=self.grab, daemon=True).start()
        return self
        
    def grab(self):
        """
        Captures frames from the RTSP stream and stores them in a queue.
        Handles reconnections in case of failure.
        """
        while not self.stopped:
            try:
                # Optimized RTSP settings
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                    'protocol_whitelist;file,rtp,udp,tcp,rtsp|'
                    'rtsp_transport;tcp|'
                    'fflags;nobuffer|'
                    'flags;low_delay|'
                    'strict;experimental|'
                    'fps;25'
                )
                
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    raise ConnectionError("Failed to open RTSP stream")
                
                # Configure/Optimized capture settings
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'HEVC'))  # Use HEVC (H.265) codec
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce buffer size for real-time capture
                cap.set(cv2.CAP_PROP_FPS, 25) # Set target FPS equal to input FPS
                
                # Reset reconnection attempts on successful connection
                self.current_reconnect_attempts = 0
                
                while not self.stopped:
                    ret, frame = cap.read()
                    if not ret:
                        raise ConnectionError("Failed to grab frame")
                    
                    # Frame counter for processing control
                    self.frame_counter += 1
                    
                    # Process every second frame for performance
                    if self.frame_counter % 1 != 0: #No skipping
                        continue
                    
                    # Optimized resize using INTER_LINEAR - Resize frame to 1280x720
                    frame = cv2.resize(frame, (1280, 720), 
                                    interpolation=cv2.INTER_LINEAR)
                    
                    with self.lock:
                        # Clear the queue if full to keep only the latest frame
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.frame_queue.put(frame)
                        self.last_frame = frame # Store the latest frame
                    
                    # Memory optimization: Force garbage collection periodically
                    if self.frame_counter % 300 == 0:  # Every 300 frames
                        gc.collect()
                    
            except Exception as e:
                logging.error(f"Stream error: {str(e)}")
                self.current_reconnect_attempts += 1
                
                if self.current_reconnect_attempts >= self.max_reconnect_attempts: # Stop if max reconnection attempts are exceeded
                    logging.error("Max reconnection attempts reached")
                    self.stopped = True
                    break
                    
                time.sleep(self.reconnect_delay) # Wait before attempting to reconnect
                continue
                
            finally:
                if 'cap' in locals():
                    cap.release()
        
    def read(self):
        """
        Reads the latest frame from the queue.
        If the queue is empty, returns the last available frame.
        
        :return: (bool, frame) - Whether a frame is available and the frame itself.
        """
        with self.lock:
            try:
                frame = self.frame_queue.get_nowait()
                return True, frame
            except queue.Empty:
                if self.last_frame is not None:
                    return True, self.last_frame.copy()
                return False, None
            
    def stop(self):
        """
        Stops the frame grabber and clears the frame queue.
        """
        self.stopped = True
        with self.lock:
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

class PeopleCounter:
    """
    This class detects and counts people using YOLOv5 while tracking movement
    in a predefined zone to determine entries and exits.
    """
    # Define presets as a class variable
    PRESETS = {
        'High Accuracy': {
            'confidence_threshold': 0.4,
            'nms_threshold': 0.3,
            'process_every_n_frames': 1,
            'tracking_distance_threshold': 75,
            'detection_zone_width': 100  # Changed from detection_zone_height
        },
        'High Performance': {
            'confidence_threshold': 0.6,
            'nms_threshold': 0.5,
            'process_every_n_frames': 1,
            'tracking_distance_threshold': 150,
            'detection_zone_width': 75  # Changed from detection_zone_height
        },
        'Crowded Scene': {
            'confidence_threshold': 0.4,
            'nms_threshold': 0.6,
            'process_every_n_frames': 2,
            'tracking_distance_threshold': 100,
            'detection_zone_width': 150  # Changed from detection_zone_height
        },
        'Sparse Scene': {
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'process_every_n_frames': 3,
            'tracking_distance_threshold': 100,
            'detection_zone_width': 100  # Changed from detection_zone_height
        }
    }

    def __init__(self, max_capacity=180): #Defines the max number of people in a location
        """
        Initializes the people counter with default settings.
        :param max_capacity: Maximum number of people that can be counted.
        """
        self.frame_counter = 0
        self.max_capacity = max_capacity
        self.zone_center_ratio = 0.5
        
        # Initialize tracking variables
        self.current_occupancy = 0
        self.people_in = 0
        self.people_out = 0
        self.tracked_objects = defaultdict(list) #self.tracked_objects → Maintains detected objects across frames.
        self.next_object_id = 0
        self.crossing_states = defaultdict(lambda: {'zone': None, 'timestamp': None}) #self.crossing_states → Stores movement history of detected people.
        
        # Initialize detection lines
        self.upper_line_y = None
        self.lower_line_y = None
        self.detection_zone_width = 100 #(zone boundary for movement tracking).
        
        # Initialize preset
        self.preset_names = list(self.PRESETS.keys())
        self.current_preset = self.preset_names[0]
        self._apply_preset(self.current_preset)
        
        try:
            # Load YOLOv5 model
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            
            #Loads the local Model if the web is not connected 
            #self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5s.pt', force_reload=True)

            #self.model = DetectMultiBackend("yolov5/models/yolov5s.pt", device="cpu")



            # Set model parameters
            self.model.conf = self.confidence_threshold  # Confidence threshold
            self.model.iou = self.nms_threshold  # NMS threshold
            self.model.classes = [0]  # Only detect people (class 0 in COCO dataset)
            
            # Use GPU if available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            # Set inference mode
            self.model.eval()
            
            logging.info(f"YOLOv5 model loaded successfully on {self.device}")
            
        except Exception as e:
            logging.error(f"Failed to initialize YOLOv5 model: {str(e)}")
            raise

    def detect_people(self, frame):
        """
        Detects people in the given frame using YOLOv5.
        :param frame: The input video frame.
        :return: List of bounding boxes for detected people.
        """
        try:
            # Convert frame to RGB for YOLOv5
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Inference
            with torch.no_grad():
                results = self.model(frame_rgb)
            
            # Extract detections for people only
            people_detections = []
            for det in results.xyxy[0]:  # xyxy format: x1, y1, x2, y2, confidence, class
                if det[5] == 0:  # Class 0 is person
                    x1, y1, x2, y2, conf = det[:5]
                    w = x2 - x1
                    h = y2 - y1
                    # Additional filtering for more stable detections
                    if w * h > 100:  # Minimum area threshold
                        people_detections.append([int(x1), int(y1), int(w), int(h)])
            
            return people_detections
            
        except Exception as e:
            logging.error(f"Error in person detection: {str(e)}")
            return []
        
    def _apply_preset(self, preset_name): 
        """
        Applies a pre-defined preset configuration.
        :param preset_name: The name of the preset to apply.
        """
        #Applies different confidence thresholds, tracking distance, and frame skipping based on scene type:
        if preset_name in self.PRESETS:
            preset = self.PRESETS[preset_name]
            for key, value in preset.items():
                setattr(self, key, value)
            
            # Update YOLOv5 model parameters
            if hasattr(self, 'model'):
                self.model.conf = self.confidence_threshold
                self.model.iou = self.nms_threshold
            
            logging.info(f"Applied preset: {preset_name}")
            return True
        return False

    def process_frame(self, frame):
        """
        Processes a single frame for detection and tracking.
        :param frame: The input frame.
        :return: Processed frame with visualization.
        """
        try:
            self.frame_counter += 1
            
            width = frame.shape[1]  # Changed from height to width
            zone_center = int(width * self.zone_center_ratio)
            self.left_line_x = zone_center - self.detection_zone_width // 2 #Defines Entry/Exit Lines
            self.right_line_x = zone_center + self.detection_zone_width // 2

            detected_boxes = []
            if self.frame_counter % self.process_every_n_frames == 0:
                detected_boxes = self.detect_people(frame) # Detects People (calls detect_people).
                self.track_and_count(frame, detected_boxes)
            
            processed_frame = self._draw_visualization(frame.copy(), detected_boxes)
            return processed_frame
            
        except Exception as e:
            logging.error(f"Error processing frame: {str(e)}")
            traceback.print_exc()
            return frame
            
    def _draw_visualization(self, frame, detected_boxes):
        """
        Draws visualization elements (detection zones, bounding boxes, and statistics).
        :param frame: The frame to draw on.
        :param detected_boxes: List of detected bounding boxes.
        :return: Frame with visualization elements.
        """
        try:
            overlay = frame.copy()
            
            # Draw vertical detection zones
            cv2.line(frame, (self.left_line_x, 0), 
                    (self.left_line_x, frame.shape[0]), (0, 255, 0), 2)
            cv2.line(frame, (self.right_line_x, 0), 
                    (self.right_line_x, frame.shape[0]), (0, 255, 0), 2)
            
            # Detection zone overlay (vertical)
            cv2.rectangle(overlay, (self.left_line_x, 0), 
                        (self.right_line_x, frame.shape[0]), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Draw all current detection boxes
            for box in detected_boxes:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Calculate capacity metrics
            remaining_capacity = self.max_capacity - self.current_occupancy
            capacity_percentage = (self.current_occupancy / self.max_capacity) * 100
            
            # Color coding based on capacity
            if capacity_percentage > 90:
                color = (0, 0, 255)  # Red
            elif capacity_percentage > 75:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green
            
            # Draw count information
            info_text = [
                f"Occupancy: {self.current_occupancy}/{self.max_capacity}",
                f"Remaining: {remaining_capacity}",
                f"Capacity: {capacity_percentage:.1f}%",
                f"Total In: {self.people_in}  Out: {self.people_out}"
            ]
            
            for i, text in enumerate(info_text):
                y_pos = 30 + i * 30
                cv2.putText(frame, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            return frame
            
        except Exception as e:
            logging.error(f"Error in visualization: {str(e)}")
            return frame

    def track_and_count(self, frame, detected_boxes): 
        """This function ensures each detected person is tracked across frames and counted when they enter/exit.""" 
        try:
            current_detections = set()
            current_time = time.time()
            
            for box in detected_boxes: # Loop through each detected person
                x, y, w, h = box
                center_x = x + w/2  # Changed from center_y to center_x
                
                # Match detection with existing tracks
                matched = False
                min_dist = float('inf')
                matched_id = None
                # Improved tracking logic with multiple position history
                for obj_id, positions in self.tracked_objects.items(): #Find if the detected person matches an existing track (IOU-based Matching).
                    if positions:
                        last_x, last_y = positions[-1]
                        dist = ((center_x - last_x) ** 2 + (y + h/2 - last_y) ** 2) ** 0.5
                        if dist < min_dist and dist < self.tracking_distance_threshold:
                            min_dist = dist
                            matched_id = obj_id
                            matched = True
                
                if matched:
                    self.tracked_objects[matched_id].append((center_x, y + h/2))
                    current_detections.add(matched_id)
                    
                    # Get current zone
                    current_zone = self._determine_zone(center_x)
                    prev_state = self.crossing_states[matched_id]
                    
                    # Logic for counting with improved accuracy
                    if prev_state['zone'] != current_zone:
                        if prev_state['zone'] == 'left' and current_zone == 'detection_zone':
                            self.crossing_states[matched_id] = {
                                'zone': current_zone,
                                'timestamp': current_time,
                                'direction': 'entering'
                            }
                        elif prev_state['zone'] == 'right' and current_zone == 'detection_zone':
                            self.crossing_states[matched_id] = {
                                'zone': current_zone,
                                'timestamp': current_time,
                                'direction': 'leaving'
                            }
                        elif current_zone in ['left', 'right']:
                            if prev_state.get('direction') == 'entering' and current_zone == 'right':
                                if self.current_occupancy < self.max_capacity:
                                    self.people_in += 1
                                    self.current_occupancy += 1
                                    logging.info(f"Person entered. Total inside: {self.current_occupancy}")
                            elif prev_state.get('direction') == 'leaving' and current_zone == 'left':
                                self.people_out += 1
                                self.current_occupancy = max(0, self.current_occupancy - 1)
                                logging.info(f"Person exited. Total inside: {self.current_occupancy}")
                            
                            self.crossing_states[matched_id]['zone'] = current_zone
                else:
                    # Create new track
                    self.tracked_objects[self.next_object_id] = [(center_x, y + h/2)]
                    self.crossing_states[self.next_object_id] = {
                        'zone': self._determine_zone(center_x),
                        'timestamp': current_time
                    }
                    current_detections.add(self.next_object_id)
                    self.next_object_id += 1
            
            # Clean up old tracks
            for obj_id in list(self.tracked_objects.keys()):
                if obj_id not in current_detections:
                    del self.tracked_objects[obj_id]
                    if obj_id in self.crossing_states:
                        del self.crossing_states[obj_id]
                        
        except Exception as e:
            logging.error(f"Error in tracking: {str(e)}")

    def _determine_zone(self, x_position):  # Changed from y_position to x_position
        if x_position < self.left_line_x:  #Divides the frame into three zones:
            return 'left'  #'left' → Before entry.
        elif x_position > self.right_line_x: #'detection_zone' → Inside the zone.
            return 'right' #'right' → After exiting.
        else:
            return 'detection_zone'

class PeopleCounterGUI(tk.Tk):
    """
    This class implements a graphical user interface (GUI) for real-time people 
    counting using YOLOv5. It displays video, allows parameter tuning, and 
    shows occupancy statistics.
    """
    def __init__(self):
        """
        Initializes the GUI window, frame grabber, people counter, and all UI elements.
        """
        super().__init__()
        
        # Initialize variables Configure main window properties
        self.title("People Counter")
        self.counter = PeopleCounter()
        self.frame_grabber = None
        self.is_running = False         # Flag to track if processing is active
        self.start_stop_btn = None      # UI Component storage
        self.preset_var = None
        self.preset_combo = None
        self.preset_labels = {}
        self.param_vars = {}
        self.param_scales = {}
        self.stats_labels = {}
        self.last_fps_time = time.time()   # Performance monitoring
        self.fps_frame_count = 0
        self.fps = 0
        
        # Configure main window
        self.geometry("1200x800")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create main container
        self.main_container = ttk.Frame(self)
        self.main_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)
        
        # Configure GUI layout
        self.setup_gui()                
        self.setup_performance_monitor()
        
        # Initialize photo image reference
        self.current_image = None
        # Initialize CUDA memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def setup_gui(self):
        """
        Sets up the main UI components including video display, control buttons, 
        parameter sliders, and statistics panel.
        """
        # Frame for video display
        self.video_frame = ttk.Frame(self.main_container)
        self.video_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky="nsew")
        
        # Label to show video feed
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(expand=True, fill="both")
        
        # Set minimum size for video frame
        self.video_frame.grid_propagate(False)
        self.video_frame.config(width=1280, height=720)
        
        # Control panel
        self.control_frame = ttk.LabelFrame(self.main_container, text="Controls")
        self.control_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Create notebook for tabbed interface
        self.control_notebook = ttk.Notebook(self.control_frame)
        self.control_notebook.pack(expand=True, fill="both", padx=5, pady=5)
        
        # Presets tab
        self.presets_frame = ttk.Frame(self.control_notebook)
        self.control_notebook.add(self.presets_frame, text="Presets")
        
        # Parameters tab
        self.params_frame = ttk.Frame(self.control_notebook)
        self.control_notebook.add(self.params_frame, text="Parameters")
        
        # Setup controls
        self.setup_preset_controls()
        self.setup_parameter_controls()
        
        # Start/Stop button
        self.start_stop_btn = ttk.Button(
            self.control_frame,
            text="Start",
            command=self.toggle_processing
        )
        self.start_stop_btn.pack(pady=10)
        
        # Stats panel
        self.stats_frame = ttk.LabelFrame(self.main_container, text="Statistics")
        self.stats_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        # Labels for displaying occupancy statistics
        stats = ["Current Occupancy", "Remaining Capacity", 
                "Capacity Used", "Total In", "Total Out", "FPS"]
        for stat in stats:
            self.stats_labels[stat] = ttk.Label(self.stats_frame, text=f"{stat}: 0")
            self.stats_labels[stat].pack(pady=2)
            
    def setup_preset_controls(self):
        # Preset selection
        ttk.Label(self.presets_frame, text="Active Preset:").pack(pady=5)
        self.preset_var = tk.StringVar(value=self.counter.preset_names[0])
        self.preset_combo = ttk.Combobox(
            self.presets_frame,
            textvariable=self.preset_var,
            values=self.counter.preset_names,
            state="readonly"
        )
        self.preset_combo.pack(pady=5)
        self.preset_combo.bind('<<ComboboxSelected>>', self.on_preset_change)
        
        # Preset information display
        self.preset_info = ttk.LabelFrame(self.presets_frame, text="Preset Details")
        self.preset_info.pack(fill="x", padx=5, pady=5)
        
        self.preset_labels = {}
        for param in ["confidence_threshold", "nms_threshold", 
                     "process_every_n_frames", "tracking_distance_threshold",
                     "detection_zone_width"]:  # Changed from detection_zone_height
            self.preset_labels[param] = ttk.Label(self.preset_info, text=f"{param}: 0")
            self.preset_labels[param].pack(pady=2)
        
        self.update_preset_info()
        
    def setup_parameter_controls(self):
        # Parameter adjustment controls
        param_configs = {
            "confidence_threshold": ("Confidence Threshold", 0.1, 1.0, 0.1),
            "nms_threshold": ("NMS Threshold", 0.1, 1.0, 0.1),
            "process_every_n_frames": ("Frame Skip", 1, 10, 1),
            "tracking_distance_threshold": ("Tracking Distance", 50, 200, 25),
            "detection_zone_width": ("Zone Width", 50, 200, 25)  # Changed from zone_height
        }
        
        for param, (display_name, min_val, max_val, step) in param_configs.items():
            self.create_parameter_control(param, display_name, min_val, max_val, step)
            
        # Control buttons
        ttk.Button(
            self.params_frame,
            text="Apply Changes",
            command=self.apply_parameter_changes
        ).pack(pady=10)
        
        ttk.Button(
            self.params_frame,
            text="Reset to Preset Defaults",
            command=self.reset_parameters
        ).pack(pady=5)
        
    def create_parameter_control(self, param_name, display_name, min_val, max_val, step):
        frame = ttk.Frame(self.params_frame)
        frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(frame, text=display_name).pack(side="left")
        
        self.param_vars[param_name] = tk.DoubleVar(
            value=getattr(self.counter, param_name, 0)
        )
        
        scale = ttk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            variable=self.param_vars[param_name],
            orient="horizontal"
        )
        scale.pack(side="left", fill="x", expand=True, padx=5)
        self.param_scales[param_name] = scale
        
        value_label = ttk.Label(frame, width=8)
        value_label.pack(side="left")
        
        def update_label(*args):
            value_label.config(
                text=f"{self.param_vars[param_name].get():.2f}"
            )
        
        self.param_vars[param_name].trace_add("write", update_label)
        update_label()
        
    def setup_performance_monitor(self):
        """
        Initializes the system performance monitoring panel to track CPU and 
        memory usage.
        """
        self.process = psutil.Process(os.getpid())                                  # Get the current process
        self.perf_frame = ttk.LabelFrame(self.main_container, text="Performance")   # Performance frame
        self.perf_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        self.cpu_label = ttk.Label(self.perf_frame, text="CPU Usage: 0%")           # CPU & memory usage labels
        self.cpu_label.pack(side="left", padx=10)
        
        self.memory_label = ttk.Label(self.perf_frame, text="Memory Usage: 0 MB")
        self.memory_label.pack(side="left", padx=10)
        
    def update_performance_stats(self):
        """
        Periodically updates CPU and memory usage statistics.
        """
        if not self.is_running:
            return
            
        cpu_percent = psutil.cpu_percent()
        memory_usage = self.process.memory_info().rss / 1024 / 1024                 # Convert to MB
        
        self.cpu_label.config(text=f"CPU Usage: {cpu_percent:.1f}%")
        self.memory_label.config(text=f"Memory Usage: {memory_usage:.1f} MB")
        
        self.after(1000, self.update_performance_stats)                             # Refresh every second
        
    def update_preset_info(self):
        preset = self.preset_var.get()
        preset_values = self.counter.PRESETS[preset]
        
        for param, value in preset_values.items():
            if param in self.preset_labels:
                self.preset_labels[param].config(
                    text=f"{param}: {value}"
                )
                
    def apply_parameter_changes(self):
        for param, var in self.param_vars.items():
            setattr(self.counter, param, var.get())
        logging.info("Applied custom parameter changes")
        
    def reset_parameters(self):
        preset = self.preset_var.get()
        self.counter._apply_preset(preset)
        self.update_preset_info()
        
        for param, value in self.counter.PRESETS[preset].items():
            if param in self.param_vars:
                self.param_vars[param].set(value)
                
        logging.info(f"Reset parameters to preset: {preset}")
        
    def on_preset_change(self, event=None):
        preset = self.preset_var.get()
        self.counter._apply_preset(preset)
        self.update_preset_info()
        logging.info(f"Changed preset to: {preset}")
        
    def update_frame(self):
        """
        Reads the latest frame, processes it, and updates the video display.
        """
        if not self.is_running:
            return
            
        try:
            ret, frame = self.frame_grabber.read()
            if ret:
                # Process frame with CUDA optimization if available
                if torch.cuda.is_available():
                    with torch.cuda.device(0):
                        processed_frame = self.counter.process_frame(frame)         # Apply YOLO processing
                else:
                    processed_frame = self.counter.process_frame(frame)
                
                # Resize frame to fit display area while maintaining aspect ratio
                display_width = self.video_frame.winfo_width()
                display_height = self.video_frame.winfo_height()
                
                h, w = processed_frame.shape[:2]
                aspect_ratio = w / h
                
                new_width = display_width
                new_height = int(display_width / aspect_ratio)
                
                if new_height > display_height:
                    new_height = display_height
                    new_width = int(display_height * aspect_ratio)
                
                # Resize frame
                resized_frame = cv2.resize(processed_frame, (new_width, new_height), 
                                         interpolation=cv2.INTER_LINEAR)
                
                # Convert to PhotoImage for Tkinter display
                img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB) 
                img = Image.fromarray(img)
                photo = ImageTk.PhotoImage(image=img)
                
                # Update video label and keep reference
                self.video_label.configure(image=photo)
                self.current_image = photo
                
                # Update statistics & FPS 
                self.update_stats()
                
            # Schedule next frame update with adaptive delay based on processing time
            self.after(max(1, int(1000/30)), self.update_frame)
            
        except Exception as e:
            logging.error(f"Error updating frame: {str(e)}")
            self.after(100, self.update_frame)
            
    def update_stats(self):
        """
        Updates the displayed statistics including occupancy and FPS.
        """
        stats = {
            "Current Occupancy": self.counter.current_occupancy,
            "Remaining Capacity": self.counter.max_capacity - self.counter.current_occupancy,
            "Capacity Used": f"{(self.counter.current_occupancy / self.counter.max_capacity * 100):.1f}%",
            "Total In": self.counter.people_in,
            "Total Out": self.counter.people_out,
            "FPS": f"{self.calculate_fps():.1f}"
        }
        
        for stat, value in stats.items():
            self.stats_labels[stat].config(text=f"{stat}: {value}")
            
    def calculate_fps(self):
        """
        Computes and returns the FPS based on processed frames.
        """
        current_time = time.time()
        if not hasattr(self, 'last_fps_time'):
            self.last_fps_time = current_time
            self.fps_frame_count = 0
            return 0
            
        self.fps_frame_count += 1
        if current_time - self.last_fps_time > 1:
            self.fps = self.fps_frame_count / (current_time - self.last_fps_time)
            self.fps_frame_count = 0
            self.last_fps_time = current_time
            
        return getattr(self, 'fps', 0)
        
    def toggle_processing(self):
        """
        Starts or stops the frame processing based on the current state.
        """
        if not self.is_running:
            self.start_processing()
        else:
            self.stop_processing()
            
    def start_processing(self):
        """
        Initializes the frame grabber and starts processing frames.
        """
        rtsp_url = "rtsp://admin:Nimda@2024@192.168.7.75:554/media/video1"
        self.frame_grabber = OptimizedFrameGrabber(rtsp_url).start()
        self.is_running = True
        self.start_stop_btn.config(text="Stop")
        self.update_frame()
        self.update_performance_stats()
        
    def stop_processing(self):
        """
        Stops frame grabbing and processing.
        """
        self.is_running = False
        if self.frame_grabber:
            self.frame_grabber.stop()
        self.start_stop_btn.config(text="Start")
        
    def on_closing(self):
        """
        Handles the window close event by stopping all processes gracefully.
        """
        self.stop_processing()
        self.quit()

from flask import Flask, jsonify
import threading

# Initialize Flask API
app = Flask(__name__)

# API Route to get real-time mess occupancy data
@app.route('/api/mess_occupancy', methods=['GET'])
def get_mess_data():
    """
    Flask API route that provides real-time mess occupancy data.
    Returns the maximum capacity and current occupancy percentage.
    """
    global people_counter
    if people_counter:
        return jsonify({
            "max_capacity": people_counter.max_capacity,
            "capacity_percentage": (people_counter.current_occupancy / people_counter.max_capacity) * 100,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Adds timestamp
        })
    else:
        return jsonify({"error": "PeopleCounter not initialized"}), 500

# Function to run Flask API in a separate thread
def run_flask():
    """
    Starts the Flask server on a separate thread to avoid blocking the GUI.
    The server listens on all network interfaces at port 5000.
    """
    app.run(host="0.0.0.0", port=5000, debug=False)

# Start Flask in a separate thread so it doesn’t block the main script
flask_thread = threading.Thread(target=run_flask)
flask_thread.daemon = True      # Allows script to exit cleanly
flask_thread.start()

if __name__ == "__main__":
    try:
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Clear CUDA cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 🔹 Initialize GUI and Fetch `people_counter` Instance
        app = PeopleCounterGUI()

         # ✅ Retrieve the `people_counter` Instance from GUI
        global people_counter
        people_counter = app.counter  # Use the existing instance from GUI

        # Start GUI main loop
        app.mainloop()

    except Exception as e:
        logging.error(f"Application error: {str(e)}")   # Log critical errors
        traceback.print_exc()   # Print full stack trace for debugging