import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
import sys
import logging
import torch
import torchvision.transforms as transforms
from collections import OrderedDict
from mmcv import Config
from mmpose.models import build_posenet
from mmcv.runner import load_checkpoint

# Increase recursion limit
sys.setrecursionlimit(10000)

class VitPoseWrapper:
    """Wrapper for ViTPose model."""
    def __init__(self, config_path, weights_path, input_size=(192, 256), device=None):
        self.input_width, self.input_height = input_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            # Load configuration and model
            cfg = Config.fromfile(config_path)
            self.model = build_posenet(cfg.model)
            load_checkpoint(self.model, weights_path, map_location=self.device)
            self.model.to(self.device)
            self.model.eval()

            # Transformation pipeline
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            # Keypoint connections (COCO format)
            self.keypoint_connections = [
                (15, 13), (13, 11), (16, 14), (14, 12),  # Limbs
                (11, 12), (5, 11), (6, 12), (5, 6),      # Hips to shoulders
                (5, 7), (6, 8), (7, 9), (8, 10),        # Neck to arms
                (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)  # Face and shoulders
            ]

            logging.info("ViTPose model initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize ViTPose model: {e}")
            raise

    def preprocess_frame(self, frame):
        """Preprocess frame for model input."""
        try:
            original_height, original_width = frame.shape[:2]
            
            # Simple resize to model input size
            frame_resized = cv2.resize(frame, (self.input_width, self.input_height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

            # Store metadata for rescaling predictions back to original size
            img_metas = [{
                'img_shape': (self.input_height, self.input_width, 3),
                'original_shape': (original_height, original_width, 3),
                'scale_factor': np.array([
                    original_width / self.input_width,
                    original_height / self.input_height
                ]),
                'center': np.array([self.input_width // 2, self.input_height // 2]),
                'scale': np.array([1.0, 1.0]),
                'rotation': 0,
                'flip_pairs': None,
                'dataset_idx': 0,
                'image_file': None
            }]
            return input_tensor, img_metas
        except Exception as e:
            logging.error(f"Error preprocessing frame: {e}")
            return None, None

    def process_frame(self, frame):
        """Run inference on a single frame and return keypoints."""
        try:
            input_tensor, img_metas = self.preprocess_frame(frame)
            if input_tensor is None:
                return None

            with torch.no_grad():
                output = self.model(img=input_tensor, img_metas=img_metas, return_loss=False)
                return output['preds'][0]  # Extract keypoints for the first person
        except Exception as e:
            logging.error(f"Error processing frame with ViTPose: {e}")
            return None

class PoseEstimationUI:
    def __init__(self, root,vitpose_weights_path = None, whampose_weight_path = None, fourDHuman_weight_path = None):
        self.root = root
        self.root.title("Pose Estimation Comparison")
        self.vitpose_weights_path = vitpose_weights_path
        self.whampose_weights_path = whampose_weight_path
        self.fourDHuman_weight_path = fourDHuman_weight_path
        # Video state
        self.video_path = None
        self.cap = None
        self.playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.frame_cache = {}  # Cache for loaded frames
        
        # Initialize models
        self.setup_models()
        
        # Create UI elements
        self.create_ui()
        
        # Update timer
        self.update_interval = 30  # milliseconds
        self.update_id = None

        self.view_states = {
            'mediapipe': {'show_background': True},
            'vitpose': {'show_background': True},
            'wham': {'show_background': True},
            '4dhuman': {'show_background': True}
        }
        
    def setup_models(self):
        """Initialize pose estimation models"""
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mediapipe_pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5
            )
        except Exception as e:
            logging.error(f"Failed to initialize MediaPipe: {e}")
            self.mediapipe_pose = None
        # Load the model
        # cfg = Config.fromfile('ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochu/vitpose_base_coco_256x192.py')
        # cfg = Config.fromfile('ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_small_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py')
        # cfg = Config.fromfile('ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_base_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py')
        # cfg = Config.fromfile('ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_large_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py')

        # self.vitpose = build_posenet(cfg.model)
        # load_checkpoint(model, './ViTPose/weights/vitpose_large_coco_aic_mpii.pth', map_location='cpu')
        # load_checkpoint(model, './ViTPose/weights/vitpose+_small.pth', map_location='cpu')
        # load_checkpoint(self.vitpose,'./ViTPose/weights/vitpose+_base.pth',map_location='cpu')
        # try:
        #     load_checkpoint(model,'./ViTPose/weights/vitpose+_large.pth',map_location='cuda')
        # except Exception as e:
        #     load_checkpoint(model,'./ViTPose/weights/vitpose+_large.pth',map_location='cpu')

        # if self.vitpose_weights_path:  # You'll need to define this path
        #     try:
        #         self.vitpose = VitPoseWrapper(self.vitpose_weights_path)
        #         logging.info("VitPose model initialized successfully")
        #     except Exception as e:
        #         logging.error(f"Failed to initialize VitPose: {e}")
        try:
            # Initialize ViTPose
            config_path = 'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_base_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py'
            weights_path = './ViTPose/weights/vitpose+_base.pth'
            self.vitpose_wrapper = VitPoseWrapper(config_path, weights_path)
        except Exception as e:
            logging.error(f"Failed to initialize ViTPose: {e}")
            self.vitpose_wrapper = None
    def create_ui(self):
        """Create the user interface"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Top control panel
        control_frame = ttk.Frame(main_container)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Add view toggle panel
        toggle_frame = ttk.Frame(main_container)
        toggle_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # View toggle buttons
        ttk.Label(toggle_frame, text="View Options:").pack(side=tk.LEFT, padx=5)
        
        def create_toggle_button(model_name):
            var = tk.BooleanVar(value=True)
            btn = ttk.Checkbutton(
                toggle_frame,
                text=f"Show {model_name} Background",
                variable=var,
                command=lambda: self.toggle_view(model_name.lower(), var.get())
            )
            btn.pack(side=tk.LEFT, padx=5)
            return var
            
        self.toggle_vars = {
            'mediapipe': create_toggle_button('MediaPipe'),
            'vitpose': create_toggle_button('VitPose'),
            'wham': create_toggle_button('WHAM'),
            '4DHuman': create_toggle_button('4DHuman')
        }
        
        # Video display area
        display_frame = ttk.Frame(main_container)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Video selection
        self.file_button = ttk.Button(control_frame, text="Open Video", command=self.safe_open_video)
        self.file_button.pack(side=tk.LEFT, padx=5)
        
        # Playback controls
        self.prev_button = ttk.Button(control_frame, text="⏮", command=self.prev_frame)
        self.prev_button.pack(side=tk.LEFT, padx=2)
        
        self.play_button = ttk.Button(control_frame, text="▶", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=2)
        
        self.next_button = ttk.Button(control_frame, text="⏭", command=self.next_frame)
        self.next_button.pack(side=tk.LEFT, padx=2)
        
        # Frame slider
        self.frame_slider = ttk.Scale(control_frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.frame_slider.bind("<ButtonRelease-1>", self.on_slider_release)
        
        # Frame counter
        self.frame_label = ttk.Label(control_frame, text="Frame: 0/0")
        self.frame_label.pack(side=tk.RIGHT, padx=5)
        
        # Video display area
        display_frame = ttk.Frame(main_container)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure grid
        display_frame.grid_columnconfigure((0,1,2,3,4), weight=1)
        display_frame.grid_rowconfigure(0, weight=1)
        
        # Create canvases for video display
        canvas_width = 320
        canvas_height = 240
        
        # Original video
        self.original_canvas = tk.Canvas(display_frame, width=canvas_width, height=canvas_height, bg='black')
        self.original_canvas.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        ttk.Label(display_frame, text="Original").grid(row=1, column=0)
        
        # MediaPipe output
        self.mediapipe_canvas = tk.Canvas(display_frame, width=canvas_width, height=canvas_height, bg='black')
        self.mediapipe_canvas.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        ttk.Label(display_frame, text="MediaPipe").grid(row=1, column=1)
        
        # VitPose output
        self.vitpose_canvas = tk.Canvas(display_frame, width=canvas_width, height=canvas_height, bg='black')
        self.vitpose_canvas.grid(row=0, column=2, padx=5, pady=5, sticky='nsew')
        ttk.Label(display_frame, text="VitPose").grid(row=1, column=2)
        
        # WHAM output
        self.wham_canvas = tk.Canvas(display_frame, width=canvas_width, height=canvas_height, bg='black')
        self.wham_canvas.grid(row=0, column=3, padx=5, pady=5, sticky='nsew')
        ttk.Label(display_frame, text="WHAM").grid(row=1, column=3)

        # 4DHuman output
        self.fourdhuman_canvas = tk.Canvas(display_frame, width=canvas_width, height=canvas_height, bg='black')
        self.fourdhuman_canvas.grid(row=0, column=4, padx=5, pady=5, sticky='nsew')
        ttk.Label(display_frame, text="4DHuman").grid(row=1, column=4)

    def toggle_view(self, model_name, show_background):
        """Toggle between full view and skeleton/mesh only view"""
        self.view_states[model_name]['show_background'] = show_background
        self.update_frame()

    def safe_open_video(self):
        """Safely open video file dialog"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
            )
            if filename:
                self.load_video(filename)
        except Exception as e:
            logging.error(f"Error opening video: {e}")
            tk.messagebox.showerror("Error", f"Failed to open video: {str(e)}")
            
    def load_video(self, filename):
        """Load the selected video file"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            self.cap = cv2.VideoCapture(filename)
            if not self.cap.isOpened():
                raise ValueError("Failed to open video file")
                
            self.video_path = filename
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame = 0
            self.frame_cache.clear()
            
            # Update slider
            self.frame_slider.configure(to=self.total_frames - 1)
            self.frame_slider.set(0)
            
            # Update first frame
            self.update_frame()
            
        except Exception as e:
            logging.error(f"Error loading video: {e}")
            tk.messagebox.showerror("Error", f"Failed to load video: {str(e)}")
            
    def on_slider_release(self, event):
        """Handle slider release event"""
        try:
            self.current_frame = int(self.frame_slider.get())
            self.update_frame()
        except Exception as e:
            logging.error(f"Error updating frame: {e}")
            
    def toggle_play(self):
        """Toggle video playback"""
        if self.cap is None:
            return
            
        self.playing = not self.playing
        self.play_button.configure(text="⏸" if self.playing else "▶")
        
        if self.playing:
            self.play()
        elif self.update_id:
            self.root.after_cancel(self.update_id)
            self.update_id = None
            
    def safe_read_frame(self):
        """Safely read a frame from video"""
        try:
            if self.current_frame in self.frame_cache:
                return True, self.frame_cache[self.current_frame]
                
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            
            if ret:
                self.frame_cache[self.current_frame] = frame
            
            return ret, frame
        except Exception as e:
            logging.error(f"Error reading frame: {e}")
            return False, None
            
    def update_frame(self):
        """Update frame display"""
        if self.cap is None:
            return
            
        ret, frame = self.safe_read_frame()
        if not ret:
            self.playing = False
            return
            
        try:
            # Update displays
            self.update_original_display(frame)
            self.update_mediapipe_display(frame.copy())
            self.update_vitpose_display(frame.copy())
            self.update_wham_display(frame.copy())
            self.update_fourdhuman_display(frame.copy())

            # Update slider and label
            self.frame_slider.set(self.current_frame)
            self.frame_label.configure(text=f"Frame: {self.current_frame}/{self.total_frames-1}")
            
            # Increment frame counter if playing
            if self.playing:
                self.current_frame = (self.current_frame + 1) % self.total_frames
                
        except Exception as e:
            logging.error(f"Error updating displays: {e}")
            
    def prepare_photo(self, frame):
        """Safely convert frame to PhotoImage"""
        try:
            frame = cv2.resize(frame, (320, 240))
            image = Image.fromarray(frame)
            return ImageTk.PhotoImage(image)
        except Exception as e:
            logging.error(f"Error preparing photo: {e}")
            return None
            
    def update_original_display(self, frame):
        """Update original video display"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            photo = self.prepare_photo(frame_rgb)
            if photo:
                self.original_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.original_canvas.photo = photo
        except Exception as e:
            logging.error(f"Error updating original display: {e}")

    def update_vitpose_display(self, frame):
        """Update ViTPose output display with skeleton visualization."""
        try:
            if self.vitpose_wrapper:
                # Create display frame based on view state
                if not self.view_states['vitpose']['show_background']:
                    display_frame = np.zeros(frame.shape, dtype=np.uint8)
                else:
                    display_frame = frame.copy()

                # Run ViTPose inference
                keypoints = self.vitpose_wrapper.process_frame(frame)
                
                if keypoints is not None and len(keypoints) > 0:
                    # Get preprocessing metadata
                    _, img_metas = self.vitpose_wrapper.preprocess_frame(frame)
                    if img_metas is None:
                        return
                        
                    meta = img_metas[0]
                    scale_factor = meta['scale_factor']  # [width_scale, height_scale]

                    # Scale keypoints back to original frame size
                    original_keypoints = []
                    for kp in keypoints:
                        orig_x = int(kp[0] * scale_factor[0])  # Scale x coordinate
                        orig_y = int(kp[1] * scale_factor[1])  # Scale y coordinate
                        original_keypoints.append((orig_x, orig_y, kp[2]))  # Keep confidence score

                    # Draw skeleton
                    confidence_threshold = 0.3
                    for connection in self.vitpose_wrapper.keypoint_connections:
                        pt1_idx, pt2_idx = connection
                        if (original_keypoints[pt1_idx][2] > confidence_threshold and
                                original_keypoints[pt2_idx][2] > confidence_threshold):
                            pt1 = tuple(map(int, original_keypoints[pt1_idx][:2]))
                            pt2 = tuple(map(int, original_keypoints[pt2_idx][:2]))
                            color = (255, 255, 255) if not self.view_states['vitpose']['show_background'] else (0, 255, 0)
                            cv2.line(display_frame, pt1, pt2, color, 2)

                    # Draw keypoints
                    for x, y, conf in original_keypoints:
                        if conf > confidence_threshold:
                            color = (255, 255, 255) if not self.view_states['vitpose']['show_background'] else (255, 0, 0)
                            cv2.circle(display_frame, (int(x), int(y)), 4, color, -1)

                # Convert frame for display
                photo = self.prepare_photo(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                if photo:
                    self.vitpose_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                    self.vitpose_canvas.photo = photo

        except Exception as e:
            logging.error(f"Error updating ViTPose display: {e}")
    def update_wham_display(self, frame):
        """Update WHAM output display with toggle support"""
        try:
            if not self.view_states['wham']['show_background']:
                display_frame = np.zeros(frame.shape, dtype=np.uint8)
            else:
                display_frame = frame.copy()
                
            # Add WHAM processing here
            # Similar to MediaPipe, draw skeleton/mesh on either blank or original background
            
            photo = self.prepare_photo(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            if photo:
                self.wham_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.wham_canvas.photo = photo
                
        except Exception as e:
            logging.error(f"Error updating WHAM display: {e}")

    def update_mediapipe_display(self, frame):
        """Update MediaPipe output display with toggle support"""
        try:
            if self.mediapipe_pose:
                # Create display frame based on toggle state
                if not self.view_states['mediapipe']['show_background']:
                    display_frame = np.zeros(frame.shape, dtype=np.uint8)
                else:
                    display_frame = frame.copy()

                # Process with MediaPipe
                results = self.mediapipe_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.pose_landmarks:
                    if not self.view_states['mediapipe']['show_background']:
                        # White skeleton on black background
                        self.mp_drawing.draw_landmarks(
                            display_frame,
                            results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                                color=(255, 255, 255),
                                thickness=2,
                                circle_radius=2
                            ),
                            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                                color=(255, 255, 255),
                                thickness=2
                            )
                        )
                    else:
                        # Colored skeleton on video
                        self.mp_drawing.draw_landmarks(
                            display_frame,
                            results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS
                        )
                
                # Convert and display the processed frame
                photo = self.prepare_photo(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                if photo:
                    self.mediapipe_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                    self.mediapipe_canvas.photo = photo
                    
        except Exception as e:
            logging.error(f"Error updating MediaPipe display: {e}")

    def update_fourdhuman_display(self, frame):
        """Update 4DHuman output display with toggle support"""
        try:
            if not self.view_states['4dhuman']['show_background']:
                display_frame = np.zeros(frame.shape, dtype=np.uint8)
            else:
                display_frame = frame.copy()
                
            #todo fix 4dhuman
            
            photo = self.prepare_photo(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            if photo:
                self.fourdhuman_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.fourdhuman_canvas.photo = photo
                
        except Exception as e:
            logging.error(f"Error updating 4DHuman display: {e}")
            
    def play(self):
        """Play video with error handling"""
        try:
            if self.playing and self.cap is not None:
                self.update_frame()
                self.update_id = self.root.after(self.update_interval, self.play)
        except Exception as e:
            logging.error(f"Error during playback: {e}")
            self.playing = False
            
    def prev_frame(self):
        """Go to previous frame"""
        if self.cap is None:
            return
        self.current_frame = max(0, self.current_frame - 1)
        self.update_frame()
        
    def next_frame(self):
        """Go to next frame"""
        if self.cap is None:
            return
        self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
        self.update_frame()
        
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.cap is not None:
                self.cap.release()
            if self.update_id:
                self.root.after_cancel(self.update_id)
            self.root.quit()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        root = tk.Tk()
        app = PoseEstimationUI(root, vitpose_weights_path="./ViTPose/weights/vitpose_large_coco_aic_mpii.pth")
        root.protocol("WM_DELETE_WINDOW", app.cleanup)
        root.mainloop()
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()