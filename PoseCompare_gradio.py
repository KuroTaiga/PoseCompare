import os
import cv2
import torch
import numpy as np
import mediapipe as mp
import gradio as gr
import logging
from mmcv import Config
from mmpose.models import build_posenet
from mmcv.runner import load_checkpoint

import mmcv
from mmpose.apis import (extract_pose_sequence, get_track_id,
                         inference_pose_lifter_model,inference_top_down_pose_model,
                         init_pose_model,
                         process_mmdet_results, vis_3d_pose_result)
try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
from mmpose.datasets import DatasetInfo

import torchvision.transforms as transforms
from FDHumans.hmr2.models import load_hmr2, DEFAULT_CHECKPOINT

class VitPoseWrapper:
    def __init__(self, config_path, weights_path, input_size=(192, 256), device=None):
        self.input_width, self.input_height = input_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def process_frame(self, frame):
        # Preprocess frame
        frame_resized = cv2.resize(frame, (self.input_width, self.input_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

        # Process with model
        with torch.no_grad():
            output = self.model(
                img=input_tensor, 
                img_metas=[{
                    'img_shape': (self.input_height, self.input_width, 3),
                    'scale_factor': np.array([
                        frame.shape[1] / self.input_width,
                        frame.shape[0] / self.input_height
                    ]),
                    'center':np.array([self.input_width // 2, self.input_height // 2]),
                    'scale': np.array([1.0, 1.0]),
                    'rotation': 0,
                    'flip_pairs': None,
                    'dataset_idx': 0,
                    'image_file': None
                }],
                return_loss=False
            )
            keypoints = output['preds'][0]

        # Draw skeleton
        output_frame = frame.copy()
        for connection in self.keypoint_connections:
            pt1_idx, pt2_idx = connection
            if keypoints[pt1_idx, 2] > 0.3 and keypoints[pt2_idx, 2] > 0.3:
                pt1 = tuple(map(int, keypoints[pt1_idx, :2]))
                pt2 = tuple(map(int, keypoints[pt2_idx, :2]))
                cv2.line(output_frame, pt1, pt2, (0, 255, 0), 2)

        # Draw keypoints
        for kp in keypoints:
            if kp[2] > 0.3:
                cv2.circle(output_frame, (int(kp[0]), int(kp[1])), 4, (255, 0, 0), -1)

        return output_frame


class ViTPoseDemoWrapper:
    def __init__(self,
                 pose_config_path, pose_weights_path, 
                 lift_config_path, lift_weight_path,
                 det_config = "ViTPose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py", det_weight = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth", 
                 device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vitdet_model = init_detector(det_config,det_weight,self.device)
        self.vitpose_model = init_pose_model(config=pose_config_path,checkpoint=pose_weights_path,device=self.device)
        self.pose_det_dataset = self.vitpose_model.cfg.data['test']['type']
        self.vitlift_model  = inference_pose_lifter_model(lift_config_path,lift_weight_path,self.device)
        self.pose_lift_dataset = self.vitlift_model.cfg.data['test']['type']
        self.dataset_info = self.vitlift_model.cfg.data['test'].get('dataset_info',None)
        
    def convert_keypoints(self, keypoints):
        keypoints_new = np.zeros((17, keypoints.shape[1]))
        # pelvis is in the middle of l_hip and r_hip
        keypoints_new[0] = (keypoints[11] + keypoints[12]) / 2
        # thorax is in the middle of l_shoulder and r_shoulder
        keypoints_new[8] = (keypoints[5] + keypoints[6]) / 2
        # head is in the middle of l_eye and r_eye
        keypoints_new[10] = (keypoints[1] + keypoints[2]) / 2
        # spine is in the middle of thorax and pelvis
        keypoints_new[7] = (keypoints_new[0] + keypoints_new[8]) / 2
        # rearrange other keypoints
        keypoints_new[[1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16]] = \
            keypoints[[12, 14, 16, 11, 13, 15, 0, 5, 7, 9, 6, 8, 10]]
        return keypoints_new 
    
    def process_frame(self, frame):
        mmdet_results = inference_detector(self.vitdet_model, frame )
        person_det_results = process_mmdet_results(mmdet_results,1) #looking at the solo person
        pose_det_results, _ = inference_top_down_pose_model(
            self.vitpose_model,
            frame,
            person_det_results,
            bbox_thr=0.9,
            format='xyxy',
            dataset=self.pose_det_dataset,
            return_heatmap=False,
            outputs=None)
        for res in pose_det_results:
            keypoints = res["keypoints"]
            res["keypoints"] = self.convert_keypoints(keypoints)

        pose_lift_results = inference_pose_lifter_model(
            self.vitlift_model,
            pose_results_2d=pose_det_results,
            dataset=self.pose_lift_dataset,
            with_track_id=False
        )
        for _, res in enumerate(pose_lift_results):
            keypoints_3d = res['keypoints_3d']
            # exchange y,z-axis, and then reverse the direction of x,z-axis
            keypoints_3d = keypoints_3d[..., [0, 2, 1]]
            keypoints_3d[..., 0] = -keypoints_3d[..., 0]
            keypoints_3d[..., 2] = -keypoints_3d[..., 2]
            res['keypoints_3d'] = keypoints_3d

        
        return vis_3d_pose_result(
            self.vitlift_model,
            result=pose_lift_results,
            img=frame,
            dataset_info=self.dataset_info
        )
    
class FourDHumanWrapper:
    def __init__(self, checkpoint_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self.model, self.model_cfg = load_hmr2(self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def process_frame(self, frame):
        frame_resized = cv2.resize(frame, (256, 256))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB) / 255.0
        input_tensor = torch.tensor(frame_rgb.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            batch = {"img": input_tensor}
            output = self.model(batch)

        # Render mesh
        vertices = output["pred_vertices"][0].cpu().numpy()
        camera_params = output["pred_cam"][0].cpu().numpy()
        
        return self.render_mesh(frame, vertices, camera_params)

    def render_mesh(self, frame, vertices, camera_params):
        s, tx, ty = camera_params
        img_h, img_w = frame.shape[:2]
        
        projected_vertices = vertices[:, :2] * s + np.array([tx, ty])
        projected_vertices[:, 0] = (projected_vertices[:, 0] + 1) * img_w / 2.0
        projected_vertices[:, 1] = img_h - (1-projected_vertices[:, 1]) * img_h / 2.0
        
        mesh_frame = frame.copy()
        for v in projected_vertices.astype(int):
            cv2.circle(mesh_frame, tuple(v), 2, (0, 255, 0), -1)
        
        return mesh_frame
class MediapipeWrapper:

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mediapipe_pose= self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5
        )

    def process_frame(self,frame):
        copied_frame = frame.copy()
        mediapipe_result = self.mediapipe_pose.process(cv2.cvtColor(copied_frame, cv2.COLOR_BGR2RGB))
        if mediapipe_result.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                copied_frame,
                mediapipe_result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        return copied_frame

def process_image(image, show_mediapipe=True, show_vitpose=True, show_4dhuman=True):
    if image is None:
        return None, None, None
    
    results = []
    
    # MediaPipe
    if show_mediapipe:
        try:
            mediapipe_frame = mediapipe_model.process_frame(image)
            results.append(mediapipe_frame)
        except Exception as e:
            logging.error(f"Mediapipe error: {e}")
            results.append(None)        
    else:
        results.append(None)
    
    # ViTPose
    if show_vitpose:
        try:
            vitpose_frame = vitpose_model.process_frame(image)
            results.append(vitpose_frame)
        except Exception as e:
            logging.error(f"ViTPose error: {e}")
            results.append(None)
    else:
        results.append(None)
    
    # 4DHuman
    if show_4dhuman:
        try:
            fdh_frame = fdh_model.process_frame(image)
            results.append(fdh_frame)
        except Exception as e:
            logging.error(f"4DHuman error: {e}")
            results.append(None)
    else:
        results.append(None)
    
    return results

def process_video(video_path, show_mediapipe=True, show_vitpose=True, show_4dhuman=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frames = process_image(frame, show_mediapipe, show_vitpose, show_4dhuman)
        valid_frames = [f for f in processed_frames if f is not None]
        if valid_frames:
            combined = np.hstack(valid_frames)
            frames.append(combined)
    
    cap.release()
    
    if not frames:
        return None
    
    # Create output video path
    output_path = "processed_video.mp4"
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    
    for frame in frames:
        out.write(frame)
    out.release()
    
    return output_path


# Create Gradio interface
with gr.Blocks(title="Pose Estimation Comparison") as demo:
    gr.Markdown("# Pose Estimation Model Comparison")
    gr.Markdown("Compare different pose estimation models: MediaPipe, ViTPose, and 4DHuman")
    
    with gr.Tabs():
        with gr.TabItem("Image Input"):
            with gr.Row():
                image_input = gr.Image(type="numpy", label="Input Image")
                
            with gr.Row():
                mp_checkbox = gr.Checkbox(label="Show MediaPipe", value=True)
                vp_checkbox = gr.Checkbox(label="Show ViTPose", value=True)
                fdh_checkbox = gr.Checkbox(label="Show 4DHuman", value=True)
                
            with gr.Row():
                image_output_mp = gr.Image(label="MediaPipe Output")
                image_output_vp = gr.Image(label="ViTPose Output")
                image_output_fdh = gr.Image(label="4DHuman Output")
                
            image_button = gr.Button("Process Image")
            
        with gr.TabItem("Video Input"):
            with gr.Row():
                video_input = gr.Video(label="Input Video")
                video_output = gr.Video(label="Processed Video")
                
            with gr.Row():
                mp_checkbox_vid = gr.Checkbox(label="Show MediaPipe", value=True)
                vp_checkbox_vid = gr.Checkbox(label="Show ViTPose", value=True)
                fdh_checkbox_vid = gr.Checkbox(label="Show 4DHuman", value=True)
                
            video_button = gr.Button("Process Video")
    
    # Set up event handlers
    image_button.click(
        process_image,
        inputs=[image_input, mp_checkbox, vp_checkbox, fdh_checkbox],
        outputs=[image_output_mp, image_output_vp, image_output_fdh]
    )
    
    video_button.click(
        process_video,
        inputs=[video_input, mp_checkbox_vid, vp_checkbox_vid, fdh_checkbox_vid],
        outputs=[video_output]
    )

# Initialize models
logging.basicConfig(level=logging.INFO)
# vitpose_config_path = 'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/vitPose+_large_coco+aic+mpii+ap10k+apt36k+wholebody_256x192_udp.py'
# vitpose_weights_path = './checkpoints/vitpose_checkpoint.pth'
vitpose_config_path="ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py"
vitpose_weights_path="https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
vitlift_config_path = "ViTPose/configs/body/3d_kpt_sview_rgb_vid/video_pose_lift/h36m/videopose3d_h36m_243frames_fullconv_supervised_cpn_ft.py "
vitlift_weights_path = "https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised_cpn_ft-88f5abbb_20210527.pth"


# vitpose_model = VitPoseWrapper(vitpose_config_path, vitpose_weights_path, )
vitpose_model = ViTPoseDemoWrapper(pose_config_path= vitpose_config_path, pose_weights_path= vitpose_weights_path, lift_config_path=vitlift_config_path, lift_weight_path= vitlift_weights_path)
fdh_model = FourDHumanWrapper()
mediapipe_model = MediapipeWrapper()


if __name__ == "__main__":
    demo.launch(share=True)