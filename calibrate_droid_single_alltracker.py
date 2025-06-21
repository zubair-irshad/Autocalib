
import sys
sys.path.append('./alltracker')  # Add alltracker to the Python path
from demo import forward_video_clean
from nets.net34 import Net
import utils.saveload


import tempfile
import os
import cv2
import time
os.environ['MUJOCO_GL'] = 'egl'
from torch.optim.lr_scheduler import StepLR

tmp_dir = os.path.join(os.getcwd(), 'tmp')
os.makedirs(tmp_dir, exist_ok=True)
tempfile.tempdir = tmp_dir
print(f"Created temporary directory: {tmp_dir}")
os.environ['TMPDIR'] = tmp_dir
if 'notebooks' not in os.listdir(os.getcwd()):
    os.chdir('../')

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

from video_api import initialize_gaussians

from gaussian_renderer import render, render_gradio, render_flow
from scene.cameras import Camera_Pose, Camera
from utils_loc.flow_utils import run_flow_on_images, run_tracker_on_images
import torch
import torch.nn.functional as F
from PIL import Image

from scipy.spatial.transform import Rotation as R
import matplotlib.colors as mcolors
from compute_image_dino_feature import resize_image, interpolate_to_patch_size, resize_tensor
import torchvision.transforms as T
from utils_loc.loss_utils import l1_loss, ssim, cosine_loss
import pickle
# from optimize_multiframe_droid_batch import flow_to_color, iou_loss
from utils_loc.generate_grid_campose import generate_camera_poses, perturb_extrinsic
import json

def flow_to_color(flow):
    dx = flow[:, :, 0]
    dy = flow[:, :, 1]
    angle = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)  # [0, 1]
    mag = np.sqrt(dx**2 + dy**2)
    mag_max = np.max(mag)
    mag_norm = mag / mag_max if mag_max > 0 else mag
    hsv = np.stack([angle, np.ones_like(angle), mag_norm], axis=2)
    rgb = mcolors.hsv_to_rgb(hsv)
    return np.clip(rgb, 0, 1)  # Clip to [0, 1]

def iou_loss(pred, target, smooth=1e-6):
    batch_size = pred.shape[0]
    pred = pred.view(batch_size, -1)  # (batch_size, H*W)
    target = target.view(batch_size, -1)
    intersection = (pred * target).sum(dim=1)  # Sum over pixels per batch
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return (1 - iou).mean()  # Average loss over batch

def visualize_features(features, path):
    """
    Visualize a DINOv2 feature map by reducing its dimensionality to 3 using PCA.

    Args:
        features (np.ndarray): Feature map of shape (H, W, D), where D is the feature dimension (>700).

    Notes:
        - If your feature map is a PyTorch tensor, convert it to NumPy with features.cpu().numpy().
        - If it's a flat array of shape (N, D), reshape it to (H, W, D) first, where N = H * W.
    """
    # Get the dimensions of the feature map
    Hf, Wf, D = features.shape
    print(f"Feature map shape: {H}x{W}x{D}")

    # Reshape to (H*W, D) for PCA
    features_reshaped = features.reshape(Hf * Wf, D)

    # Apply PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_reshaped)
    print(f"Explained variance ratio of 3 components: {pca.explained_variance_ratio_}")

    # Reshape back to (H, W, 3) for visualization
    features_pca_image = features_pca.reshape(Hf, Wf, 3)

    # Normalize each channel to [0,1] for RGB display
    for i in range(3):
        channel = features_pca_image[:, :, i]
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:  # Avoid division by zero
            features_pca_image[:, :, i] = (channel - min_val) / (max_val - min_val)
        else:
            features_pca_image[:, :, i] = 0.5  # Set to mid-range if constant

    # Display the visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(features_pca_image)
    plt.axis('off')
    plt.title("DINOv2 Feature Visualization (PCA to RGB)")
    plt.savefig(path, bbox_inches='tight')



def optimize(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, fwd_flows, fwd_valids, frame_idx):
    all_cameras = []
    all_joint_poses = []

    mujoco_masks = []
    mujoco_images = []
    mujoco_depths = []
    mujoco_feats = []
    mujoco_direct_feats = []
    
    global first_camera
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if render_feat or use_direct_feat:
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        dinov2 = dinov2.to(device)
        dino_transform = T.Compose([
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.5], std=[0.5]),
                                    ])
    os.makedirs('inspection_features', exist_ok=True)
    os.makedirs('inspection_features1', exist_ok=True)
    for c_idx in range(image_list.shape[0]):
        mujoco_image = image_list[c_idx]
        if render_feat:
            dino_image = mujoco_image
            # print(dino_image.shape, 'dino_image')
            dino_image = Image.fromarray(dino_image)
            dino_image.save(f'inspection_features1/dino_image_{c_idx}.png')
            dino_image = resize_image(dino_image, 800)
            dino_image = dino_transform(dino_image)[:3].unsqueeze(0)
            dino_image, target_H, target_W = interpolate_to_patch_size(dino_image, dinov2.patch_size)
            dino_image = dino_image.cuda()
            with torch.no_grad():
                features = dinov2.forward_features(dino_image)["x_norm_patchtokens"][0]
            features = features.cpu().numpy()
            # print(features.shape, 'features')
            features_hwc = features.reshape((target_H // dinov2.patch_size, target_W // dinov2.patch_size, -1))
            features_hwc = torch.from_numpy(features_hwc).float().cuda()
            # print(features_hwc.shape, 'features_hwc')
            mujoco_feats.append(features_hwc)
        mujoco_tensor = torch.from_numpy(mujoco_image).float().cuda().permute(2, 0, 1).to(device, non_blocking=True)
        mujoco_images.append(mujoco_tensor)
        if use_sam_mask:
            mujoco_mask = mask_list[c_idx]
            mujoco_mask_binary = (mujoco_mask > 0).astype(np.uint8)
            mujoco_mask_tensor = torch.from_numpy(mujoco_mask_binary).cuda()
            mujoco_masks.append(mujoco_mask_tensor)
        if have_depth:
            mujoco_depth = depth_list[c_idx]
            mujoco_depth_tensor = torch.from_numpy(mujoco_depth).float().cuda()
            mujoco_depth_tensor = mujoco_depth_tensor
            mujoco_depths.append(mujoco_depth_tensor)
        # Only perturb parameters that are being optimized
        # print("gaussian_params", gaussian_params)
        
        joint_angles = torch.tensor(joints[c_idx])
        intrinsic_rh = intrinsics
        fx = intrinsic_rh[0, 0]
        fy = intrinsic_rh[1, 1]
        fovx = 2 * np.arctan(W / (2 * fx))
        fovy = 2 * np.arctan(H / (2 * fy))
        camera_extrinsic_matrix = extrinsics
        
        camera = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), fovx, fovy,\
                                W, H, joint_pose=joint_angles, zero_init=True).cuda()
        if first_camera is None:
            first_camera = camera  # First camera keeps its initialized parameters
        else:
            # Replace subsequent cameras' parameters with first_camera's
            for name, param in first_camera.named_parameters():
                if name in camera._parameters:
                    camera._parameters[name] = param
        all_cameras.append(camera)
        all_joint_poses.append(joint_angles)

    optimizers = []
    schedulers = []
    if optimize_camera:
        optimizer = torch.optim.Adam([param for camera in all_cameras for param in camera.parameters()], lr=camera_lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
    if optimize_joints:
        optimizers.append(torch.optim.Adam(all_joint_poses, lr=joints_lr))

    for step in tqdm(range(optimization_steps)):
        all_losses = []
        all_images = []

        for optimizer in optimizers:
            optimizer.zero_grad()

        gaussian_tensors = []
        gaussian_masks = []
        gaussian_depths = []
        gaussian_feats = []
        gaussian_flows = []
        i = 0
        for camera, joint_pose in zip(all_cameras, all_joint_poses):
            if FRANKA:
                output = render_gradio(camera, gaussians, background_color, render_features=render_feat)
                if i % 2 == 1:
                    flow_output = render_flow([all_cameras[i], all_cameras[i-1]], gaussians, background_color, render_features=False)
                    # print(flow_output["flow"].shape, 'flow output shape')
                    _, H_flow, W_flow, _ = flow_output["flow"].shape
                    flow = flow_output["flow"].squeeze().reshape((-1, 2)) # [3, H, W] -> [H*W, 3]
                    flow_2d = flow[:, 0:2]
                    flow_2d = flow_2d.reshape((H_flow, W_flow, 2)).permute(2,0,1)
                    # flow_2d = flow_2d.unsqueeze(0) # [1, 2, H, W]
                    gaussian_flows.append(flow_2d)
            
                    # gaussian_2d_pos_curr = flow_output['gaussian_2d_pos_curr']
                    
            new_image = output['render']
            new_depth = output['depth']
            new_pcd = output['pcd']
            # if step == 0 and i in [0, 10, 19]:
            #     # print(f'-----------------------------------saving pcd from {i}')
            #     os.makedirs("pcd", exist_ok=True)
            #     np.save(f"pcd/means3D_{i}.npy", new_pcd)
            i += 1
            new_depth = torch.nan_to_num(new_depth)
            gaussian_tensors.append(new_image)
            gaussian_depths.append(new_depth)
            # mask_tensor = (new_image > 0.01).any(dim=0).float() # no grad
            # thresholded = torch.sigmoid((new_image - 0.01) * 500)
            # mask_tensor = (1 - thresholded.max(dim=0)[0])
            
            thresholded = torch.sigmoid((new_image - 0.01) * 500)
            mask_tensor = thresholded.max(dim=0)[0]
            
            # mask_to_save = (mask_tensor * 255).byte().cpu().numpy()
            # mask_image = Image.fromarray(mask_to_save, mode='L')  # 'L' for grayscale
            # os.makedirs('tmp', exist_ok=True)
            # mask_image.save('tmp/extracted_mask.png')
            gaussian_masks.append(mask_tensor)
        # print(i, 'final')
            
        
        gaussian_tensors = torch.stack(gaussian_tensors)
        gaussian_masks = torch.stack(gaussian_masks)
        gaussian_depths = torch.stack(gaussian_depths)
        if render_feat:
            gaussian_feats = torch.stack(gaussian_feats)
        # gaussian_depths = torch.where(gaussian_depths == 0, 4.0, gaussian_depths)
        if tracking_loss:
            gaussian_flows = torch.stack(gaussian_flows)
        if render_feat:
            mujoco_feats_target = torch.stack(mujoco_feats)

        have_motion = True
        if tracking_loss and step == 0:
            if use_sam_mask:
                mujoco_mask_target = torch.stack(mujoco_masks)
            mujoco_images_target = torch.stack(mujoco_images)
            if have_depth:
                mujoco_depths_target = torch.stack(mujoco_depths)
            # print(mujoco_images_target.shape, 'mujoco images target shape')
            
            fwd_flows = fwd_flows.to(device)
            fwd_valids = fwd_valids.to(device)
                
            # print(fwd_valids)
            fwd_valids = (fwd_valids > 0.3).float()
            # print(fwd_valids)
            # print(fwd_flows.shape, 'forward flow shape')
            motion_mask = (torch.norm(fwd_flows, dim=1, keepdim=True) > 10)
            print(motion_mask.sum(dim=(1, 2, 3)), 'motion mask sum')
            motion_idx = (motion_mask.sum(dim=(1, 2, 3)) >= 10000) & (motion_mask.sum(dim=(1, 2, 3)) < 500000)
            # print()
            motion_idx = motion_idx.float().unsqueeze(1).unsqueeze(1).unsqueeze(1)
            print(motion_idx, motion_idx.shape)
            # if motion_idx.sum() < 1:
            #     print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!valid images < 0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            #     have_motion = False
            #     torch.cuda.empty_cache()
            #     break
            motion_mask = motion_mask.float() # [T-1, 1, H, W]
            
            
        if tracking_loss and step % 50 == 1 and epoch in [0, 2, 3, 4, 5, 6] and save_inspection:
            for i in range(1, min(30, mujoco_images_target.shape[0])):
                fig, axes = plt.subplots(3, 3, figsize=(15, 15))
                
                flow_2d_np_render = gaussian_flows[i-1].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 2]
                step_grid = 10
                new_image_render = gaussian_tensors[i].permute(1, 2, 0).detach().cpu().numpy()
                flow_color_np_render = flow_to_color(flow_2d_np_render)
                H_f, W_f = flow_2d_np_render.shape[:2]
                x_render, y_render = np.meshgrid(np.arange(0, W_f, step_grid), np.arange(0, H_f, step_grid))
                dx_render = flow_2d_np_render[y_render, x_render, 0]
                dy_render = flow_2d_np_render[y_render, x_render, 1]
                
                flow_2d_mujoco = (fwd_flows * fwd_valids * motion_mask)[i-1].permute(1, 2, 0).detach().cpu().numpy()
                new_image_np = (mujoco_images_target[i] / 255).permute(1, 2, 0).detach().cpu().numpy()
                flow_color_mujoco = flow_to_color(flow_2d_mujoco)
                step_grid = 10
                H_f, W_f = flow_2d_mujoco.shape[:2]
                x, y = np.meshgrid(np.arange(0, W_f, step_grid), np.arange(0, H_f, step_grid))
                dx = flow_2d_mujoco[y, x, 0]
                dy = flow_2d_mujoco[y, x, 1]
                
                mask_i = motion_mask[i-1, 0, :, :].float().cpu().numpy()
                flow_valid_mask_i = fwd_valids[i-1, 0, :, :].float().cpu().numpy()
                if have_depth:
                    depth_mask_i = depth_mask[i-1, 0, :, :].float().cpu().numpy()
                    depth_near_mask_i = depth_near_mask[i-1, 0, :, :].float().cpu().numpy()
                    
                    motion_depth_near_mask_i = motion_depth_near_mask[i-1, 0, :, :].float().cpu().numpy()
                # print(flow_valid_mask_i.min(), flow_valid_mask_i.max(), 'flow valid mask min max')
                

                axes[0, 0].imshow(new_image_np)
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(flow_color_mujoco)
                axes[0, 1].set_title('Flow Color Map')
                axes[0, 1].axis('off')

                axes[0, 2].imshow(new_image_np)
                axes[0, 2].quiver(x, y, dx, dy, color='red', angles='xy', scale_units='xy', scale=1, width=0.001)
                axes[0, 2].set_title('Image with Flow Arrows')
                axes[0, 2].axis('off')

                axes[1, 0].imshow(new_image_render)
                axes[1, 0].set_title('Rendered Image')
                axes[1, 0].axis('off')

                axes[1, 1].imshow(flow_color_np_render)
                axes[1, 1].set_title('Rendered Flow Color Map')
                axes[1, 1].axis('off')

                axes[1, 2].imshow(new_image_render)
                axes[1, 2].quiver(x, y, dx_render, dy_render, color='red', angles='xy', scale_units='xy', scale=1, width=0.001)
                axes[1, 2].set_title('Rendered Flow Arrows')
                axes[1, 2].axis('off')

                axes[2, 0].imshow(new_image_np)
                axes[2, 0].imshow(mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[2, 0].set_title('Motion Mask')
                axes[2, 0].axis('off')

                axes[2, 1].imshow(new_image_np)
                axes[2, 1].imshow(mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[2, 1].set_title('Depth Mask')
                axes[2, 1].axis('off')

                axes[2, 2].imshow(new_image_np)
                axes[2, 2].quiver(x, y, dx - dx_render, dy - dy_render, color='red', angles='xy', scale_units='xy', scale=1, width=0.001)
                axes[2, 2].set_title('Image with Flow Arrows')
                axes[2, 2].axis('off')

                # Step 5: Save the figure as an image
                plt.tight_layout()  # Adjusts spacing to minimize white space
                plt.savefig(f'inspection_features1/super_flow_visualization{i}.png', dpi=300)  # Save as PNG with 300 DPI
                plt.close()
    
        image = gaussian_tensors
        mujoco_images_target1 = mujoco_images_target / 255
        normalize = T.Normalize(mean=[0.5], std=[0.5])
        
        if tracking_loss:
            flow_ratio = gaussian_flows.max() / fwd_flows.max()
            flow_loss = (gaussian_flows - fwd_flows).abs()
            flow_loss[:, 0] /= W
            flow_loss[:, 1] /= H
            flow_loss = (flow_loss * fwd_valids * motion_mask).sum() / fwd_valids.sum()
            print(torch.norm(fwd_flows, dim=1, keepdim=True).mean().item(), 'forward flow norm mean', torch.norm(fwd_flows, dim=1, keepdim=True).min().item(), torch.norm(fwd_flows, dim=1, keepdim=True).max().item())
            print(torch.norm(gaussian_flows, dim=1, keepdim=True).mean().item(), 'gaussian flow norm mean', torch.norm(gaussian_flows, dim=1, keepdim=True).min().item(), torch.norm(gaussian_flows, dim=1, keepdim=True).max().item())
            motion_mask = (torch.norm(fwd_flows, dim=1, keepdim=True) > 3)
            # render_motion_mask = (torch.norm(gaussian_flows, dim=1, keepdim=True) > 10)
            norm = torch.norm(gaussian_flows, dim=1, keepdim=True)
            temperature = 0.1  # Adjust this value based on your needs
            render_motion_mask = torch.sigmoid((norm - 10) / temperature)
            
            # mask = motion_mask[0].squeeze().cpu().numpy()
            # # Display the mask as a grayscale image
            # plt.imshow(mask, cmap='gray')
            # plt.title('Motion Mask')
            # plt.axis('off')  # Optional: hide axes for a cleaner visualization
            # plt.savefig('inspection_features1/mask.png', bbox_inches='tight', pad_inches=0)
            # mask = gaussian_masks[0].squeeze().cpu().detach().numpy()
            # # Display the mask as a grayscale image
            # plt.imshow(mask, cmap='gray')
            # plt.title('Motion Mask')
            # plt.axis('off')  # Optional: hide axes for a cleaner visualization
            # plt.savefig('inspection_features1/rendermask.png', bbox_inches='tight', pad_inches=0)
            
            # print(motion_mask.shape, 'motion mask')
            # print(render_motion_mask.shape, 'render mask')
            print('motion_mask sum:', motion_mask.sum().item(), 'render_motion_mask sum:', render_motion_mask.sum().item())
            print('gaussian_mask shape', gaussian_masks.shape, 'motion_mask shape:', motion_mask.shape)
            # visualize motion mask
            ratio_loss = (torch.norm(gaussian_flows, dim=1, keepdim=True).mean() / (torch.norm(fwd_flows, dim=1, keepdim=True).mean() + 1e6) - 1).abs()
            print('ratio_loss:', ratio_loss.item())
            # flow_mask_loss = iou_loss(render_motion_mask.float(), motion_mask.float())
            flow_mask_loss = iou_loss(gaussian_masks[::2, :, :], motion_mask)
            if epoch == 0:
                weight_flow = 1
                weight_mask = 0
            elif epoch == 1:
                weight_flow = 1
                weight_mask = 0.0
            elif epoch >= 2:
                weight_flow = 1
                weight_mask = 0.0
            flow_loss = flow_loss * weight_flow + flow_mask_loss * weight_mask
        
        if vis_features and step % 10 == 0:
            for i in range(len(Optimize_list)):
                image = mujoco_images_target[i].permute(1, 2, 0)
                # print(image.shape, 'image_shape')
                image_np = image.to(torch.uint8)
                image_np = image_np.detach().cpu().numpy()
                image_np = Image.fromarray(image_np)
                # image_np.save(f'inspection_features1/images_{i}.png')
                
                image_render = (gaussian_tensors[i] * 255).permute(1, 2, 0)
                image_np_render = image_render.to(torch.uint8)
                image_np_render = image_np_render.detach().cpu().numpy()
                image_np_render = Image.fromarray(image_np_render)
                image_np_final = Image.fromarray(np.concatenate([image_np_render, image_np], axis=1))
                image_np_final.save(f'inspection_features1/image_render_{i}.png')
                
                image_blend = Image.blend(image_np_render, image_np, alpha=0.5)
                if step == 40 and epoch in [0,2] and i in [0, 10, 19]:
                    image_blend.save(os.path.join(SCENE_PATH, f'blend_epoch_{epoch}_step_{step}_i_{i}.png'))
                
        threshold = 0.5
        gaussian_masks_scaled = (gaussian_masks * 255).byte()  # uint8 format
        # if reg_mask:
        #     gaussian_mask_sum = gaussian_masks.sum()
        #     gaussian_mask_sum = gaussian_mask_sum / gaussian_masks.shape[0]
        #     reg_mask_loss = (10000 - gaussian_mask_sum)
        #     print(reg_mask_loss.item(), 'reg mask loss')

        # Step 3: Convert to NumPy
        gaussian_masks_np = gaussian_masks_scaled.detach().cpu().numpy()
        

        # # Step 4: Save masks as images
        # os.makedirs('masks_inspection', exist_ok=True)
        # for i in range(gaussian_masks_np.shape[0]):
        #     gaussian_filename = f"masks_inspection/gaussian_mask_{i}.png"
        #     cv2.imwrite(gaussian_filename, gaussian_masks_np[i])
        #     if use_sam_mask:
        #         gt_filename = f"masks_inspection/ground_truth_mask_{i}.png"
        #         cv2.imwrite(gt_filename, ground_truth_masks_np[i])
        
        mse_loss = F.mse_loss(mujoco_images_target / 255, gaussian_tensors, reduction='none')
        l2_diff = mse_loss.mean(dim=(1, 2, 3)).mean()
        
        # print(loss_feat, 'loss_feat')
        if use_sam_mask:
            IoU_loss = iou_loss(gaussian_masks, mujoco_mask_target)
            # print(IoU_loss, 'IoU_loss')
            
        
        if have_depth:
            IoU_loss_motion = iou_loss(gaussian_masks[:-1], motion_depth_near_mask)
            depth_loss = F.mse_loss(mujoco_depths_target, gaussian_depths, reduction='none')
            print(depth_loss.shape, 'depth_loss shape', depth_near_mask1.shape)
            depth_diff = (depth_loss * depth_near_mask1).mean()

        # total_loss = l2_diff.sum()
        # total_loss = loss_feat + IoU_loss
        # total_loss = l2_diff.sum() + loss_feat + depth_diff * 1 + IoU_loss * 10
        flow_loss = flow_loss * 1000
        # flow_loss_motion = flo     w_loss_motion * 10
        if have_depth:
            depth_diff = depth_diff * 0.1
            IoU_loss_motion = IoU_loss_motion * 0.01
        # total_loss = flow_loss + flow_loss_motion
        # total_loss = depth_diff * 0.01 + flow_loss
        # if flow_loss == 0:
        #     print('Flow loss is zero, skipping optimization step.')
        #     continue
        total_loss = flow_loss + l2_diff * 0.1
        # total_loss = 0
        if use_direct_feat:
            loss_direct_feat = loss_direct_feat * 1
            total_loss += loss_direct_feat
        print('flow_loss :', flow_loss.item())
        if use_direct_feat:
            print('loss_feat:', loss_direct_feat.item())
        print('rgb loss:', l2_diff.item())
        if have_depth:
            print('depth_diff:', depth_diff.item(),  'IoU_loss_motion :', IoU_loss_motion.item())
        # print('total_loss: ', total_loss.item())
        print('epoch:', epoch, Optimize_list)
        
        # if IoU_loss > 0.85: 
        #     total_loss = l2_diff * 3 + depth_diff * 0.1 + IoU_loss * 10
        # else:
        #     total_loss = l2_diff + depth_diff * 1 + IoU_loss * 10
        # total_loss = 10 * depth_diff + 0.1 * IoU_loss
        total_loss = flow_loss
        total_loss.backward()

        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            scheduler.step()    

    # print('Camera results robot to world transformation:')
    # for i in range(len(all_cameras)):
    #     print(all_cameras[i].robot_to_world())
    #     print(all_cameras[i].get_camera_pose().tolist(), 'camera pose final')
    # print(best_camera.robot_to_world())
    valid_masks = False
    # if flow_loss < 0.5: # TODO: use other metrics
    valid_masks = True
    # yield (best_image, *rounded_final_params)
    # print(all_cameras[0].world_view_transform.transpose(0, 1).detach().cpu().numpy())
    return all_cameras[0].world_view_transform.transpose(0, 1).detach().cpu().numpy(), True
    
@ torch.no_grad()
def compute_loss(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, fwd_flows, fwd_valids):
    all_cameras = []
    all_joint_poses = []
    
    global first_camera
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for c_idx in range(image_list.shape[0]):
        joint_angles = torch.tensor(joints[c_idx])
        intrinsic_rh = intrinsics
        fx = intrinsic_rh[0, 0]
        fy = intrinsic_rh[1, 1]
        fovx = 2 * np.arctan(W / (2 * fx))
        fovy = 2 * np.arctan(H / (2 * fy))
        camera_extrinsic_matrix = extrinsics
        
        camera = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), fovx, fovy,\
                                W, H, joint_pose=joint_angles, zero_init=True).cuda()
        if first_camera is None:
            first_camera = camera  # First camera keeps its initialized parameters
        else:
            # Replace subsequent cameras' parameters with first_camera's
            for name, param in first_camera.named_parameters():
                if name in camera._parameters:
                    camera._parameters[name] = param
        all_cameras.append(camera)
        all_joint_poses.append(joint_angles)

    for step in tqdm(range(optimization_steps)):
        all_losses = []
        all_images = []

        gaussian_masks = []
        gaussian_flows = []
        i = 0
        for camera, joint_pose in zip(all_cameras, all_joint_poses):
            # if i % 2 == 1:
            #     flow_output = render_flow([all_cameras[i], all_cameras[i-1]], gaussians, background_color, render_features=False)
            #     # print(flow_output["flow"].shape, 'flow output shape')
            #     _, H_flow, W_flow, _ = flow_output["flow"].shape
            #     flow = flow_output["flow"].squeeze().reshape((-1, 2)) # [3, H, W] -> [H*W, 3]
            #     flow_2d = flow[:, 0:2]
            #     flow_2d = flow_2d.reshape((H_flow, W_flow, 2)).permute(2,0,1)
            #     mask_tensor = (torch.norm(flow_2d, dim=0, keepdim=True) > 4)
            #     gaussian_masks.append(mask_tensor)
            #     gaussian_flows.append(flow_2d)
            #     # print(mask_tensor.shape, 'mask tensor shape')
                
            output = render_gradio(camera, gaussians, background_color, render_features=False)
            new_image = output['render']
            thresholded = torch.sigmoid((new_image - 0.01) * 500)
            mask_tensor = thresholded.max(dim=0)[0]
            gaussian_masks.append(mask_tensor)
            i += 1
    
        gaussian_masks = torch.stack(gaussian_masks)
        # gaussian_flows = torch.stack(gaussian_flows)
        print(fwd_flows.shape, 'forward flow shape')
        motion_mask = (torch.norm(fwd_flows, dim=1, keepdim=True) > 3)
        # mask = motion_mask[0].squeeze().cpu().numpy()
        # # Display the mask as a grayscale image
        # plt.imshow(mask, cmap='gray')
        # plt.title('Motion Mask')
        # plt.axis('off')  # Optional: hide axes for a cleaner visualization
        # plt.savefig('inspection_features1/mask.png', bbox_inches='tight', pad_inches=0)
        # mask = gaussian_masks[0].squeeze().cpu().detach().numpy()
        # # Display the mask as a grayscale image
        # plt.imshow(mask, cmap='gray')
        # plt.title('Motion Mask')
        # plt.axis('off')  # Optional: hide axes for a cleaner visualization
        # plt.savefig('inspection_features1/rendermask.png', bbox_inches='tight', pad_inches=0)
        
        print(gaussian_masks.shape, 'gaussian masks shape', motion_mask.shape, 'motion mask shape') 
        flow_mask_loss = iou_loss(gaussian_masks[::2, :, :], motion_mask)
        # flow_mask_loss = iou_loss(gaussian_masks[:1, :, :], motion_mask[:1])
        
        # flow_loss = (gaussian_flows - fwd_flows).abs() * motion_mask
        # flow_loss[:, 0] /= W
        # flow_loss[:, 1] /= H
        # flow_loss = (flow_loss).mean()
        # flow_mask_loss = flow_loss

        print('iou_loss :', flow_mask_loss.item())
    return all_cameras[0].world_view_transform.transpose(0, 1).detach().cpu().numpy(), flow_mask_loss.item()

def set_nested_dict(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def get_nested_keys(scene_path):
    """
    Read the metadata JSON file from the scene directory and extract nested keys.
    
    Args:
        scene_path (str): Path to the scene directory (e.g., "droid_extract_whole/scene_2")
    
    Returns:
        list: Nested keys derived from the constructed path (e.g., ['AUTOLab', 'success', ...])
    
    Raises:
        ValueError: If exactly one metadata file is not found in the scene directory
    """
    # Step 1: Find the metadata JSON file
    metadata_files = [f for f in os.listdir(scene_path) 
                      if f.startswith('metadata_') and f.endswith('.json')]
    if len(metadata_files) != 1:
        raise ValueError(f"Expected one metadata file in {scene_path}, found {len(metadata_files)}")
    metadata_file = os.path.join(scene_path, metadata_files[0])
    
    # Step 2: Load the JSON file into a dictionary
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Step 3: Extract the required keys
    lab = metadata['lab']              # e.g., "AUTOLab"
    hdf5_path = metadata['hdf5_path']  # e.g., "success/2023-07-12/Wed_Jul_12_11:47:50_2023/trajectory.h5"
    cam_name = metadata['cam_name']    # e.g., "24400334"
    
    # Step 4: Get the directory part of hdf5_path (remove the file name)
    hdf5_dir = os.path.dirname(hdf5_path)  # e.g., "success/2023-07-12/Wed_Jul_12_11:47:50_2023"
    
    # Step 5: Construct the full path
    full_path = os.path.join(lab, hdf5_dir, cam_name)  
    # e.g., "AUTOLab/success/2023-07-12/Wed_Jul_12_11:47:50_2023/24400334"
    
    # Step 6: Split the path into nested keys using the OS-specific separator
    keys = full_path.split(os.sep)
    # e.g., ['AUTOLab', 'success', '2023-07-12', 'Wed_Jul_12_11:47:50_2023', '24400334']
    
    return keys

"""
python calibrate_bridge_single_alltracker.py --model_path output/widow0 --scene_path /data/group_data/katefgroup/datasets/bridge_chenyu/raw/bridge_data_v1/berkeley/laundry_machine/put_clothes_in_laundry_machine/2022-03-19_14-53-07/raw/traj_group0/traj0
python calibrate_bridge_single_alltracker.py --model_path output/widow0 --scene_path /data/group_data/katefgroup/datasets/bridge_chenyu/raw/bridge_data_v1/berkeley/toykitchen2_room8052/flip_orange_pot_upright_in_sink/2021-06-16_14-13-45/raw/traj_group0/traj0
python calibrate_droid_single_alltracker.py --model_path output/franka_fr3_2f85_highres_finetune_0 --scene_path /data/group_data/katefgroup/datasets/droid_chenyu/droid_extract_whole/scene_2
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', type=str, required=True, help='Path to the scene directory')
    parser.add_argument('--model_path', type=str, default='output/franka_fr3_2f85_highres_finetune_0', help='Path to the scene directory')
    parser.add_argument('--raw_path', type=str, default=None)
    parser.add_argument('--dict_path', type=str, default="/data/group_data/katefgroup/datasets/droid_chenyu/droid_extract_whole/calib_dict.pkl")
    args = parser.parse_args()
    
    print(args.scene_path, 'scene path')
    os.makedirs('inspection_features1', exist_ok=True)
    
    save_inspection = False
    
    # Load or initialize calibration dictionary
    if os.path.exists(args.dict_path):
        with open(args.dict_path, 'rb') as f:
            calib_dict = pickle.load(f)
    else:
        calib_dict = {}

    # Compute relative path and keys from scene_path
    keys = get_nested_keys(args.scene_path)
    
    print('here')
    render_feat = False
    use_direct_feat = False
    use_direct_feat_compute = False
    gaussians, background_color, sample_cameras, kinematic_chain = initialize_gaussians(model_path=args.model_path)
    background_color = torch.zeros((3,)).cuda()

    SCENE_PATH = args.scene_path
    image_folder = os.path.join(SCENE_PATH, 'images0')
    # obs_dict = pickle.load(open(os.path.join(SCENE_PATH, 'obs_dict.pkl'), 'rb'))
    # joints = obs_dict['qpos']
    # states = obs_dict["full_state"]
    
    # grippers_whole = states[:, -1:]
    # fingers_whole = grippers_whole * 0.022 + 0.015
    # fingers_whole = np.concatenate([fingers_whole, -fingers_whole], axis=1)
    # joints_whole = np.concatenate([joints, fingers_whole], axis=1)
    
    grippers_whole = np.load(os.path.join(SCENE_PATH, 'grippers.npy'))
    joints_whole = np.load(os.path.join(SCENE_PATH, 'joints.npy'))
    grippers_whole = np.expand_dims(grippers_whole, axis=1)
    fingers_whole = grippers_whole * np.array([[0.75, -0.4, 0.6, -0.26, 0.75, -0.4, 0.6, -0.26]])
    joints_whole = np.concatenate([joints_whole, fingers_whole], axis=1)
    
    # Initialize intrinsics and extrinsics using droid's original data
    intrinsics = np.load(os.path.join(SCENE_PATH, 'intrinsics.npy'))
    w2c = np.load(os.path.join(SCENE_PATH, 'extrinsics0.npy'))
    
    image_list_whole = []
    image_names = os.listdir(image_folder)
    image_names = sorted(image_names, key=lambda x: int(x.split('.')[0]))
    image_paths = [os.path.join(image_folder, path) for path in image_names]
    print(image_paths, 'image paths')
    for k in range(len(joints_whole)):
        image_path = image_paths[k]
        img = Image.open(image_path)
        img = np.array(img)
        image_list_whole.append(img)
    image_list_whole = np.stack(image_list_whole)
    B, H, W, C = image_list_whole.shape
    
    extrinsics_save_path = os.path.join(SCENE_PATH, 'extrinsics.npy')
    intrinsics_save_path = os.path.join(SCENE_PATH, 'intrinsics.npy')
    vis_features = True
    tracking_loss = True
    use_sam_mask = False
    have_depth = False
    reg_mask = True
    FRANKA = True
    UR5 = False
    ROBOTIQ = True        
        
    sampled_frames = []
    l_max = min(200, len(image_list_whole))
    for i in range(8, l_max-9, 10):
        sampled_frames.append(i)
    
    # image_list_whole = image_list_whole[initial_downsample]
    # joints_whole = joints_whole[initial_downsample]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fwd_flows_whole = torch.zeros((len(sampled_frames), 2, image_list_whole.shape[1], image_list_whole.shape[2]), dtype=torch.float32).to(device)
    fwd_valids_whole = torch.zeros((len(sampled_frames), 1, image_list_whole.shape[1], image_list_whole.shape[2]), dtype=torch.float32).to(device)
    # print(fwd_flows_whole.shape, 'fwd flows whole shape') # torch.Size([24, 2, 480, 640]) fwd flows whole shape
    
    video_tensor_alltracker = torch.from_numpy(image_list_whole).float().permute(0, 3, 1, 2).unsqueeze(0)
    window_len = 16  # Must match the model's expected window length
    alltracker_model = Net(window_len)
    load_dir = 'alltracker/checkpoints/reference_model'  # Path to checkpoint
    utils.saveload.load(None, load_dir, alltracker_model, optimizer=None, scheduler=None, 
                        ignore_load=None, strict=True, verbose=False, weights_only=False)
    alltracker_model.cuda()  # Move to GPU (remove .cuda() if using CPU)
    alltracker_model.eval()  # Set to evaluation mode

    with torch.no_grad():
        for i_ufm, i_frame in enumerate(sampled_frames):
            print(i_ufm, i_frame, 'i ufm and i frame')
            video_curr = video_tensor_alltracker[:, i_frame:i_frame+5, :, :, :].cuda()  # Shape: [B, T, C, H, W]
            
            flow, covisibility = forward_video_clean(video_curr, alltracker_model)
            flow = flow[0, 1, :, :, :] # [B, T, 2, H, W] -> [2, H, W]
            covisibility = covisibility[0, 1, 0, :, :] # [B, T, 2, H, W] -> [H, W]
            print(flow.shape, 'flow shape after alltracker')
            print(covisibility.shape, 'covisibility shape after alltracker')
            # (2, 480, 640) flow shape after ufm
            # (480, 640) covisibility shape after ufm
            # visualize the flow
            flow_2d_mujoco = flow.permute(1, 2, 0).detach().cpu().numpy()
            flow_color_mujoco = flow_to_color(flow_2d_mujoco) # np array
            flow_visual = Image.fromarray((flow_color_mujoco * 255).astype(np.uint8))
            flow_visual.save(f'inspection_features1/alltracker_flow_{i_ufm}.png')
            
            flow_valid = (covisibility > 0.3)
            fwd_flows_whole[i_ufm] = flow
            fwd_valids_whole[i_ufm] = flow_valid
    fwd_flows_whole = fwd_flows_whole.to(device)
    fwd_valids_whole = fwd_valids_whole.to(device)
    
    # Define perturbation ranges
    thetas_x = torch.linspace(-0.2, 0.2, 3)  # 3 steps, ±0.1 radians
    thetas_y = torch.linspace(-0.2, 0.2, 3)
    thetas_z = torch.linspace(-0.2, 0.2, 3)
    dxs = torch.linspace(-0.00, 0.00, 1)     # 3 steps, ±0.05 units
    dys = torch.linspace(-0.00, 0.00, 1)
    dzs = torch.linspace(-0.00, 0.00, 1)
    
    perturbed_extrinsics = perturb_extrinsic(torch.tensor(w2c, dtype=torch.float32), thetas_x, thetas_y, thetas_z, dxs, dys, dzs)
    best_extrinsics = w2c
    best_score = float('inf')
    epoch = 0
    first_camera = None  # Track the first camera to share its parameters
    # print(len(perturbed_extrinsics), 'perturbed extrinsics')
    Optimize_list = []
    for batch_idx, frame_idx in enumerate(sampled_frames):
        Optimize_list.append(frame_idx)
        Optimize_list.append(frame_idx + 1)
    
    cutoff = 4
    offset = 3
    Optimize_list = Optimize_list[offset*2:cutoff*2]
    # [0, 1, 10, 11, 20, 21, 30, 31]
    image_list = image_list_whole[Optimize_list]
    fingers = fingers_whole[Optimize_list]
    joints = joints_whole[Optimize_list]
    fwd_flows = fwd_flows_whole[offset:cutoff]
    fwd_valids = fwd_valids_whole[offset:cutoff]
    
    if True:
        for extrinsic in perturbed_extrinsics:
            print(extrinsic, 'extrinsic')
            extrinsics = extrinsic
            optimize_camera = False
            optimize_joints = False
            camera_lr = 0.0
            joints_lr = 0.0
            optimization_steps = 1
            params, flow_loss_item = compute_loss(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, 
                        fwd_flows, fwd_valids)
            if flow_loss_item < best_score:
                best_score = flow_loss_item
                best_extrinsics = extrinsics
            # time.sleep(1)
        print('best pose: ', best_extrinsics)
        print(best_score, 'best score')

    first_camera = None
    extrinsics = best_extrinsics

    epochs = 3
        
    Optimize_list = []
    for batch_idx, frame_idx in enumerate(sampled_frames):
        Optimize_list.append(frame_idx)
        Optimize_list.append(frame_idx + 1)
    
    Optimize_list = Optimize_list
        
    image_list = image_list_whole[Optimize_list]
    fingers = fingers_whole[Optimize_list]
    joints = joints_whole[Optimize_list]
    fwd_flows = fwd_flows_whole
    fwd_valids = fwd_valids_whole

    for epoch in tqdm(range(epochs)):
        optimize_camera = True
        optimize_joints = False
        camera_lr = 0.01 # 0.001
        camera_lr = camera_lr * (0.5 ** (epoch // 4))
        joints_lr = 0.001
        optimization_steps = 41
        powerful_optimize_dropdown = "Disabled"
        params, valid_optimization = optimize(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, 
                    fwd_flows, fwd_valids, frame_idx)
        torch.cuda.empty_cache()
        print(params)
        np.save(extrinsics_save_path, params)
        
    # Update calib_dict after all epochs
    set_nested_dict(calib_dict, keys, {'w2c': params.tolist(), 'intrinsic': intrinsics.tolist()})
    with open(args.dict_path, 'wb') as f:
        pickle.dump(calib_dict, f)

    print(params, 'params')
    print(intrinsics, 'intrinsics')