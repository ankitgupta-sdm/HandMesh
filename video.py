import cv2
import torch
import os.path as osp
import numpy as np
from cmr.models.cmr_pg import CMR_PG
from cmr.models.cmr_g import CMR_G
from cmr.models.mobrecon_densestack import MobRecon
from utils.read import spiral_tramsform
from options.base_options import BaseOptions
from utils.vis import registration, map2uv, inv_base_tranmsform, base_transform, tensor2array
from utils.draw3d import save_a_image_with_mesh_joints, draw_2d_skeleton
from utils.read import save_mesh
from cmr.datasets.FreiHAND.kinematics import mano_to_mpii
from termcolor import cprint
import pickle

video_path = 'hand_video.MP4'

device = torch.device('cpu')
j_regressor = np.zeros([21, 778])
std = torch.tensor(0.20)

work_dir = osp.dirname(osp.realpath(__file__))
seq_length = [9, 9, 9, 9]
ds_factors = [2, 2, 2, 2]
dilation = [1, 1, 1, 1]

args = BaseOptions().parse()
args.size = 128
args.exp_name = 'mobrecon_spconv'
args.backbone = 'DenseStack'
args.resume = 'mobrecon_densestack.pt'
args.dataset = 'FreiHAND'
args.out_channels= [32, 64, 128, 256]
args.dsconv = False
args.model = 'mobrecon'
args.seq_length = [9, 9, 9, 9]

args.out_dir = osp.join(work_dir, './cmr/out', args.dataset, args.exp_name)
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')

template_fp = osp.join(work_dir, './template/template.ply')
transform_fp = osp.join(work_dir, './template/transform.pkl')
model = None

faces = []

# Function to preprocess a frame (this will depend on your model's requirements)
def preprocess_frame(frame, size=(128, 128)):
    resize_frame = cv2.resize(frame[..., ::-1], size)  # Resize frame
    cv2.imwrite('test.png', resize_frame)
    frame_tensor = torch.tensor(resize_frame).permute(2, 0, 1).float().unsqueeze(0)
    return frame_tensor, resize_frame


def visualize_and_process(frame_tensor, out, frame):
    frame = cv2.resize(frame, (args.size, args.size))
    out_frame = frame
    mask_pred = out.get('mask_pred')
    focal_length = args.size  # This is a simplification; you might want a different value
    center = args.size // 2
    # Estimated K
    K = np.array([
        [focal_length, 0, center],
        [0, focal_length, center],
        [0, 0, 1]
    ])

    if mask_pred is not None:
        mask_pred = (mask_pred[0] > 0.3).cpu().numpy().astype(np.uint8)

        mask_pred = cv2.resize(mask_pred, (frame_tensor.size(3), frame_tensor.size(2)))
        try:
            contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.sort(key=cnt_area, reverse=True)
            poly = contours[0].transpose(1, 0, 2).astype(np.int32)
        except:
            poly = None
    else:
        mask_pred = np.zeros([frame_tensor.size(3), frame_tensor.size(2)])
        poly = None
    pred = out['mesh_pred'][0] if isinstance(out['mesh_pred'], list) else out['mesh_pred']
    vertex = (pred[0].cpu() * std.cpu()).numpy()
    uv_pred = out['uv_pred']
    if uv_pred.ndim == 4:
        uv_point_pred, uv_pred_conf = map2uv(uv_pred.cpu().numpy(), (frame_tensor.size(2), frame_tensor.size(3)))
    else:
        uv_point_pred, uv_pred_conf = (uv_pred * args.size).cpu().numpy(), [None,]
    vertex, align_state = registration(vertex, uv_point_pred[0], j_regressor, K, args.size, uv_conf=uv_pred_conf[0], poly=poly)

    vertex2xyz = mano_to_mpii(np.matmul(j_regressor, vertex))
    out_frame = save_a_image_with_mesh_joints(
        frame, 
        mask_pred, 
        poly, 
        K, 
        vertex, 
        faces[0], 
        uv_point_pred[0], 
        vertex2xyz,
        osp.join(work_dir, 'demo', 'output' + '_plot.jpg'),
        0,
        True
    )
    
    # out_frame = draw_2d_skeleton(frame, uv_point_pred[0])

    return out_frame


def load_model():
    global model, j_regressor, faces
    spiral_indices_list, down_transform_list, up_transform_list, tmp = spiral_tramsform(transform_fp, template_fp, ds_factors, seq_length, dilation)

    faces = tmp['face']
    for i in range(len(up_transform_list)):
        up_transform_list[i] = (*up_transform_list[i]._indices(), up_transform_list[i]._values())

    model = MobRecon(args, spiral_indices_list, up_transform_list)
    # Loading checkpoints
    epoch = 0
    if args.resume:
        if len(args.resume.split('/')) > 1:
            model_path = args.resume
        else:
            model_path = osp.join(args.checkpoints_dir, args.resume)
        checkpoint = torch.load(model_path, map_location='cpu')
        if checkpoint.get('model_state_dict', None) is not None:
            checkpoint = checkpoint['model_state_dict']
        model.load_state_dict(checkpoint)
        epoch = checkpoint.get('epoch', -1) + 1
        cprint('Load checkpoint {}'.format(model_path), 'yellow')
    model = model.to(device)

    model.eval()  # Set the model to evaluation mode

    # Loading Mano
    with open(osp.join(work_dir, './template/MANO_RIGHT.pkl'), 'rb') as f:
        mano = pickle.load(f, encoding='latin1')
    j_regressor = np.zeros([21, 778])
    j_regressor[:16] = mano['J_regressor'].toarray()
    for k, v in {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}.items():
        j_regressor[k, v] = 1

def main():
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # or use 'XVID' if MP4V doesn't work
    out = cv2.VideoWriter('output.MP4', fourcc, 20.0, (args.size * 5, args.size))

    load_model()

    # Processing video
    while True:
        ret, frame = cap.read()
        # frame = cv2.imread('64_img.jpg')
        if not ret:
            break  # Break the loop if there are no frames to read

        # Preprocess the frame
        # frame_tensor, resize_frame = preprocess_frame(frame, (args.size, args.size))
        resize_frame = base_transform(frame, size=args.size)
        frame_tensor = torch.from_numpy(resize_frame)
        frame_tensor = frame_tensor.unsqueeze(0)

        # Perform inference (adapt this based on your actual model input/output)
        with torch.no_grad():
            prediction = model(frame_tensor)
            # Process the model's predictions here
            vis_frame = visualize_and_process(frame_tensor, prediction, frame)

        # Here you might want to visualize the predictions on the frame
        # For example, drawing bounding boxes or keypoints

        # Display the frame (optional)
        out.write(vis_frame)
        cv2.imwrite('output.png', vis_frame)
        # break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()


main()