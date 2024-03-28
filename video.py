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

device = torch.device('cpu')

# Function to preprocess a frame (this will depend on your model's requirements)
def preprocess_frame(frame, size=(128, 128)):
    resize_frame = cv2.resize(frame[..., ::-1], size)  # Resize frame
    cv2.imwrite('test.png', resize_frame)
    frame_tensor = torch.tensor(resize_frame).permute(2, 0, 1).float().unsqueeze(0)
    return frame_tensor, resize_frame


def visualize_and_process(frame_tensor, out, frame):
    out_frame = frame
    mask_pred = out.get('mask_pred')
    print(mask_pred)
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
    # vertex
    pred = out['mesh_pred'][0] if isinstance(out['mesh_pred'], list) else out['mesh_pred']
    # vertex = (pred[0].cpu() * self.std.cpu()).numpy()
    uv_pred = out['uv_pred']

    uv_point_pred, uv_pred_conf = (uv_pred * args.size).cpu().numpy(), [None,]
    # uv_point_pred, uv_pred_conf = map2uv(uv_pred.cpu().numpy(), (frame_tensor.size(2), frame_tensor.size(3)))
    print(uv_point_pred)
    # vertex, align_state = registration(vertex, uv_point_pred[0], self.j_regressor, data['K'][0].cpu().numpy(), args.size, uv_conf=uv_pred_conf[0], poly=poly)

    # vertex2xyz = mano_to_mpii(np.matmul(self.j_regressor, vertex))
    # xyz_pred_list.append(vertex2xyz)
    # verts_pred_list.append(vertex)
    # out_frame = save_a_image_with_mesh_joints(
    #     inv_base_tranmsform(frame[0].cpu().numpy())[:, :, ::-1], 
    #     mask_pred, 
    #     poly, 
    #     data['K'][0].cpu().numpy(), 
    #     vertex, 
    #     self.faces[0], 
    #     uv_point_pred[0], 
    #     vertex2xyz,
    #     osp.join(args.out_dir, 'eval', str(step) + '_plot.jpg'),
    #     0,
    #     True
    # )
    
    out_frame = draw_2d_skeleton(frame, uv_point_pred[0])

    return out_frame



# Load the video
video_path = 'hand_video.MP4'
cap = cv2.VideoCapture(video_path)


# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or use 'XVID' if MP4V doesn't work
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (128, 128))

work_dir = osp.dirname(osp.realpath(__file__))
seq_length = [9, 9, 9, 9]
ds_factors = [3.5, 3.5, 3.5, 3.5]
dilation = [1, 1, 1, 1]


args = BaseOptions().parse()
args.size = 128
args.exp_name = 'mobrecon_spconv'
args.backbone = 'DenseStack'
args.resume = 'mobrecon_densestack.pt'
args.dataset = 'FreiHAND'
args.out_channels= [32, 64, 128, 256]

args.out_dir = osp.join(work_dir, './cmr/out', args.dataset, args.exp_name)
args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')

template_fp = osp.join(work_dir, './template/template.ply')
transform_fp = osp.join(work_dir, './template/transform.pkl')
print(template_fp)
spiral_indices_list, down_transform_list, up_transform_list, tmp = spiral_tramsform(transform_fp, template_fp, ds_factors, seq_length, dilation)

for i in range(len(up_transform_list)):
    up_transform_list[i] = (*up_transform_list[i]._indices(), up_transform_list[i]._values())

model = MobRecon(args, spiral_indices_list, up_transform_list)

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

while True:
    ret, frame = cap.read()
    # frame = cv2.imread('64_img.jpg')
    if not ret:
        break  # Break the loop if there are no frames to read

    # Preprocess the frame
    frame_tensor, resize_frame = preprocess_frame(frame)
    # img = base_transform(frame, size=args.size)
    # frame_tensor = torch.from_numpy(img)
    # frame_tensor = frame_tensor.unsqueeze(0)

    # Perform inference (adapt this based on your actual model input/output)
    with torch.no_grad():
        prediction = model(frame_tensor)
        # Process the model's predictions here
        vis_frame = visualize_and_process(frame_tensor, prediction, resize_frame)

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
