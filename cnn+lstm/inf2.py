import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from mean import get_mean, get_std
from PIL import Image
import cv2
from datasets.ucf101 import load_annotation_data
from datasets.ucf101 import get_class_labels
from model import generate_model
from utils import AverageMeter
from opts import parse_opts
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
import albumentations
import numpy as np

aug = albumentations.Compose([
    albumentations.Resize(224, 224),
    ])


def resume_model(opt, model):
    """ Resume model 
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    checkpoint = torch.load(opt.resume_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()


def predict(clip, model):
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    spatial_transform = Compose([
        Scale((224, 224)),
        #Scale(int(opt.sample_size / opt.scale_in_test)),
        #CornerCrop(opt.sample_size, opt.crop_position_in_test),
        ToTensor(opt.norm_value), norm_method
    ])
    if spatial_transform is not None:
        # spatial_transform.randomize_parameters()
        clip = [spatial_transform(img) for img in clip]

    clip = torch.stack(clip, dim=0)
    clip = clip.unsqueeze(0)
    with torch.no_grad():
        print(clip.shape)
        outputs = model(clip)
        outputs = F.softmax(outputs)
    print(outputs)
    scores, idx = torch.topk(outputs, k=1)
    mask = scores > 0.6
    preds = idx[mask]
    return preds


if __name__ == "__main__":
    opt = parse_opts()
    print(opt)
    data = load_annotation_data(opt.annotation_path)
    class_to_idx = get_class_labels(data)
    print(class_to_idx)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    model = generate_model(opt, device)

    # model = nn.DataParallel(model, device_ids=None)
    # print(model)
    
    if opt.resume_path:
        resume_model(opt, model)
        opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
        opt.std = get_std(opt.norm_value)
        model.eval()

        cap = cv2.VideoCapture('baseball.mp4')
        if (cap.isOpened() == False):
            print('Error while trying to read video. Plese check again...')
# get the frame width and height
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
# define codec and create VideoWriter object
        out = cv2.VideoWriter('out', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))

        while(cap.isOpened()):
    # capture each frame of the video
            ret, frame = cap.read()
            if ret == True:
                model.eval()
                with torch.no_grad():
                    # conver to PIL RGB format before predictions
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    pil_image = aug(image=np.array(pil_image))['image']
                    pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
                    pil_image = torch.tensor(pil_image, dtype=torch.float).cuda()
                    pil_image = pil_image.unsqueeze(0)
                    
                    preds = predict(pil_image, model)
                    
                
                cv2.putText(frame, idx_to_class[preds.item()], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
                cv2.imshow('image', frame)
                out.write(frame)
                # press `q` to exit
                if cv2.waitKey(27) & 0xFF == ord('q'):
                    break
            else: 
                break
# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
        # cam = cv2.VideoCapture(
        #     'C:/Action/lstm/baseball.mp4')
        # clip = []
        # frame_count = 0
        # while True:
        #     ret, img = cam.read()
        #     if frame_count == 16:
        #         print(len(clip))
        #         preds = predict(clip, model)
        #         draw = img.copy()
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         if preds.size(0) != 0:
        #             print(idx_to_class[preds.item()])
        #             cv2.putText(draw, idx_to_class[preds.item(
        #             )], (100, 100), font, .5, (255, 255, 255), 1, cv2.LINE_AA)
        #             cv2.imshow('window', draw)
        #             cv2.waitKey(1)
        #         frame_count = 0
        #         clip = []

        #     #img = cv2.resize(img, (224,224))
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #     #img = Image.fromarray(img.astype('uint8'), 'RGB')
        #     img = Image.fromarray(img)
        #     clip.append(img)
        #     frame_count += 1