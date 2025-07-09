import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ✅ Custom reshape_transform for RetinaNet
def retinanet_reshape_transform(feats):
    if isinstance(feats, dict):
        feats = list(feats.values())  # Trường hợp backbone trả ra dict
    elif isinstance(feats, torch.Tensor):
        feats = [feats]  # Nếu chỉ là một Tensor thì đóng gói lại thành list
    elif not isinstance(feats, (list, tuple)):
        raise TypeError(f'Unexpected type: {type(feats)}')

    target_size = feats[0].shape[-2:]
    upsampled = []
    for f in feats:
        if f.dim() == 4:  # (N, C, H, W)
            upsampled.append(torch.nn.functional.interpolate(f, size=target_size, mode='bilinear', align_corners=False))
    return torch.cat(upsampled, dim=1)



# Tên class (dựa trên IP102 – bạn cần chỉnh lại nếu dùng COCO)
coco_names = ['rice leaf roller', 'rice leaf caterpillar', 'paddy stem maggot', 'asiatic rice borer', 'yellow rice borer',
    'rice gall midge', 'Rice Stemfly', 'brown plant hopper', 'white backed plant hopper', 'small brown plant hopper',
    'rice water weevil', 'rice leafhopper', 'grain spreader thrips', 'rice shell pest', 'grub', 'mole cricket', 'wireworm',
    'white margined moth', 'black cutworm', 'large cutworm', 'yellow cutworm', 'red spider', 'corn borer', 'army worm', 'aphids',
    'Potosiabre vitarsis', 'peach borer', 'english grain aphid', 'green bug', 'bird cherry-oataphid', 'wheat blossom midge',
    'penthaleus major', 'longlegged spider mite', 'wheat phloeothrips', 'wheat sawfly', 'cerodonta denticornis', 'beet fly',
    'flea beetle', 'cabbage army worm', 'beet army worm', 'Beet spot flies', 'meadow moth', 'beet weevil', 'sericaorient alismots chulsky',
    'alfalfa weevil', 'flax budworm', 'alfalfa plant bug', 'tarnished plant bug', 'Locustoidea', 'lytta polita', 'legume blister beetle',
    'blister beetle', 'therioaphis maculata Buckton', 'odontothrips loti', 'Thrips', 'alfalfa seed chalcid', 'Pieris canidia',
    'Apolygus lucorum', 'Limacodidae', 'Viteus vitifoliae', 'Colomerus vitis', 'Brevipoalpus lewisi McGregor', 'oides decempunctata',
    'Polyphagotars onemus latus', 'Pseudococcus comstocki Kuwana', 'parathrene regalis', 'Ampelophaga', 'Lycorma delicatula', 'Xylotrechus',
    'Cicadella viridis', 'Miridae', 'Trialeurodes vaporariorum', 'Erythroneura apicalis', 'Papilio xuthus', 'Panonchus citri McGregor',
    'Phyllocoptes oleiverus ashmead', 'Icerya purchasi Maskell', 'Unaspis yanonensis', 'Ceroplastes rubens', 'Chrysomphalus aonidum',
    'Parlatoria zizyphus Lucus', 'Nipaecoccus vastalor', 'Aleurocanthus spiniferus', 'Tetradacus c Bactrocera minax ', 'Dacus dorsalis(Hendel)',
    'Bactrocera tsuneonis', 'Prodenia litura', 'Adristyrannus', 'Phyllocnistis citrella Stainton', 'Toxoptera citricidus', 'Toxoptera aurantii',
    'Aphis citricola Vander Goot', 'Scirtothrips dorsalis Hood', 'Dasineura sp', 'Lawana imitata Melichar', 'Salurnis marginella Guerr',
    'Deporaus marginatus Pascoe', 'Chlumetia transversa', 'Mango flat beak leafhopper', 'Rhytidodera bowrinii white', 'Sternochetus frigidus',
    'Cicadellidae']
# ✅ Prediction wrapper
def predict(input_tensor, model, device, detection_threshold=0.5):
    with torch.no_grad():
        outputs = model(input_tensor)
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].cpu().numpy()

    boxes, classes, labels, indices = [], [], [], []
    for i in range(len(pred_scores)):
        if pred_scores[i] >= detection_threshold:
            boxes.append(pred_bboxes[i].astype(np.int32))
            classes.append(pred_classes[i])
            labels.append(pred_labels[i])
            indices.append(i)
    return np.int32(boxes), classes, labels, indices

# ✅ Main script
if __name__ == '__main__':
    image_path = "images/IP010000217.jpg"
    image = Image.open(image_path).convert("RGB").resize((640, 640))
    image_float_np = np.array(image).astype(np.float32) / 255.0

    transform = torchvision.transforms.ToTensor()
    input_tensor = transform(image).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)

    # ✅ Load RetinaNet from torchvision
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    model = model.to(device).eval()

    # ✅ Inference to get detection results
    boxes, classes, labels, indices = predict(input_tensor, model, device, detection_threshold=0.9)

    # ✅ Grad-CAM setup
    target_layers = [model.backbone.body.layer4]  # Target ResNet stage 4 (e.g., for deep features)
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]

    cam = EigenCAM(model, target_layers=target_layers, reshape_transform=retinanet_reshape_transform)
    grayscale_cam = cam(input_tensor, targets=targets)[0, :]

    cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)

    # ✅ Display or save
    plt.imshow(cam_image)
    plt.axis('off')
    plt.savefig("retinanet_gradcam.jpg", bbox_inches='tight', pad_inches=0)
    plt.show()
