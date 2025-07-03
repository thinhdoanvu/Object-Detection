import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform

# Danh sách nhãn COCO tùy chỉnh
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

# Hàm dự đoán đối tượng
def predict(input_tensor, model, device, detection_threshold=0.9):
    outputs = model(input_tensor)
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    return np.int32(boxes), classes, labels, indices

# Hàm reshape lại đầu ra backbone
def fasterrcnn_reshape_transform(x):
    target_size = x['pool'].size()[-2:]
    activations = []
    for key, value in x.items():
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    return torch.cat(activations, axis=1)

# Main
if __name__ == '__main__':
    # image_path = "images/14_00135_.jpg" # r2000
    image_path = "images/IP010000217.jpg" # ip102

    image = Image.open(image_path).convert("RGB")
    image = image.resize((640, 640))
    image_float_np = np.array(image).astype(np.float32) / 255.0

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval().to(device)

    # Dự đoán và lấy nhãn + bounding box
    boxes, classes, labels, indices = predict(input_tensor, model, device, detection_threshold=0.9)

    # Thiết lập Grad-CAM
    target_layers = [model.backbone]
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]

    cam = EigenCAM(model, target_layers, reshape_transform=fasterrcnn_reshape_transform)
    grayscale_cam = cam(input_tensor, targets=targets)[0]

    # Chuẩn hóa CAM và áp dụng colormap
    grayscale_cam = (grayscale_cam - np.min(grayscale_cam)) / (np.max(grayscale_cam) - np.min(grayscale_cam))
    colormap = cm.get_cmap('jet')
    colored_cam = colormap(grayscale_cam)[..., :3]  # Bỏ kênh alpha

    # Kết hợp CAM với ảnh gốc
    overlay = 0.5 * image_float_np + 0.5 * colored_cam
    overlay = np.clip(overlay, 0, 1)

    # Vẽ hình + colorbar
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(overlay)
    ax.axis('off')

    cbar = plt.colorbar(cm.ScalarMappable(cmap='jet'), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['0', '64', '128', '192', '255'])

    # Lưu ảnh
    # plt.savefig('outputs_r2000/faster_rcnn.jpg', bbox_inches='tight', pad_inches=0.1) # r2000
    plt.savefig('outputs_ip102/faster_rcnn.jpg', bbox_inches='tight', pad_inches=0.1) # ip102

    plt.show()
    plt.close()
