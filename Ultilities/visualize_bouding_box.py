import cv2
import os
import matplotlib.pyplot as plt

# Path to the image and its corresponding annotation file
image_path = "E:/thanh/ntu_group/thinh/ObjectDetection/datasets/IP102/val/images/IP038000453.jpg"
annotation_path = "E:/thanh/ntu_group/thinh/ObjectDetection/datasets/IP102/val/labels/IP038000453.txt"

# Load the image
image = cv2.imread(image_path)
height, width, _ = image.shape

# Define colors for different class IDs (just using 3 classes here for example)
colors = {
    0: (0, 255, 0),  # Green for class 0
    1: (0, 0, 255),  # Red for class 1
    2: (255, 0, 0)   # Blue for class 2
}

# List of class names (partial Coco names for illustration, you can use the full list)
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

# Read the annotation file
with open(annotation_path, "r") as file:
    for line in file.readlines():
        # Each line: class_id x_center y_center width height
        parts = line.strip().split()
        class_id = int(parts[0])  # Class ID
        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height
        box_width = float(parts[3]) * width
        box_height = float(parts[4]) * height

        # Convert YOLO format to bounding box coordinates
        xmin = int(x_center - box_width / 2)
        ymin = int(y_center - box_height / 2)
        xmax = int(x_center + box_width / 2)
        ymax = int(y_center + box_height / 2)

        # Draw the rectangle and label on the image
        color = colors.get(class_id, (255, 255, 255))  # Default color: white
        class_name = coco_names[class_id]  # Get class name from coco_names
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Convert BGR (OpenCV format) to RGB (Matplotlib format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display and save the image
plt.imshow(image_rgb)
plt.axis('off')  # Remove axes for better visual clarity
plt.savefig('visual_IP038000453_with_labels.jpg', bbox_inches='tight', pad_inches=0)  # Save before showing
plt.show()
plt.close()
