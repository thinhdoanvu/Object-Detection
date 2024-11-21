import json
import glob
import os
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET


def json_to_text(in_json, out_folder):
    """
    Convert dataset in a JSON file (for MMDet) to text files (for YOLOv4 and YOLOv5)
    Please read the Tutorials to understand the required format for MMDet, YOLOv4, and YOLOv5
    :param in_json: input JSON file
    :param out_folder: output folder storing the text files
    """
    # Read the annotation file
    with open(in_json) as f:
        data = json.load(f)

    # Process the images
    for data_img in data['images']:
        H = data_img['height']
        W = data_img['width']
        idx = data_img['id']

        # Create a new text file for each image
        f = open(out_folder + os.sep + data_img['file_name'].replace('jpg', 'txt'), 'w')

        # Get all annotations of the image
        for ann in data['annotations']:
            if ann['image_id'] == idx:
                bbox = ann['bbox']
                x_center = (bbox[0] + bbox[2] / 2) / W
                y_center = (bbox[1] + bbox[3] / 2) / H
                width = bbox[2] / W
                height = bbox[3] / H
                text = '0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(x_center, y_center, width, height)  # Ship label: 0
                f.write(text)

        f.close()
        print(data_img['file_name'].replace('jpg', 'txt') + ' done')


def text_to_json(in_labels, in_images, out_json):
    """
    Convert dataset in text files (for YOLOv4 and YOLOv5) to a JSON file (for MMdet)
    Please read the Tutorials to understand the required format for MMDet, YOLOv4, and YOLOv5
    :param in_labels: input folder containing the label text files
    :param in_images: input folder containing the image files (just for getting the image size)
    :param out_json: output JSON file
    """
    # Initialize the output JSON file
    data = dict()
    data['type'] = 'instance'
    data['categories'] = [{'supercategory': str('none'),
                           'id': 1,
                           'name': 'ship'}]  # assume that we only have a class of 'ship'
    data['images'] = []
    data['annotations'] = []

    # Initial the number of annotations
    num_annotations = 0

    # Process the text files
    txt_files = glob.glob(in_labels + '/*.txt')
    for k in range(len(txt_files)):
        # Read the image to get the information of width and height
        img = Image.open(in_images + '/' + os.path.basename(txt_files[k]).replace('txt', 'jpg'))
        W, H = img.size

        # Create an new image item and append it to the list
        img_item = dict()
        img_item['id'] = k
        img_item['file_name'] = os.path.basename(txt_files[k]).replace('txt', 'jpg')
        img_item['height'] = H
        img_item['width'] = W
        data['images'].append(img_item)

        # Creates annotation items of the image and append them to the list
        with open(txt_files[k]) as f:
            for line in f:
                line = line.split()
                x_center = float(line[1]) * W  # denormalize to get the actual coordinates in pixels
                y_center = float(line[2]) * H
                width = float(line[3]) * W
                height = float(line[4]) * H
                x = int(x_center - width / 2)
                y = int(y_center - height / 2)

                # Create a new annotation item
                ann_item = dict()
                ann_item['id'] = num_annotations
                ann_item['image_id'] = k
                ann_item['category_id'] = 1  # assume that we only have a class of 'Ship'
                ann_item['bbox'] = [x, y, int(width), int(height)]
                ann_item['area'] = int(width) * int(height)
                ann_item['iscrowd'] = 0
                data['annotations'].append(ann_item)
                num_annotations += 1

        print(os.path.basename(txt_files[k]) + ' done')

    # Write the dictionary to a JSON file
    with open(out_json, 'w') as f:
        json.dump(data, f)


def xml_to_text(xml_folder, txt_folder):
    """
    Convert Pascal VOC XML files to YOLO format text files.

    :param xml_folder: Path to the folder containing the XML files.
    :param txt_folder: Path to the folder where text files will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(txt_folder, exist_ok=True)

    # Get all XML files in the folder
    xml_files = glob.glob(os.path.join(xml_folder, '*.xml'))

    # Define class mappings here if needed
    class_mapping = {
        'D00': 0,  # Map class names to IDs
        'D10': 1,
        'D20': 2,
        'D40': 3
    }

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Get filename
        filename = root.find('filename').text
        # Remove extension to create the text filename
        txt_filename = os.path.join(txt_folder, os.path.splitext(filename)[0] + '.txt')

        # Get image size
        img_size = root.find('size')
        W = int(img_size.find('width').text)
        H = int(img_size.find('height').text)

        # Open text file for writing
        with open(txt_filename, 'w') as f:
            # Iterate through all objects
            for obj in root.findall('object'):
                # Get class label and convert to numeric ID
                class_name = obj.find('name').text
                class_id = class_mapping.get(class_name, -1)  # Default to -1 if not in mapping

                # Get bounding box coordinates
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # Normalize coordinates to YOLO format
                x_center = (xmin + xmax) / 2.0 / W
                y_center = (ymin + ymax) / 2.0 / H
                width = (xmax - xmin) / float(W)
                height = (ymax - ymin) / float(H)

                # Write to text file
                f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n')

        print(f'{filename} converted to YOLO format and saved as {txt_filename}')


def json_to_json(in_folder, out_json):
    """
    Convert the JSON files of the dataset SSDD to a single JSON file (for YOLOv4 and YOLOv5)
    Reference: https://ieeexplore.ieee.org/document/8124934
    :param in_folder:input SSDD folder containing the JSON files
    :param out_json: output JSON file
    """
    # Initialize the output JSON file
    data = dict()
    data['type'] = 'instance'
    data['categories'] = [{'supercategory': str('none'),
                           'id': 1,
                           'name': 'ship'}]  # assume that we only have a class of 'ship'
    data['images'] = []
    data['annotations'] = []

    # Initial the number of annotations
    num_annotations = 0

    # Get all JSON files in the SSDD folder
    json_files = glob.glob(in_folder + '/*.json')

    for k in range(len(json_files)):
        # Read a JSON file
        with open(json_files[k]) as f:
            img = json.load(f)

        # Create an new image item and append it to the list
        img_item = dict()
        img_item['id'] = k
        img_item['file_name'] = img['imagePath']
        img_item['height'] = img['imageHeight']
        img_item['width'] = img['imageWidth']
        data['images'].append(img_item)

        # Creates annotation items of the image and append them to the list
        objects = img['shapes']

        for obj in objects:
            points = obj['points']
            points = np.array(points)  # convert the list to a Numpy array for further computation
            xmax, ymax = points.max(axis=0)
            xmin, ymin = points.min(axis=0)

            # Create a new annotation item
            ann_item = dict()
            ann_item['id'] = num_annotations
            ann_item['image_id'] = k
            ann_item['category_id'] = 1  # assume that we only have a class of 'Ship'
            ann_item['bbox'] = [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]
            ann_item['area'] = int(xmax - xmin) * int(ymax - ymin)
            ann_item['iscrowd'] = 0
            data['annotations'].append(ann_item)
            num_annotations += 1

        print(os.path.basename(json_files[k]) + ' done')

    # Write the dictionary to a JSON file
    with open(out_json, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    # 1. Test JSON to text
    #json_to_text(in_json='./annotations/instances_val2017.json',
    #             out_folder='./labels/val')

    # 2. Test text to JSON
    # text_to_json(in_labels='/data/SAR_ship_detection/datasets/HRSID/train/labels',
    #              in_images='/data/SAR_ship_detection/datasets/HRSID/train/images',
    #              out_json='/data/SAR_ship_detection/datasets/HRSID/train/train.json')

    # 3. Test XML to text
    xml_to_text(xml_folder='./rdd2022/RDD2022_released_through_CRDDC2022/RDD2022/China_Drone/China_Drone/train/annotations/xmls',
                txt_folder='./rdd2022/RDD2022_released_through_CRDDC2022/RDD2022/China_Drone/China_Drone/train/labels')
    # xml_to_text(xml_folder='./BCCD/valid/Annotations',
    #             txt_folder='./BCCD/valid/labels')
    xml_to_text(xml_folder='./rdd2022/RDD2022_released_through_CRDDC2022/RDD2022/China_Drone/China_Drone/test/annotations/xmls',
                txt_folder='./rdd2022/RDD2022_released_through_CRDDC2022/RDD2022/China_Drone/China_Drone/test/labels')

    # 4. Test XML to text
    # json_to_json(in_folder='/data/SAR_ship_detection/datasets/SSDD/annotations',
    #              out_json='/data/SAR_ship_detection/datasets/SSDD/dataset.json')