import json
from glob import glob
import os
import sys
from tf_tpu_models.official.detection.utils.imat2020.mask import convert_to_coco_rle
import PIL.Image


def generate_coco_annotations_from_images(images_dir: str):
    images = []
    annotations = []
    for i, file_path in enumerate(sorted(glob(os.path.join(images_dir, '*.jpg')))):
        image = PIL.Image.open(file_path)

        images.append({
            'id': i + 1,
            'width': image.width,
            'height': image.height,
            'file_name': os.path.basename(file_path),
        })

        annotations.append({
            'id': i + 1,
            'image_id': i + 1,
            'category_id': 1,
            'segmentation': convert_to_coco_rle([1, 1], image.height, image.width),
            'area': 1,
            'bbox': [0, 0, 1, 1],
            'iscrowd': 0,
            'attribute_ids': [],
        })

        if i % 1000 == 0:
            print(i)

    return {
        'info': {
            'num_attributes': 294,
        },
        'images': images,
        'categories': [{
            'id': 1,
            'name': '',
            'supercategory': '',
        }],
        'annotations': annotations,
    }


if __name__ == '__main__':
    images_dir = sys.argv[1]
    output_path = sys.argv[2]

    res = generate_coco_annotations_from_images(images_dir)
    with open(output_path, 'w') as f:
        json.dump(res, f)

