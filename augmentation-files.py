import albumentations as alb
import numpy as np
import tensorflow as tf
import cv2
import os
import json

def load_image(x): 
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

def plot_images_pre_augment():
    # Avoid OOM errors by setting GPU Memory Consumption Growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True)

    images = tf.data.Dataset.list_files('data/images/*.jpg')
    images.as_numpy_iterator().next()
    images = images.map(load_image)
    images.as_numpy_iterator().next()

    image_generator = images.batch(4).as_numpy_iterator()
    plot_images = image_generator.next()

    # fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    # for idx, image in enumerate(plot_images):
    #     ax[idx].imshow(image) 
    # plt.show()

def augment_images():
    augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), 
                            alb.HorizontalFlip(p=0.5), 
                            alb.RandomBrightnessContrast(p=0.2),
                            alb.RandomGamma(p=0.2), 
                            alb.RGBShift(p=0.2), 
                            alb.VerticalFlip(p=0.5)], 
                            bbox_params=alb.BboxParams(format='albumentations', 
                                                    label_fields=['class_labels']))

    for partition in ['train','test','val']: 
        for image in os.listdir(os.path.join('data', partition, 'images')):
            img = cv2.imread(os.path.join('data', partition, 'images', image))

            coords = [0,0,0.00001,0.00001]
            label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label = json.load(f)

                coords[0] = label['shapes'][0]['points'][0][0]
                coords[1] = label['shapes'][0]['points'][0][1]
                coords[2] = label['shapes'][0]['points'][1][0]
                coords[3] = label['shapes'][0]['points'][1][1]
                coords = list(np.divide(coords, [640,480,640,480]))

            try: 
                for x in range(60):
                    augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                    cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                    annotation = {}
                    annotation['image'] = image

                    if os.path.exists(label_path):
                        if len(augmented['bboxes']) == 0: 
                            annotation['bbox'] = [0,0,0,0]
                            annotation['class'] = 0 
                        else: 
                            annotation['bbox'] = augmented['bboxes'][0]
                            annotation['class'] = 1
                    else: 
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0 


                    with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                        json.dump(annotation, f)

            except Exception as e:
                print(e)


def main():
    augment_images()

if __name__ == "__main__":
    main()