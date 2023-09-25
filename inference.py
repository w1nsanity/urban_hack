import numpy as np
import tensorflow as tf
import cv2
import os
import os


arr = os.listdir('images/')

PATH_TO_MODEL_DIR = 'model'
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

def process_image(image_data):
    image_np = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    return image

def main():
    input_path = "images/4_04_03_05_672687-2023-09-09_35352.jpg"
    image = cv2.imread(input_path)
    
    h, w, _ = image.shape
    
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() 
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    labels = detections['detection_classes']
    boxes = detections['detection_boxes']
    scores = detections['detection_scores']

    output_path = f'output/{input_path.split("/")[-1].split(".")[0]}.txt'
    
    
    with open(output_path, 'w') as f:
        for (label, score, [ymin, xmin, ymax, xmax]) in zip(labels, scores, boxes):
            print(f'{label} {round(score, 2):.2f} {int(np.round(xmin * w, 2))} {int(np.round(ymin * h))} {int(np.round(xmax * w))} {int(np.round(ymax * h))}')
            f.write(f'{label} {round(score, 2):.2f} {int(np.round(xmin * w, 2))} {int(np.round(ymin * h))} {int(np.round(xmax * w))} {int(np.round(ymax * h))}')
            f.write('\n')
        f.close()


if __name__ == '__main__':
    main()