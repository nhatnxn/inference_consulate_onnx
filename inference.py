import numpy as np
import os
import cv2
from utils import *


def inference(model='.onnx', img_path='.jpg'):
    

    provider = os.getenv('PROVIDER', 'CPUExecutionProvider')
    model = onnxruntime.InferenceSession(model, providers=[provider])
    labels, out = detect_batch_frame(model, [img_path], image_size=(640,640))
    
    return labels, out

if __name__ == '__main__':

    model = '.onnx'
    img_path = '.jpg'
    labels, out = inference(model=model, img_path=img_path)

    img = cv2.imread(img_path)
    for box in out:
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]), (128,256,0), 2)
    cv2.imwrite('img.jpg', img)