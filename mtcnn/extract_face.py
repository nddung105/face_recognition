import cv2
import os
from detect_face import MTCNN
from scipy import misc
import numpy as np
import argparse
import sys


def read_video(path_file, path_output, rotate=False):
    '''
        path_file: path file mp4
        path_output: path folder save image output
        rotate: 
    '''
    index = 0  # Index for file name image
    cap = cv2.VideoCapture(path_file)
    while (cap.isOpened()):
        ret, frame = cap.read()  # Read frame from Video
        if ret:
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(path_output + str(index) + '.jpg',
                        frame)  # Write image
            print('==== ', index)
            index += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break


def extract_face(path_input, path_output, image_size, margin,
                 path_weights_file):
    '''
        path_input: path folder include all data
        path_output: path folder include all output data
        detector: object MTCNN's detector 
    '''
    detector = MTCNN(path_weights_file)
    folders = os.listdir(path_input)

    # Loop all folders in input
    for folder in folders:
        # If output not folder -> create
        if not os.path.exists(os.path.join(path_output, folder)):
            os.mkdir(os.path.join(path_output, folder))
        # Gat all file image name
        file_images = os.listdir(os.path.join(path_input, folder))
        # Loop all image
        for file_image in file_images:
            print(folder, file_image)
            img = cv2.imread(path_input + '/' + folder + '/' + file_image)
            # Cut face
            cut_face(
                img, detector, image_size, margin, path_output + '/' + folder +
                '/' + file_image.split('.')[0] + '.png')


def cut_face(img, detector, image_size, margin, path_output=None):
    '''
        img: image data
        path_output: path save face file 
        detector: object MTCNN's detector 
    '''
    result, _ = detector.detect_faces(
        img)  # Get result from detect faces MTCNN
    faces = []  # List all face in image
    bboxs = []
    img_size = np.asarray(img.shape)[0:2]
    if len(result):
        for j in range(len(result)):
            # Get face
            bounding_box = result[j]
            # bbox = map(int, (bounding_box[1], bounding_box[0], bounding_box[3],
            #                  bounding_box[2]))
            # (startY, startX, endY, endX) = bbox
            # minX, maxX = min(startX, endX), max(startX, endX)
            # minY, maxY = min(startY, endY), max(startY, endY)
            # face = img[minY:maxY, minX:maxX,:].copy()
            det = bounding_box
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            face = misc.imresize(cropped, (image_size, image_size),
                                 interp='bilinear')
            # Save and append face
            if path_output != None:
                cv2.imwrite(path_output, face)
            faces.append(face)
            bboxs.append(bb)
    return faces, bboxs


def main(args):
    path_input = args.path_input
    path_output = args.path_output
    path_weights_file = args.path_weights_file
    image_size = args.image_size
    margin = args.margin
    extract_face(path_input, path_output, image_size, margin,
                 path_weights_file)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('path_input',
                        type=str,
                        help='Directory with unaligned images.')
    parser.add_argument('path_output',
                        type=str,
                        help='Directory with aligned face thumbnails.')
    parser.add_argument('path_weights_file',
                        type=str,
                        help='Directory with weights MTCNN')
    parser.add_argument('--image_size',
                        type=int,
                        help='Image size (height, width) in pixels.',
                        default=182)
    parser.add_argument(
        '--margin',
        type=int,
        help=
        'Margin for the crop around the bounding box (height, width) in pixels.',
        default=44)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
