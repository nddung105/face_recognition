import tensorflow as tf
import numpy as np
import argparse
import facenet
import pickle
from sklearn.svm import SVC
from mtcnn.detect_face import MTCNN
from mtcnn.extract_face import cut_face
import cv2

parser = argparse.ArgumentParser()

parser.add_argument(
    '--frozen_model',
    type=str,
    help="Path of FaceNet's frozen file",
    default='./frozen_facenet/20180408-102900/20180408-102900.pb')
parser.add_argument('--svc_model',
                    type=str,
                    help='Path of SVC model classification',
                    default='./weights/classifier.pkl')
parser.add_argument('--mtcnn_model',
                    type=str,
                    help='Directory with weights MTCNN',
                    default='./weights/mtcnn_weights.npy')


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def load_data(img,
              do_random_crop,
              do_random_flip,
              image_size,
              do_prewhiten=True):
    images = np.zeros((1, image_size, image_size, 3))
    if img.ndim == 2:
        img = to_rgb(img)
    if do_prewhiten:
        img = prewhiten(img)
    img = crop(img, do_random_crop, image_size)
    img = flip(img, do_random_flip)
    images[0, :, :, :] = img
    return images


def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1),
                      np.random.randint(-diff, diff + 1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1 - sz2 + v):(sz1 + sz2 + v),
                      (sz1 - sz2 + h):(sz1 + sz2 + h), :]
    return image


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


def facenet_svc(frozen_model,
                svc_model,
                class_names,
                face,
                mtcnn_model=None,
                image_size=160,
                seed=666):
    with tf.Graph().as_default():

        with tf.Session() as sess:

            np.random.seed(seed=seed)
            # Load the model
            facenet.load_model(frozen_model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name(
                "input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name(
                "embeddings:0")
            phase_train_placeholder = tf.get_default_graph(
            ).get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            nrof_images = 1
            emb_array = np.zeros((nrof_images, embedding_size))
            images = load_data(face, False, False, image_size)
            feed_dict = {
                images_placeholder: images,
                phase_train_placeholder: False
            }
            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

            # Classify images
            predictions = svc_model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[
                np.arange(len(best_class_indices)), best_class_indices]

            for i in range(len(best_class_indices)):
                index = best_class_indices[i]
                name = class_names[index]
                score = best_class_probabilities[i]
                print('%4d  %s: %.3f' % (i, name, score))

    return index, score


def main(args):
    frozen_model = args.frozen_model
    svc_model = args.svc_model
    mtcnn_model = args.mtcnn_model

    with open(svc_model, 'rb') as infile:
        (model, class_names) = pickle.load(infile)

    class_names_print = ['Lai Ngoc Thang Long', 'Luong Trong Tri', \
        'Ngo Song Viet Hoang', 'Nguyen Ba Đuc', 'Nguyen Hoang Son',\
            'Nguyen Dinh Dung', 'Nguyen Dinh Hung', 'Tran Quang Ha', 'Vu Minh Thanh']

    detector = MTCNN(mtcnn_model)
    cap = cv2.VideoCapture(0)
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces, bboxs = cut_face(gray, detector)
        if faces:
            for i in range(len(faces)):
                face = faces[i]
                bounding_box = bboxs[i]
                index, score = facenet_svc(frozen_model, model, class_names,
                                           face)
                if score >= 0.5:
                    cv2.rectangle(frame, (bounding_box[0], bounding_box[1]),
                                  (bounding_box[2], bounding_box[3]),
                                  (0, 155, 255), 2)
                    cv2.putText(frame,
                                class_names_print[index] + " : " + str(score),
                                (bounding_box[0], bounding_box[1]),
                                cv2.FONT_ITALIC, 0.8, (0, 255, 0), 1,
                                cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(parser.parse_args())