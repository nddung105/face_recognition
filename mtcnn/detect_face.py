from utils import *
import model
import cv2
import tensorflow as tf


class MTCNN:
    def __init__(self, path_weights_file):
        self.path_weights_file = path_weights_file
        self.pnet, self.rnet, self.onet = self.load_model(
            self.path_weights_file)

    def load_model(self, path_weights_file):
        return model.load_model(path_weights_file)

    def detect_faces(self,
                    img,
                    minsize=20,
                    threshold=[0.6, 0.7, 0.7],
                    factor=0.709):
        """Detects faces in an image, and returns bounding boxes and points for them.
        img: input image
        minsize: minimum faces' size
        pnet, rnet, onet: caffemodel
        threshold: threshold=[th1, th2, th3], th1-3 are three steps's threshold
        factor: the factor used to create a scaling pyramid of face sizes to detect in the image.
        """
        total_boxes = np.empty((0, 9))
        points = np.empty(0)
        h = img.shape[0]
        w = img.shape[1]
        m = 12 / minsize
        minl = np.amin([h, w]) * m
        # create scale pyramid
        scales = scale_pyramid(m, minl, factor)

        # first stage
        for scale in scales:
            im_data = scale_image(img, scale)
            img_x = np.expand_dims(im_data, 0)
            img_y = np.transpose(img_x, (0, 2, 1, 3))

            out = self.pnet.predict(img_y)

            out0 = np.transpose(out[0], (0, 2, 1, 3))
            out1 = np.transpose(out[1], (0, 2, 1, 3))

            boxes, _ = generateBoundingBox(out1[0, :, :, 1].copy(),
                                           out0[0, :, :, :].copy(), scale,
                                           threshold[0])

            # inter-scale nms
            pick = nms(boxes.copy(), 0.5, 'Union')
            if boxes.size > 0 and pick.size > 0:
                boxes = boxes[pick, :]
                total_boxes = np.append(total_boxes, boxes, axis=0)

        numbox = total_boxes.shape[0]

        if numbox > 0:
            pick = nms(total_boxes.copy(), 0.7, 'Union')
            total_boxes = total_boxes[pick, :]

            regw = total_boxes[:, 2] - total_boxes[:, 0]
            regh = total_boxes[:, 3] - total_boxes[:, 1]

            qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
            qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
            qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
            qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh

            total_boxes = np.transpose(
                np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))

            total_boxes = rerec(total_boxes.copy())

            total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(
                total_boxes.copy(), w, h)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # second stage
            tempimg = np.zeros((24, 24, 3, numbox))
            for k in range(0, numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                tmp[dy[k] - 1:edy[k],
                    dx[k] - 1:edx[k], :] = img[y[k] - 1:ey[k],
                                               x[k] - 1:ex[k], :]
                if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[
                        0] == 0 and tmp.shape[1] == 0:
                    tempimg[:, :, :, k] = imresample(tmp, (24, 24))
                else:
                    return np.empty()
            tempimg = (tempimg - 127.5) * 0.0078125
            tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
            out = self.rnet.predict(tempimg1)
            out0 = np.transpose(out[0])
            out1 = np.transpose(out[1])
            score = out1[1, :]
            ipass = np.where(score > threshold[1])
            total_boxes = np.hstack([
                total_boxes[ipass[0], 0:4].copy(),
                np.expand_dims(score[ipass].copy(), 1)
            ])
            mv = out0[:, ipass[0]]
            if total_boxes.shape[0] > 0:
                pick = nms(total_boxes, 0.7, 'Union')
                total_boxes = total_boxes[pick, :]
                total_boxes = bbreg(total_boxes.copy(),
                                    np.transpose(mv[:, pick]))
                total_boxes = rerec(total_boxes.copy())

        numbox = total_boxes.shape[0]
        if numbox > 0:
            # third stage
            total_boxes = np.fix(total_boxes).astype(np.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(
                total_boxes.copy(), w, h)
            tempimg = np.zeros((48, 48, 3, numbox))
            for k in range(0, numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                tmp[dy[k] - 1:edy[k],
                    dx[k] - 1:edx[k], :] = img[y[k] - 1:ey[k],
                                               x[k] - 1:ex[k], :]
                if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[
                        0] == 0 and tmp.shape[1] == 0:
                    tempimg[:, :, :, k] = imresample(tmp, (48, 48))
                else:
                    return np.empty()
            tempimg = (tempimg - 127.5) * 0.0078125
            tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
            out = self.onet.predict(tempimg1)
            out0 = np.transpose(out[0])
            out1 = np.transpose(out[1])
            out2 = np.transpose(out[2])
            score = out2[1, :]
            points = out1
            ipass = np.where(score > threshold[2])
            points = points[:, ipass[0]]
            total_boxes = np.hstack([
                total_boxes[ipass[0], 0:4].copy(),
                np.expand_dims(score[ipass].copy(), 1)
            ])
            mv = out0[:, ipass[0]]

            w = total_boxes[:, 2] - total_boxes[:, 0] + 1
            h = total_boxes[:, 3] - total_boxes[:, 1] + 1
            points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(
                total_boxes[:, 0], (5, 1)) - 1
            points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(
                total_boxes[:, 1], (5, 1)) - 1
            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
                pick = nms(total_boxes.copy(), 0.7, 'Min')
                total_boxes = total_boxes[pick, :]
                points = points[:, pick]

        return total_boxes, points


if __name__ == '__main__':
    img = cv2.imread(
        '/Users/thuongto30/StudyHust/NhapMonAI/Code/get_data/output/Lương Trọng Trí/0.jpg'
    )
    detector = MTCNN(
        '/Users/thuongto30/StudyHust/NhapMonAI/Code/mtcnn/mtcnn/data/mtcnn_weights.npy'
    )
    box, _ = detector.detect_faces(img)
    cv2.rectangle(img, (int(box[0][0]), int(box[0][1])),
                  (int(box[0][2]), int(box[0][3])), (0, 155, 255), 2)
    # print(dtype(img)
    cv2.imwrite('./out.jpg', img)
    print(box)