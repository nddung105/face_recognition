import cv2
import os


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


def extract_face(path_input, path_output, detector):
    '''
        path_input: path folder include all data
        path_output: path folder include all output data
        detector: object MTCNN's detector 
    '''
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
            cut_face(img, path_output + '/' + folder + '/' + file_image,
                     detector)


def cut_face(img, detector,path_output=None):
    '''
        img: image data
        path_output: path save face file 
        detector: object MTCNN's detector 
    '''
    result,_ = detector.detect_faces(img)  # Get result from detect faces MTCNN
    faces = []  # List all face in image
    bboxs = []
    if len(result):
        for j in range(len(result)):
            # Get face
            bounding_box = result[j]
            bbox = map(int,(bounding_box[1], bounding_box[0], bounding_box[3],
                    bounding_box[2]))
            (startY, startX, endY, endX) = bbox
            minX, maxX = min(startX, endX), max(startX, endX)
            minY, maxY = min(startY, endY), max(startY, endY)
            face = img[minY:maxY, minX:maxX].copy()
            # Save and append face
            # if face:
            if path_output != None:
                cv2.imwrite(path_output, face)
            faces.append(face)
            bboxs.append(bbox)
    return faces, bboxs


if __name__ == '__main__':
    # from detect_face import MTCNN
    # img = cv2.imread(
    #     '/Users/thuongto30/StudyHust/NhapMonAI/Code/get_data/output/Lương Trọng Trí/0.jpg'
    # )
    # detector = MTCNN(
    #     '/Users/thuongto30/StudyHust/NhapMonAI/Code/mtcnn/mtcnn/data/mtcnn_weights.npy'
    # )
    # face, _ = cut_face(img=img, detector=detector, path_output='./1.jpg')
    path_video = './data_face/'
    list_folder = os.listdir(path_video)
    for folder in list_folder:
        if not os.path.exists('./output/' + folder):
            os.mkdir('./output/' + folder)
        file_video = os.listdir(path_video + folder)[0]
        if folder == 'Lại Ngọc Thăng Long':
            print('----->',folder)
            read_video(path_video+folder+'/'+file_video,'./output/' + folder+'/')
        else:
            print('----->',folder)
            read_video(path_video+folder+'/'+file_video,'./output/' + folder+'/',rotate=True)