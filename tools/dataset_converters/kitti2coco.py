import os
import cv2

def kitti2yolo():
    classes = dict()
    classes['Cyclist'] = 1
    classes['Pedestrian'] = 1
    classes['Person_sitting'] = 1
    classes['Car'] = 2
    classes['Van'] = 2
    classes['Truck'] = 2

    annotation_root = '/media/042b457e-d855-4182-bc38-8f182e78adce1/KITTI/labels/val_annotations/'
    annotation_list = os.listdir(annotation_root)

    fd_root = '/media/042b457e-d855-4182-bc38-8f182e78adce1/KITTI/videos/validation'
    fd_list = os.listdir(fd_root)

    label_root = '/media/042b457e-d855-4182-bc38-8f182e78adce1/KITTI/labels/validation'
    for fd in fd_list:
        fd_path = os.path.join(label_root, fd)
        os.mkdir(fd_path)

    for annotation in annotation_list:
        annotation_path = os.path.join(annotation_root, annotation)
        
        with open(annotation_path, 'r') as rf:
            while True:
                line = rf.readline()
                if not line:
                    break
                line = line.split(' ')
                
                frame_num = int(line[0])
                frame_num = '{:06d}'.format(frame_num)
                
                cls = line[2]
                if cls not in classes.keys():
                    continue

                frame_path = label_root + '/' + annotation.split('.')[0] + '/' + frame_num + '.txt'
                img_path = frame_path.replace('txt', 'png').replace('labels', 'videos')
                img = cv2.imread(img_path)
                im_h, im_w, _ = img.shape

                left, top, right, bottom = list(map(int, list(map(float, line[6:10]))))
                
                cx = ((left + right) // 2) / im_w
                cy = ((top + bottom) // 2) / im_h
                w = (right - left) / im_w
                h = (bottom - top) / im_h
                
                with open(frame_path, 'a+') as wf:
                    wf.write('{} {} {} {} {}\n'.format(classes[cls] - 1, cx, cy, w, h))

def visualize():
    img_root = '/media/042b457e-d855-4182-bc38-8f182e78adce1/KITTI/Data/VID/training'
    ann_root = '/media/042b457e-d855-4182-bc38-8f182e78adce1/KITTI/labels/training'

    fd_list = os.listdir(img_root)
    for fd in fd_list:
        img_fd_path = os.path.join(img_root, fd)
        ann_fd_path = os.path.join(ann_root, fd)

        img_list = os.listdir(img_fd_path)
        ann_list = os.listdir(ann_fd_path)
        
        img_list.sort()
        ann_list.sort()

        for img in img_list:
            img_path = os.path.join(img_fd_path, img)
            label_path = ann_root + '/' + img_path.split('/')[-2] + '/' + img_path.split('/')[-1].replace('.png', '.txt')
            
            im = cv2.imread(img_path)

            with open(label_path, 'r') as rf:
                while True:
                    line = rf.readline()
                    if not line:
                        break
                    line = line.split(' ')
                    x1, y1, x2, y2 = list(map(int, line[1:]))
                    
                    im = cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)

            cv2.imshow('test', im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def main():
    # visualize()
    kitti2yolo()
                
                

if __name__ == '__main__':
    main()