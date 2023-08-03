import os
import cv2
import json

def build_images(id2frames):
    fd_root = '/media/042b457e-d855-4182-bc38-8f182e78adce1/KITTI/videos/validation'
    fd_list = os.listdir(fd_root)
    fd_list.sort()

    image_id = 7000
    video_id = 20
    fn2id = dict()
    images = []
    for i, fd in enumerate(fd_list):
        
        fd_path = os.path.join(fd_root, fd)
        frame_list = os.listdir(fd_path)
        frame_list.sort()
        for j, fr in enumerate(frame_list):
            fr_path = os.path.join(fd_path, fr)
            frame_num = int(fr.split('.')[0])

            if frame_num in id2frames[video_id]:
                fn2id[os.path.join(fd, str(frame_num))] = image_id

                img = cv2.imread(fr_path)
                im_h, im_w, _ = img.shape
                
                images.append({
                    'file_name' : fr_path,
                    'height' : im_h,
                    'width' : im_w,
                    'id' : image_id,
                    'frame_id' : frame_num,
                    'video_id' : video_id,
                    'is_vid_train_frame' : True
                })
            else:
                fn2id[os.path.join(fd, str(frame_num))] = image_id

                img = cv2.imread(fr_path)
                im_h, im_w, _ = img.shape
                
                images.append({
                    'file_name' : fr_path,
                    'height' : im_h,
                    'width' : im_w,
                    'id' : image_id,
                    'frame_id' : frame_num,
                    'video_id' : video_id,
                    'is_vid_train_frame' : False
                })

            image_id = image_id + 1
        video_id = video_id + 1
        print('{}/{} building images..!'.format(i + 1, len(fd_list)))

    return fn2id, images

def build_annotations(id2frames, fn2id):
    ann_root = '/media/042b457e-d855-4182-bc38-8f182e78adce1/KITTI/labels/val_annotations'
    ann_list = os.listdir(ann_root)
    ann_list.sort()

    classes = dict()
    classes['Cyclist'] = 1
    classes['Pedestrian'] = 1
    classes['Person_sitting'] = 1
    classes['Car'] = 2
    classes['Van'] = 2
    classes['Truck'] = 2

    ann_id = 35000
    video_id = 20
    tracking_id = 1000
    annotations = []
    for i, ann in enumerate(ann_list):
        num_obj = -1
        ann_path = os.path.join(ann_root, ann)
        with open(ann_path, 'r') as rf:
            while True:
                line = rf.readline()
                if not line:
                    break
                
                line = line.split(' ')
                
                cls = line[2]

                if cls not in classes.keys():
                    continue
                
                if int(line[1]) > num_obj:
                    num_obj = int(line[1])
                frame_num = int(line[0])
                if frame_num in id2frames[video_id]:
                    track_id = int(line[1]) + tracking_id
                    
                    left, top, right, bottom = list(map(int, list(map(float, line[6:10]))))
                    width = right - left
                    height = bottom - top
                    image_id = fn2id[os.path.join(ann.split('.')[0], str(frame_num))]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                    annotations.append({
                        'id' : ann_id,
                        'video_id' : video_id,
                        'image_id' : image_id,
                        'category_id' : classes[cls],
                        'instance_id' : track_id,
                        'bbox' : [
                            left,
                            top,
                            width,
                            height
                        ],
                        'area' : width * height,
                        'iscrowd' : False,
                        'occluded' : -1,
                        'generated' : -1
                    })

                else:
                    track_id = int(line[1]) + tracking_id
                    
                    left, top, right, bottom = list(map(int, list(map(float, line[6:10]))))
                    width = right - left
                    height = bottom - top
                    image_id = fn2id[os.path.join(ann.split('.')[0], str(frame_num))]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                    annotations.append({
                        'id' : ann_id,
                        'video_id' : video_id,
                        'image_id' : image_id,
                        'category_id' : classes[cls],
                        'instance_id' : track_id,
                        'bbox' : [
                            left,
                            top,
                            width,
                            height
                        ],
                        'area' : width * height,
                        'iscrowd' : False,
                        'occluded' : -1,
                        'generated' : -1
                    })

                ann_id = ann_id + 1
            tracking_id = tracking_id + num_obj + 1

        print('{}/{} building annotatinos..!'.format(i + 1, len(ann_list)))
        video_id = video_id + 1
    return annotations

def build_videos():
    fd_root = '/media/042b457e-d855-4182-bc38-8f182e78adce1/KITTI/videos/validation'
    fd_list = os.listdir(fd_root)
    fd_list.sort()

    id2frames = dict()

    videos = []
    video_id = 20
    for i, fd in enumerate(fd_list):
        
        fd_path = os.path.join(fd_root, fd)
        num_frames = len(os.listdir(fd_path))
        # offset = num_frames // 15
        
        vid_train_frames = [75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89]
        # for j in range(0, num_frames):
        #     if j % offset == 0:
        #         vid_train_frames.append(j)

        id2frames[video_id] = vid_train_frames

        videos.append({
            'id' : video_id,
            'name' : fd_path,
            'vid_train_frames' : vid_train_frames
        })

        print('{}/{} building videos..!'.format(i + 1, len(fd_list)))
        video_id = video_id + 1
    return id2frames, videos

def main():
    id2frames, videos = build_videos()
    fn2id, images = build_images(id2frames)
    annotations = build_annotations(id2frames, fn2id)
    
    print(images[-1])
    print(annotations[-1])
    print(videos[-1])

    categories=[{'id' : 1, 'name': 'person'}, {'id' : 2, 'name' : 'car'}, {'id' : 3, 'name' : 'bicycle'}, {'id' : 4, 'name' : 'animal'}]

    result = dict()
    result['images'] = images
    result['annotations'] = annotations
    result['videos'] = videos
    result['categories'] = categories

    with open('kitti_vid_val.json', 'w') as wf:
        json.dump(result, wf)

if __name__ == '__main__':
    main()