import torch
import torchvision.transforms as transforms
import PIL.Image
import cv2
import torch2trt
import trt_pose.coco
from trt_pose.parse_objects import ParseObjects
import pprint

COCO_CATEGORY = {
        "supercategory": "person", 
        "id": 1, 
        "name": "person", 
        "keypoints": [
            "nose", 
            "left_eye", 
            "right_eye", 
            "left_ear", 
            "right_ear", 
            "left_shoulder", 
            "right_shoulder", 
            "left_elbow", 
            "right_elbow", 
            "left_wrist", 
            "right_wrist", 
            "left_hip", 
            "right_hip", 
            "left_knee", 
            "right_knee", 
            "left_ankle", 
            "right_ankle", 
            "neck"
        ], 
        "skeleton": [
            [16, 14], 
            [14, 12], 
            [17, 15], 
            [15, 13], 
            [12, 13], 
            [6, 8], 
            [7, 9], 
            [8, 10], 
            [9, 11], 
            [2, 3], 
            [1, 2], 
            [1, 3], 
            [2, 4], 
            [3, 5], 
            [4, 6], 
            [5, 7], 
            [18, 1], 
            [18, 6], 
            [18, 7], 
            [18, 12], 
            [18, 13]
        ]
}


TOPOLOGY = trt_pose.coco.coco_category_to_topology(COCO_CATEGORY)


class PosePre(object):
    
    def __init__(self, shape=(224, 224), dtype=torch.float32, device=torch.device('cuda')):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(device).type(dtype)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(device).type(dtype)
        
    def __call__(self, image):
        with torch.no_grad():
            image = cv2.resize(image, self.shape)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = PIL.Image.fromarray(image)
            image = transforms.functional.to_tensor(image).to(self.device).type(self.dtype)
            image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
            image = image[None, ...]
        return image
 

class PoseEngine(object):

    def __init__(self, path):
        self.module = torch2trt.TRTModule()
        self.module.load_state_dict(torch.load(path))


    def __call__(self, tensor):
        cmap, paf = self.module(tensor)
        return cmap, paf


class PosePost(object):
    
    def __init__(self, *args, **kwargs):
        self.parse_objects = ParseObjects(TOPOLOGY, *args, **kwargs)
        
    def __call__(self, cmap, paf):
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf)
        return counts, objects, peaks
 

class PoseDraw(object):

    def __init__(self, joint_color=(0, 255, 0), link_color=(100, 100, 100)):
        self.joint_color = joint_color
        self.link_color = link_color

    def _draw_point_names(self, image, point, point_name):
        (x, y) = point
        blue = (240, 180, 60)
        thickness = 2
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(image, point_name, (x-20, y+10), font, 2, thickness=thickness, color=blue)  
        #print('{} ({}) : ({},{})'.format(j, COCO_CATEGORY['keypoints'][j], x, y))


    def _draw_rect(self, img, rect, color, width, text_u1=None, text_u2=None, text_d1=None, text_d2=None, fill=False, lowerbar=False):

        # # Following colors are supported:
        # if color == 'brown':
        #     outline_color = (77, 195, 255)
        #     text_color = (77, 195, 255)
        #     fill_color = (77, 195, 255)
        # elif color == 'pink':
        #     outline_color = (182, 84, 231)
        #     text_color = (182, 84, 231)
        #     fill_color = (131, 59, 236)
        # elif color == 'yellow':
        #     outline_color = (79, 244, 255)
        #     text_color = (55, 250, 250)
        #     fill_color = (79, 244, 255)
        # elif color == 'blue':
        #     outline_color = (230, 72, 0)
        #     text_color = (230, 72, 0)
        #     fill_color = (230, 72, 0)
        # elif color == 'orange':
        #     outline_color = (0, 140, 255)
        #     text_color = (0, 140, 255)
        #     fill_color = (0, 140, 255)
        # elif color == 'red':
        #     outline_color = (49, 60, 224)
        #     text_color = (49, 60, 224)
        #     fill_color = (49, 60, 224)
        # else:
        #     assert "Color {} not supported".format(color)
        #     outline_color = (200, 200, 200)
        #     text_color = (200, 200, 200)
        #     fill_color = (200, 200, 200)

        if color == 'brown':
            outline_color = (42, 42, 165)
            text_color = (42, 42, 165)
            fill_color = (42, 42, 165)
            lowerbar_color = (42, 42, 165)
        elif color == 'pink':
            outline_color = (182, 84, 231)
            text_color = (182, 84, 231)
            fill_color = (131, 59, 236)
            lowerbar_color = (131, 59, 236)
        elif color == 'yellow':
            outline_color = (55, 250, 250)
            text_color = (55, 250, 250)
            fill_color = (55, 250, 250)
            lowerbar_color = (55, 250, 250)
        elif color == 'blue':
            outline_color = (240, 120, 0)
            text_color = (240, 120, 0)
            fill_color = (240, 120, 0)
            lowerbar_color = (240, 120, 0)
        elif color == 'green':
            green = (120, 255, 60)
            outline_color = green
            text_color = green
            fill_color = green
            lowerbar_color = green
        elif color == 'orange':
            outline_color = (25, 140, 255)
            text_color = (25, 140, 255)
            fill_color = (25, 140, 255)
            lowerbar_color = (25, 140, 255)
        elif color == 'red':
            outline_color = (49, 60, 255)
            text_color = (49, 60, 255)
            fill_color = (49, 60, 255)
            lowerbar_color = (49, 60, 255)
        else:
            assert "Color {} not supported".format(color)
            outline_color = (200, 200, 200)
            text_color = (200, 200, 200)
            fill_color = (200, 200, 200)
            lowerbar_color = (200, 200, 200)

        # If lowerbar is set, just show text in white.
        if lowerbar is True:
            if color == 'yellow':
                text_color = (255, 0, 0)
            elif color == 'green':
                text_color = (255, 0, 0)
            else:
                text_color = (255, 255, 255)

        font_size = .75
        font_type = cv2.FONT_HERSHEY_PLAIN
        thickness = 1
        x1, y1, x2, y2 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
        h, w, _ = img.shape
        cv2.rectangle(img, (x1, y1), (x2, y2), outline_color, width)

        if fill is True:
            alpha = 0.3
            overlay = img.copy()
            overlay[y1:y2, x1:x2] = fill_color
            cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

        if lowerbar is True:
            alpha = 0.8
            lowerbar_img = img.copy()
            lowerbar_img[y2-30:y2, x1:x2] = lowerbar_color
            cv2.addWeighted(lowerbar_img, alpha, img, 1 - alpha, 0, img)

        if text_u1 is not None:
            box_size, _ = cv2.getTextSize(text_u1, font_type, font_size, thickness)
            txt_size = box_size[0]
            txt_loc = (int(x1 + (x2 - x1)/2 - txt_size/2), y1 + 12)
            cv2.putText(img, text_u1, txt_loc, font_type, font_size, text_color, thickness)

        if text_u2 is not None:
            box_size, _ = cv2.getTextSize(text_u2, font_type, font_size, thickness)
            txt_size = box_size[0]
            txt_loc = (int(x1 + (x2 - x1)/2 - txt_size/2), y1 + 22)
            cv2.putText(img, text_u2, txt_loc, font_type, font_size, text_color, thickness)

        if text_d1 is not None:
            box_size, _ = cv2.getTextSize(text_d1, font_type, font_size, thickness)
            txt_size = box_size[0]
            txt_loc = (int(x1 + (x2 - x1)/2 - txt_size/2), y2 - 15)
            cv2.putText(img, text_d1, txt_loc, font_type, font_size, text_color, thickness)

        if text_d2 is not None:
            box_size, _ = cv2.getTextSize(text_d2, font_type, font_size, thickness)
            txt_size = box_size[0]
            txt_loc = (int(x1 + (x2 - x1)/2 - txt_size/2), y2 - 5)
            cv2.putText(img, text_d2, txt_loc, font_type, font_size, text_color, thickness)



    def _draw_bounding_box(self, image, points):

        if len(points) <= 4:
            return

        print('Points:')
        print(points)
        print('len(points): {}'.format(len(points)))
        coords = []
        for point in points:
            x = point['coords'][0]
            y = point['coords'][1]
            #print('Point: ({},{})'.format(x, y))
            coords.append((x, y))
        print('coords:')
        print(coords)
        xs = [coords[idx][0] for idx in range(len(coords))]
        ys = [coords[idx][1] for idx in range(len(coords))]

        num = len(xs)
        limit = int(num * 0.20)
        if limit <= 0:
            limit = 1

        print('xs: {}'.format(xs))
        print('ys: {}'.format(ys))

        xs = sorted(xs)
        x_min = sum(xs[:limit])/float(limit)
        x_max = sum(xs[-1*limit:])/float(limit)
        ys = sorted(ys)
        y_min = int(0.8 * sum(ys[:limit])/float(limit))
        y_max = int(1.1 * sum(ys[-1*limit:])/float(limit))

        # Draw bounding box.
        self._draw_rect(image, 
                        (x_min, y_min, x_max, y_max),
                        'green',
                        2,
                        text_u1='Person',
                        fill=True)
                        
    def __call__(self, image, object_counts, objects, normalized_peaks):
        topology = TOPOLOGY
        joint_color = self.joint_color
        link_color = self.link_color
        height = image.shape[0]
        width = image.shape[1]
        
        K = topology.shape[0]
        count = int(object_counts[0])
        for i in range(count):
            
            obj = objects[0][i]
            # filter no-neck
            if obj[-1] < 0:
                continue
                

            for k in range(K):
                c_a = topology[k][2]
                c_b = topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)
                    cv2.line(image, (x0, y0), (x1, y1), link_color, 2)
                    
            C = obj.shape[0]
            points = []
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    cv2.circle(image, (x, y), 3, joint_color, -1)
                    self._draw_point_names(image, (x,y), COCO_CATEGORY['keypoints'][j])
                    points.append({
                        'point_idx' : j,
                        'point_name' : COCO_CATEGORY['keypoints'][j],
                        'coords' : (x, y)
                        })
            

            self._draw_bounding_box(image, points)

