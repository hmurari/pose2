import torch
import torchvision.transforms as transforms
import PIL.Image
import cv2
import torch2trt
import trt_pose.coco
from trt_pose.parse_objects import ParseObjects
import pprint
import time


# Say max 10 people

norm_frames_cnt = 30
norm_areas = [[0] * norm_frames_cnt] * 10
norm_areas_idx = 0
curr_time = time.time()
prev_time = time.time()
fps = 20
durations = [0] * 10
active = [0] * 10
prev_xs = [0] * 10
prev_ys = [0] * 10

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

    def pixels_to_dist(self, dist):
        if dist < 400:
            return ">20'"
        elif dist < 450:
            return "18'"
        elif dist < 500:
            return "16'"
        elif dist < 550:
            return "14'"
        elif dist < 600:
            return "12'"
        elif dist < 650:
            return "9'"
        elif dist < 700:
            return "6'"
        elif dist < 750:
            return "5'"
        elif dist < 800:
            return "4'"
        else:
            return "3'"

    def pixels_to_dist2(self, area):
        area = area/1000
        area = area*0.9
        if area < 20:
            return "15'"
        elif area < 30:
            return "12'"
        elif area < 40:
            return "10'"
        elif area < 50:
            return "9'"
        elif area < 70:
            return "8'"
        elif area < 80:
            return "7'"
        elif area < 90:
            return "6'"
        elif area < 100:
            return "5'"
        elif area < 150:
            return "4'"
        else:
            return "3'"

    def __init__(self, joint_color=(0, 255, 0), link_color=(120, 255, 60)):
        self.joint_color = joint_color
        self.link_color = link_color

    def _draw_point_names(self, image, point, point_name):
        (x, y) = point
        blue = (240, 180, 60)
        thickness = 2
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(image, point_name, (x-20, y+10), font, 2, thickness=thickness, color=blue)  
        #print('{} ({}) : ({},{})'.format(j, COCO_CATEGORY['keypoints'][j], x, y))


    def _draw_rect(self, img, rect, color, width, text_u1=None, text_u2=None, text_d1=None, text_d2=None, fill=False, lowerbar=False, upperbar=False, smaller_bb=False):

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
            bar_color = (42, 42, 165)
        elif color == 'pink':
            outline_color = (182, 84, 231)
            text_color = (182, 84, 231)
            fill_color = (131, 59, 236)
            bar_color = (131, 59, 236)
        elif color == 'yellow':
            outline_color = (55, 250, 250)
            text_color = (55, 250, 250)
            fill_color = (55, 250, 250)
            bar_color = (55, 250, 250)
        elif color == 'blue':
            outline_color = (240, 120, 0)
            text_color = (240, 120, 0)
            fill_color = (240, 120, 0)
            bar_color = (240, 120, 0)
        elif color == 'green':
            green = (120, 255, 60)
            outline_color = green
            text_color = green
            fill_color = green
            bar_color = green
        elif color == 'orange':
            outline_color = (25, 140, 255)
            text_color = (25, 140, 255)
            fill_color = (25, 140, 255)
            bar_color = (25, 140, 255)
        elif color == 'red':
            outline_color = (49, 60, 255)
            text_color = (49, 60, 255)
            fill_color = (49, 60, 255)
            bar_color = (49, 60, 255)
        else:
            assert "Color {} not supported".format(color)
            outline_color = (200, 200, 200)
            text_color = (200, 200, 200)
            fill_color = (200, 200, 200)
            bar_color = (200, 200, 200)

        # If lowerbar is set, just show text in white.
        if lowerbar is True or upperbar is True:
            if color == 'yellow':
                text_color = (255, 0, 0)
            elif color == 'green':
                text_color = (255, 255, 255)
            else:
                text_color = (255, 255, 255)

        font_size = 1.5
        font_type = cv2.FONT_HERSHEY_PLAIN
        thickness = 2
        x1, y1, x2, y2 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
        h, w, _ = img.shape

        if smaller_bb is False:
            cv2.rectangle(img, (x1, y1), (x2, y2), outline_color, width)

        if fill is True:
            alpha = 0.3
            overlay = img.copy()
            overlay[y1:y2, x1:x2] = fill_color
            cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

        if lowerbar is True:
            alpha = 0.8
            lowerbar_img = img.copy()
            lowerbar_img[y2-30:y2, x1:x2] = bar_color
            cv2.addWeighted(lowerbar_img, alpha, img, 1 - alpha, 0, img)

        if upperbar is True:
            alpha = 0.8
            upperbar_img = img.copy()
            upperbar_img[y1:y1+60, x1:x2] = bar_color
            cv2.addWeighted(upperbar_img, alpha, img, 1 - alpha, 0, img)

        if text_u1 is not None:
            box_size, _ = cv2.getTextSize(text_u1, font_type, font_size, thickness)
            txt_size = box_size[0]
            txt_loc = (int(x1 + (x2 - x1)/2 - txt_size/2), y1 + 25)
            cv2.putText(img, text_u1, txt_loc, font_type, font_size, text_color, thickness)

        if text_u2 is not None:
            box_size, _ = cv2.getTextSize(text_u2, font_type, font_size, thickness)
            txt_size = box_size[0]
            txt_loc = (int(x1 + (x2 - x1)/2 - txt_size/2), y1 + 45)
            cv2.putText(img, text_u2, txt_loc, font_type, font_size, text_color, thickness)

        if text_d1 is not None:
            box_size, _ = cv2.getTextSize(text_d1, font_type, font_size, thickness)
            txt_size = box_size[0]
            txt_loc = (int(x1 + (x2 - x1)/2 - txt_size/2), y2 - 8)
            cv2.putText(img, text_d1, txt_loc, font_type, font_size, text_color, thickness)

        if text_d2 is not None:
            box_size, _ = cv2.getTextSize(text_d2, font_type, font_size, thickness)
            txt_size = box_size[0]
            txt_loc = (int(x1 + (x2 - x1)/2 - txt_size/2), y2 - 5)
            cv2.putText(img, text_d2, txt_loc, font_type, font_size, text_color, thickness)

    def _calc_duration(self, person_idx, xs, ys):
        global durations
        global prev_xs
        global prev_ys
        global fps

        x = int(sum(xs)/len(xs))
        y = int(sum(ys)/len(ys))

        if abs(x - prev_xs[person_idx]) + abs(y - prev_ys[person_idx]) < 100:
            durations[person_idx] = durations[person_idx] + 1.0/fps
            if durations[person_idx] > 6:
                active[person_idx] = 1
        else:
            durations[person_idx] = 0
            #active[person_idx] = 0

        prev_xs[person_idx] = x
        prev_ys[person_idx] = y

        return round(durations[person_idx], 2)

    def _draw_bounding_box(self, image, points, person_idx):
        global norm_areas
        global norm_areas_idx
        global curr_time
        global prev_time
        
        if len(points) <= 4:
            return

        h, w, _ = image.shape

        coords = []
        for point in points:
            x = point['coords'][0]
            y = point['coords'][1]
            #print('Point: ({},{})'.format(x, y))
            coords.append((x, y))
        xs = [coords[idx][0] for idx in range(len(coords))]
        ys = [coords[idx][1] for idx in range(len(coords))]

        num = len(xs)
        limit = int(num * 0.20)
        if limit <= 0:
            limit = 1

        if len(xs) < 7:
            return

        sorted_xs = sorted(xs)
        x_min = sum(sorted_xs[:limit])/float(limit)
        x_max = sum(sorted_xs[-1*limit:])/float(limit)
        sorted_ys = sorted(ys)
        y_min = int(0.84 * sum(sorted_ys[:limit])/float(limit))
        y_max = min(int(1.1 * sum(sorted_ys[-1*limit:])/float(limit)), h-10)

        # Find area of the rectangle.
        area = int((x_max - x_min) * (y_max - y_min))

        # Find dist between shoulders
        x_ls= xs[5]
        x_rs = xs[6]
        y_ls = ys[5]
        y_rs = ys[6]
        dist = abs(y_rs - y_ls) + abs(x_rs - x_ls)
        
        # Ensure dist > 0
        dist = dist or 1

        # There are total 18 points, normalizing based on number of points detected and shoulder width.
        num_points_detected = len(xs)
        norm_area = int(area * num_points_detected / (dist * 18))

        # Put this into norm_areas global
        #print('norm_areas: {}'.format(norm_areas))
        #print('person_idx: {} '.format(person_idx))
        norm_areas[person_idx][norm_areas_idx % norm_frames_cnt] = norm_area
        norm_areas_idx = (norm_areas_idx + 1) % norm_frames_cnt 
        arr = [norm_areas[person_idx][idx] for idx in range(len(norm_areas[person_idx])) if norm_areas[person_idx][idx] > 0]
        len_arr = len(arr)
        avg_arr = int(sum(arr)/len(arr))
        avg = avg_arr 

        # Multiply by dist_bet_shoulders/total_width
        #shoulder_width = dist
        #avg = area * shoulder_width / abs(x_max - x_min) 
        #avg = int(avg)

        duration = self._calc_duration(person_idx, xs, ys)
        if active[person_idx] == 1:
            color = 'red'
        else:
            color = 'green'

        
        # Draw bounding box.
        smaller_bb = True
        if smaller_bb is True:
            y_min = (y_min + y_max)/2 - 15
            if y_min < 15:
                y_min = 15

            y_max = (y_min + y_max)/2 + 15
            if y_max > h * 0.95:
                y_max = h * 0.95

            center = (x_min + x_max)/2
            x_min = center - 120
            x_max = center + 120
            if x_min < 50:
                x_min = 50
            if x_max > w * 0.95:
                x_max = w * 0.95

        self._draw_rect(image, 
                        (x_min, y_min, x_max, y_max),
                        color,
                        2,
                        #text_u1='Person {}'.format(person_idx),
                        #text_u1='Area {:.0f}'.format(area/1000),
                        #text_u1='NormArea {}'.format(avg),
                        text_u1='Person {}'.format(person_idx + 1),
                        text_u2='Dist {}, Time {:.2f}'.format(self.pixels_to_dist2(area), duration),
                        upperbar=True,
                        lowerbar=False,
                        fill=False, 
                        smaller_bb=smaller_bb)


    def _get_centroid(self, points):
        xs = [points[idx]['coords'][0] for idx in range(len(points))]
        ys = [points[idx]['coords'][1] for idx in range(len(points))]
        x = sum(xs)/len(xs)
        y = sum(ys)/len(ys)
        return (x, y)
                        
    def __call__(self, image, object_counts, objects, normalized_peaks):
        topology = TOPOLOGY
        joint_color = self.joint_color
        link_color = self.link_color
        height = image.shape[0]
        width = image.shape[1]

        K = topology.shape[0]
        count = int(object_counts[0])

        # FPS Calculations
        global fps
        global curr_time
        global prev_time

        curr_time = time.time()
        elapsed_time = curr_time - prev_time
        if elapsed_time > 0:
            fps = 1.0 / elapsed_time
        prev_time = curr_time

        # Reset durations if object not detected.
        for idx in range(count,10):
            durations[idx] = 0
            active[idx] = 0


        persons = []
        for i in range(count):
            
            obj = objects[0][i]
            # filter no-neck
            if obj[-1] < 0:
                continue

            # filter right and left shoulders
            if obj[5] < 0 or obj[6] < 0:
                continue

            # Disable drawing lines.
            for k in range(K):
                c_a = topology[k][2]
                c_b = topology[k][3]

                # Filter out anything that is face related.
                # nose - 0
                # left_eye - 1
                # right_eye - 2
                # left_ear - 3
                # right_ear - 4

                if c_a < 5 or c_b < 5:
                    continue

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
                    if j >= 5:
                        cv2.circle(image, (x, y), 3, joint_color, 3)
                        #self._draw_point_names(image, (x,y), COCO_CATEGORY['keypoints'][j])
                        points.append({
                                'point_idx' : j,
                                'point_name' : COCO_CATEGORY['keypoints'][j],
                                'coords' : (x, y)
                                })
            
            # If less keypoints detected, just ignore this frame.
            #if len(points) < 8:
            #    continue

            # People detected
            persons.append({'idx': i, 'points': points, 'centroid': self._get_centroid(points)})


        persons = sorted(persons, key=lambda item: item['centroid'])
        for person_idx, person in enumerate(persons):
            self._draw_bounding_box(image, person['points'], person_idx)

