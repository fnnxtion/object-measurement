import math
class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
        self.temp_center_points = {}

    def update(self, objects_rect, labels, confidences):
        self.temp_center_points = {}  # Reset temp_center_points for each frame
        objects_bbs_ids = []
        for rect, label, confidence in zip(objects_rect, labels, confidences):
            x1, y1, w, h = rect
            cx = (x1 + x1 + w) // 2
            cy = (y1 + y1 + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 100:
                    self.temp_center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x1, y1, w, h, id, label, confidence])
                    same_object_detected = True
                    break
            if not same_object_detected:
                self.temp_center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x1, y1, w, h, self.id_count, label, confidence])
                self.id_count += 1  # Increment ID count for each new object

        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, _, _ = obj_bb_id
            center = self.temp_center_points[object_id]
            new_center_points[object_id] = center
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
