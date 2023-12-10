import random
import math
import numpy as np
import pandas as pd
import params
import map

# ランダムウォーク
class Random_walk:
    def __init__(self, x, y): # x, yにはmarkerの座標が入力
        self.x = random.uniform(x - params.OUTER_BOUNDARY / 4, x + params.OUTER_BOUNDARY / 4)
        self.y = random.uniform(y - params.OUTER_BOUNDARY / 4, y + params.OUTER_BOUNDARY / 4)
        self.point = np.array([self.x, self.y])
        self.amount_of_movement = 0.0
        self.direction_angle = 0.0
        self.step = 0
    
    
    def get_arguments(self):
        return pd.DataFrame({'step': [self.step], 'x': [self.x], 'y': [self.y], 'point': [self.point], 'amount_of_movement': [self.amount_of_movement], 'direction_angle': [self.direction_angle]})
    
    
    def step_motion(self):
        self.direction_angle = np.rad2deg(random.uniform(0, 2.0 * math.pi))
        self.amount_of_movement = random.uniform(params.MIN_MOVEMENT, params.MAX_MOVEMENT)
        self.x = self.x + self.amount_of_movement * math.cos(math.radians(self.direction_angle))
        self.y = self.y + self.amount_of_movement * math.sin(math.radians(self.direction_angle))
        self.point = np.array([self.x, self.y])
        self.step += 1


# Red
class Red(Random_walk):
    def __init__(self, x, y, red_id, marker, **rest):
        super().__init__(x, y, **rest)
        self.red_id = red_id
        self.distance = np.linalg.norm(self.point - marker.point)
        self.azimuth = self.azimuth_adjustment(marker)
        self.collision_flag = 0
        self.boids_flag = 0
        self.estimated_probability = 0.0
    
    
    def azimuth_adjustment(self, marker):
        azimuth = 0.0
        if self.x != marker.x:
            vec_d = np.array(self.point - marker.point)
            vec_x = np.array([self.x - marker.x, 0])

            azimuth = np.rad2deg(math.acos(vec_d @ vec_x / (np.linalg.norm(vec_d) * np.linalg.norm(vec_x))))
        
        if self.x - marker.x < 0:
            if self.y - marker.y >= 0:
                azimuth = np.rad2deg(math.pi) - azimuth
            else:
                azimuth += np.rad2deg(math.pi)
        else:
            if self.y - marker.y < 0:
               azimuth = np.rad2deg(2.0 * math.pi) - azimuth
        return azimuth
    
    
    def get_arguments(self):
        return pd.DataFrame({'step': [self.step], 'x': [self.x], 'y': [self.y], 'point': [self.point], 'amount_of_movement': [self.amount_of_movement], 'direction_angle': [self.direction_angle], 'distance': [self.distance], 'azimuth': [self.azimuth], 'collision_flag': [self.collision_flag], 'boids_flag': [self.boids_flag], 'estimated_probability': [self.estimated_probability]})
    
    
    # boids判定
    def boids_judgement(self, marker):
        self.distance = np.linalg.norm(self.point - marker.point)
        if self.distance > marker.outer_boundary:
            self.boids_flag = 1
        elif self.distance < marker.inner_boundary:
            self.boids_flag = 2
        else:
            self.boids_flag = 0
    
    
    # boids行動
    def boids_behavior(self, marker):
        self.direction_angle = self.azimuth
        if self.boids_flag == 1:
            if self.y - marker.y >= 0:
                self.direction_angle += np.rad2deg(math.pi)
            else:
                self.direction_angle -= np.rad2deg(math.pi)
        
        amount_of_movement = random.uniform(params.MIN_BOIDS_MOVEMENT, params.MAX_BOIDS_MOVEMENT)
        dx = amount_of_movement * math.cos(math.radians(self.direction_angle))
        dy = amount_of_movement * math.sin(math.radians(self.direction_angle))
        prediction_point = np.array([self.x + dx, self.y + dy])
        return prediction_point
    
    
    # 障害物判定
    def obstacle_judgement(self, obstacle_list, prediction_point):
        intersection = []
        for i in range(len(obstacle_list)):
            if obstacle_list[i].x <= prediction_point[0] <= obstacle_list[i].x + obstacle_list[i].w and obstacle_list[i].y <= prediction_point[1] <= obstacle_list[i].y + obstacle_list[i].h:
                a = (prediction_point[1] - self.y) / (prediction_point[0] - self.x)
                b = -a * prediction_point[0] + prediction_point[1]
                intersection.append(np.array([(obstacle_list[i].y - b) / a, obstacle_list[i].y]))
                intersection.append(np.array([(obstacle_list[i].y + obstacle_list[i].h - b) / a, obstacle_list[i].y + obstacle_list[i].h]))
                intersection.append(np.array([obstacle_list[i].x, obstacle_list[i].x * a + b]))
                intersection.append(np.array([obstacle_list[i].x + obstacle_list[i].w, (obstacle_list[i].x + obstacle_list[i].w) * a + b]))
        
        distance_list = []
        for i in range(len(intersection)):
            distance = np.linalg.norm(intersection[i] - self.point)
            distance_list.append(distance)
        
        if intersection == []:
            self.collision_flag = 0
            return False
        else:
            self.collision_flag = 1
            return intersection[distance_list.index(min(distance_list))]
    
    
    # 回避行動
    def avoidance_behavior(self):
        self.direction_angle = (self.direction_angle + random.uniform(90.0, 270.0)) % np.rad2deg(math.pi * 2.0)
        amount_of_movement = random.uniform(params.MIN_MOVEMENT, params.MAX_MOVEMENT)
        dx = amount_of_movement * math.cos(math.radians(self.direction_angle))
        dy = amount_of_movement * math.sin(math.radians(self.direction_angle))
        prediction_point = np.array([self.x + dx, self.y + dy])
        return prediction_point
    
    
    # 棄却決定
    def rejection_decision(self, marker):
        # 棄却
        while True:
            direction_angle = np.rad2deg(random.uniform(0.0,  2.0 * math.pi))
            amount_of_movement = random.uniform(params.MIN_MOVEMENT, params.MAX_MOVEMENT)
            dx = amount_of_movement * math.cos(math.radians(direction_angle))
            dy = amount_of_movement * math.sin(math.radians(direction_angle))
            prediction_point = np.array([self.x + dx, self.y + dy])
            distance = np.linalg.norm(prediction_point - marker.point)
            estimated_probability = distribution(distance, marker.mean, marker.variance)
            if self.estimated_probability == 0:
                self.estimated_probability = estimated_probability
                self.direction_angle = direction_angle
            else:
                if estimated_probability / self.estimated_probability > np.random.rand():
                    self.estimated_probability = estimated_probability
                    self.direction_angle = direction_angle
                    return prediction_point
                else:
                    continue
    
    
    def step_motion(self, obstacle_list, marker, main_map, sub_map):
        # 障害物回避行動
        if self.collision_flag == 1:
            prediction_point = self.avoidance_behavior()
        else:
            # boids判定
            self.boids_judgement(marker)
            if self.boids_flag != 0:
                prediction_point = self.boids_behavior(marker)
            else:
                # 棄却判定
                prediction_point = self.rejection_decision(marker)
        # 障害物判定
        inspection_point = self.obstacle_judgement(obstacle_list, prediction_point)
        if self.collision_flag == 1:
            self.amount_of_movement = np.linalg.norm(inspection_point - self.point)
            self.point = inspection_point
        else:
            self.amount_of_movement = np.linalg.norm(prediction_point - self.point)
            self.point = prediction_point
        
        # grid map用コード
        map.create_map(main_map, self.point, self.x, self.y)
        map.update_map(sub_map, round(marker.x), round(marker.y), self.point, self.x, self.y)
        
        self.x = self.point[0]
        self.y = self.point[1]
        self.distance = np.linalg.norm(self.point - marker.point)
        self.azimuth = self.azimuth_adjustment(marker)
        self.step += 1


# ------------------------------- function -----------------------------------------------------
# 提案分布
def distribution(distance, mean, variance):
    return 1 / math.sqrt(2 * math.pi) * math.exp(-(distance - mean) ** 2 / (2 * variance ** 2))