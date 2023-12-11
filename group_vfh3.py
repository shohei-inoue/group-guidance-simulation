# 探査向上性を過去の探査中心の平均にした場合のvfhを用いた群誘導
import random
import math
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.stats import vonmises


# -------------------------------- const parameter ---------------------------------------------------
MAX_MOVEMENT = 3.0                          # 最大移動量
MIN_MOVEMENT = 2.0                          # 最小移動量
MAX_BOIDS_MOVEMENT = 3.0                    # boids判定時の最大移動量
MIN_BOIDS_MOVEMENT = 2.0                    # boids判定時の最小移動量
OUTER_BOUNDARY = 10.0                       # マーカ-の外側境界
INNER_BOUNDARY = 0.0                        # マーカーの内側境界
MEAN = 0.0                                  # マーカーの分布平均
VARIANCE = 10.0                              # マーカーの分布標準偏差
MAP_HEIGHT = 60                             # マップの縦サイズ
MAP_WIDTH = 200                              # マップの横サイズ
CENTER_X = math.floor(MAP_WIDTH / 2)
CENTER_Y = math.floor(MAP_HEIGHT / 2)
AREA_COVERAGE_THRESHOLD = 80.0              # 領域網羅率のマーカー変換値
AREA_STEP_THRESHOLD = 140                   # 領域探査のステップ上限
ALL_AREA_COVERAGE_THRESHOLD = 70.0          # マップ全体の網羅率の閾値
AREA_COVERAGE_STEP = 10
SAVE_DIRECTORY = 'csv/'                     # csvファイルの格納先フォルダ
# ------------ VFHに使用するパラメータ -----------------------
VFH_DRIVABILITY_BIAS = 0.4
VFH_EXPLORATION_BIAS = 0.6
VFH_EXPLORATION_STD = 120
VFH_MIN_VALUE = 0.01
VFH_VONMISES_KAPPA = 2
VFH_BINS = 16
# ----------------------------------- class ----------------------------------------------------------

# ランダムウォーク
class Random_walk:
    def __init__(self, x, y): # x, yにはmarkerの座標が入力
        self.x = random.uniform(x - OUTER_BOUNDARY / 4, x + OUTER_BOUNDARY / 4)
        self.y = random.uniform(y - OUTER_BOUNDARY / 4, y + OUTER_BOUNDARY / 4)
        self.point = np.array([self.x, self.y])
        self.amount_of_movement = 0.0
        self.direction_angle = 0.0
        self.step = 0
    
    
    def get_arguments(self):
        return pd.DataFrame({'step': [self.step], 'x': [self.x], 'y': [self.y], 'point': [self.point], 'amount_of_movement': [self.amount_of_movement], 'direction_angle': [self.direction_angle]})
    
    
    def step_motion(self):
        self.direction_angle = np.rad2deg(random.uniform(0, 2.0 * math.pi))
        self.amount_of_movement = random.uniform(MIN_MOVEMENT, MAX_MOVEMENT)
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
        
        amount_of_movement = random.uniform(MIN_BOIDS_MOVEMENT, MAX_BOIDS_MOVEMENT)
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
        amount_of_movement = random.uniform(MIN_MOVEMENT, MAX_MOVEMENT)
        dx = amount_of_movement * math.cos(math.radians(self.direction_angle))
        dy = amount_of_movement * math.sin(math.radians(self.direction_angle))
        prediction_point = np.array([self.x + dx, self.y + dy])
        return prediction_point
    
    
    # 棄却決定
    def rejection_decision(self, marker):
        # 棄却
        while True:
            direction_angle = np.rad2deg(random.uniform(0.0,  2.0 * math.pi))
            amount_of_movement = random.uniform(MIN_MOVEMENT, MAX_MOVEMENT)
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
        create_map(main_map, self.point, self.x, self.y)
        update_map(sub_map, round(marker.x), round(marker.y), self.point, self.x, self.y)
        
        self.x = self.point[0]
        self.y = self.point[1]
        self.distance = np.linalg.norm(self.point - marker.point)
        self.azimuth = self.azimuth_adjustment(marker)
        self.step += 1


# 実マーカー
class Real_marker:
    def __init__(self, marker_id, x, y):
        self.marker_id = marker_id
        self.x = x
        self.y = y
        self.point = np.array([self.x, self.y])
        self.coverage_ratio = 0.0
        self.mean = MEAN
        self.variance = VARIANCE
        self.inner_boundary = INNER_BOUNDARY
        self.outer_boundary = OUTER_BOUNDARY
        self.collision_df = pd.DataFrame(columns=['x', 'y', 'azimuth', 'distance'])
        self.already_direction_index = []
        self.parent = None
    
    
    def get_arguments(self):
        return pd.DataFrame({'marker_id': [self.marker_id], 'x': [self.x], 'y': [self.y], 'point': [self.point], 'coverage_ratio': [self.coverage_ratio], 'mean': [self.mean], 'variance': [self.variance], 'inner_boundary': [self.inner_boundary], 'outer_boundary': [self.outer_boundary], 'parent': [self.parent]})
    
    
    # 親方向から見て最も探査向上性が高くなる角度を求める
    def calculate_mu_azimuth(self):
        if self.parent != None:
            azimuth = 0.0
            if self.parent.x != self.x:
                vec_d = np.array(self.point - self.parent.point)
                vec_x = np.array([self.x - self.parent.x, 0])
                azimuth = np.rad2deg(math.acos(vec_d @ vec_x / (np.linalg.norm(vec_d) * np.linalg.norm(vec_x))))
        
            if self.x - self.parent.x < 0:
                if self.y - self.parent.y >= 0:
                    azimuth = np.rad2deg(math.pi) - azimuth
                else:
                    azimuth += np.rad2deg(math.pi)
            else:
                if self.y - self.parent.y < 0:
                    azimuth = np.rad2deg(2.0 * math.pi) - azimuth
        else:
            azimuth = None
        
        return azimuth
    
    
    # 今までの探査中心の座標から平均を求める
    def calculate_mu_azimuth2(self, marker_list):
        if len(marker_list) != 0:
            x_sum = 0
            y_sum = 0
            for i in range(len(marker_list)):
                x_sum += marker_list[i].x
                y_sum += marker_list[i].y
        
            x_mean = x_sum / len(marker_list)
            y_mean = y_sum / len(marker_list)
        
            azimuth = 0.0
            if x_mean != self.x:
                vec_d = np.array([self.x - x_mean, self.y - y_mean])
                vec_x = np.array([self.x - x_mean, 0])
                azimuth = np.rad2deg(math.acos(vec_d @ vec_x / (np.linalg.norm(vec_d) * np.linalg.norm(vec_x))))
        
            if self.x - x_mean < 0:
                if self.y - y_mean >= 0:
                    azimuth = np.rad2deg(math.pi) - azimuth
                else:
                    azimuth += np.rad2deg(math.pi)
            else:
                if self.y - y_mean < 0:
                    azimuth = np.rad2deg(2.0 * math.pi) - azimuth
        else:
            azimuth = None
        
        return azimuth
    
    
    def calculate_collision_df(self, x, y):
        collision_azimuth = 0.0
        if x != self.x:
            vec_d = np.array([x - self.x, y - self.y])
            vec_x = np.array([x - self.x, 0])
            collision_azimuth = np.rad2deg(math.acos(vec_d @ vec_x / (np.linalg.norm(vec_d) * np.linalg.norm(vec_x))))
        else:
            vec_d = np.array([x - self.x, y - self.y])
        
        if x - self.x < 0:
            if y - self.y >= 0:
                collision_azimuth = np.rad2deg(math.pi) - collision_azimuth
            else:
                collision_azimuth += np.rad2deg(math.pi)
        else:
            if y - self.y < 0:
                collision_azimuth = np.rad2deg(2.0 * math.pi) - collision_azimuth
        
        collision_distance = np.linalg.norm(vec_d)
        
        add_collision_data = pd.DataFrame({'x': [x], 'y': [y], 'azimuth': [collision_azimuth], 'distance': [collision_distance]})
        
        self.collision_df = pd.concat([self.collision_df, add_collision_data])
    
    
    # VFHから確率分布を作成し移動先を決定する(探査向上性と走行可能性による確率密度分布)
    def vfh_using_probability(self, marker_list, bins = VFH_BINS):
        loc = self.calculate_mu_azimuth2(marker_list)
        histogram = [1 for i in range(bins)]
        split_arg = np.rad2deg(2.0 * math.pi) / bins

        for i in range(self.collision_df.shape[0]):
            collision_azimuth_value = self.collision_df['azimuth'].iloc[i]
            
            for j in range(bins):
                bin_range_start = j * split_arg
                bin_range_end = (j + 1) * split_arg
                
                if bin_range_start <= collision_azimuth_value < bin_range_end:
                    histogram[j] += 1
                    break
        
        for i in range(bins):
            histogram[i] = 1 / histogram[i]
        
        total_histogram = np.sum(histogram)
        normalize_histogram = np.array(histogram) / total_histogram
        
        scaled_histogram = VFH_MIN_VALUE + normalize_histogram * (1 - VFH_MIN_VALUE)
        scaled_histogram = scaled_histogram / np.sum(scaled_histogram)
        
        if loc == None:
            scaled_histogram = scaled_histogram.tolist()
            select_index = np.random.choice(np.arange(0, 16), p=scaled_histogram)
        else:
            probability_of_exploration_rate = [0 for i in range(bins)]
            for i in range(bins):
                bin_range_start = vonmises.cdf(loc = np.deg2rad(loc), kappa = VFH_VONMISES_KAPPA, x = np.deg2rad(i * split_arg))
                bin_range_end = vonmises.cdf(loc = np.deg2rad(loc), kappa = VFH_VONMISES_KAPPA, x = np.deg2rad((i + 1) * split_arg))
                probability_of_exploration_rate[i] = abs(bin_range_end - bin_range_start)
            
            probability_of_exploration_rate = np.array(probability_of_exploration_rate)
            probability_density = VFH_DRIVABILITY_BIAS * scaled_histogram + VFH_EXPLORATION_BIAS * probability_of_exploration_rate
            
            probability_density = probability_density.tolist()
            
            select_index = np.random.choice(np.arange(0, 16), p=probability_density)
            self.already_direction_index.append(select_index)
        
        return (split_arg + 0.5) * select_index
    
    
    # VFHから確率分布を作成し移動先を決定する(走行可能性による確率密度分布)
    def vfh_using_probability_only_obstacle_density(self, bins=VFH_BINS):
        histogram = [1 for i in range(bins)]
        split_arg = np.rad2deg(2.0 * math.pi) / bins
        
        for i in range(self.collision_df.shape[0]):
            collision_azimuth_value = self.collision_df['azimuth'].iloc[i]
            
            for j in range(bins):
                bin_range_start = j * split_arg
                bin_range_end = (j + 1) * split_arg
                
                if bin_range_start <= collision_azimuth_value < bin_range_end:
                    histogram[j] += 1
                    break
        
        for i in range(bins):
            histogram[i] = 1 / histogram[i]
        
        total_histogram = np.sum(histogram)
        normalized_histogram = np.array(histogram) / total_histogram
        
        scaled_histogram = VFH_MIN_VALUE + normalized_histogram * (1 - VFH_MIN_VALUE)
        scaled_histogram = scaled_histogram / np.sum(scaled_histogram)
        
        scaled_histogram = scaled_histogram.tolist()
        select_index = np.random.choice(np.arange(0, 16), p=scaled_histogram)
        self.already_direction_index.append(select_index)
        
        return split_arg * (select_index + 0.5)
    
    
    # VFHにより最も効率の良いとされる方向を決定する(探査向上性と走行可能性)
    def vfh(self, marker_list, bins = VFH_BINS):
        loc = self.calculate_mu_azimuth2(marker_list)
        histogram = [1 for i in range(bins)]
        split_arg = np.rad2deg(2.0 * math.pi) / bins

        for i in range(self.collision_df.shape[0]):
            collision_azimuth_value = self.collision_df['azimuth'].iloc[i]
            
            for j in range(bins):
                bin_range_start = j * split_arg
                bin_range_end = (j + 1) * split_arg
                
                if bin_range_start <= collision_azimuth_value < bin_range_end:
                    histogram[j] += 1
                    break
        
        for i in range(bins):
            histogram[i] = 1 / histogram[i]
        
        total_histogram = np.sum(histogram)
        normalized_histogram = np.array(histogram) / total_histogram
        
        scaled_histogram = VFH_MIN_VALUE + normalized_histogram * (1 - VFH_MIN_VALUE)
        scaled_histogram = scaled_histogram / np.sum(scaled_histogram)
        
        if loc == None:
            scaled_histogram = scaled_histogram.tolist()
            best_index = scaled_histogram.index(max(scaled_histogram))
        else:
            probability_of_exploration_rate = []
            for i in range(bins):
                probability_of_exploration_rate.append(0)
            
            for i in range(bins):
                bin_range_start = vonmises.cdf(loc = np.deg2rad(loc), kappa = VFH_VONMISES_KAPPA, x = np.deg2rad(i * split_arg))
                bin_range_end = vonmises.cdf(loc = np.deg2rad(loc), kappa = VFH_VONMISES_KAPPA, x = np.deg2rad((i + 1) * split_arg))
                probability_of_exploration_rate[i] = abs(bin_range_end - bin_range_start)
        
            probability_of_exploration_rate = np.array(probability_of_exploration_rate)
            probability_density = VFH_DRIVABILITY_BIAS * scaled_histogram + VFH_EXPLORATION_BIAS * probability_of_exploration_rate
            
            probability_density = probability_density.tolist()
            best_index = probability_density.index(max(probability_density))
        
        return split_arg * (best_index + 0.5)
    
    
    # VFHにより最も効率の良いとされる方向を決定する(走行可能性)
    def vfh_only_obstacle_density(self, bins = VFH_BINS):
        histogram = [1 for i in range(bins)]
        split_arg = np.rad2deg(2.0 * math.pi) / bins

        for i in range(self.collision_df.shape[0]):
            collision_azimuth_value = self.collision_df['azimuth'].iloc[i]
            
            for j in range(bins):
                bin_range_start = j * split_arg
                bin_range_end = (j + 1) * split_arg
                
                if bin_range_start <= collision_azimuth_value < bin_range_end:
                    histogram[j] += 1
                    break
        
        for i in range(bins):
            histogram[i] = 1 / histogram[i]
        
        total_histogram = np.sum(histogram)
        normalized_histogram = np.array(histogram) / total_histogram
        
        scaled_histogram = VFH_MIN_VALUE + normalized_histogram * (1 - VFH_MIN_VALUE)
        scaled_histogram = scaled_histogram / np.sum(scaled_histogram)
        scaled_histogram = scaled_histogram.tolist()
        
        best_index = scaled_histogram.index(max(scaled_histogram))
        
        return split_arg * (best_index + 0.5)


# 仮想マーカー
class Virtual_marker(Real_marker):
    def __init__(self,marker_id, x, y, parent):
        super().__init__(marker_id, x, y)
        self.parent = parent
    
    
    def get_arguments(self):
        arguments = super().get_arguments()
        return arguments
    
    
    def calculate_parent_azimuth(self):
        return super().calculate_parent_azimuth()
    
    
    def calculate_collision_df(self, x, y):
        return super().calculate_collision_df(x, y)
    
    
    def vfh(self, bins=16):
        return super().vfh(bins)
    
    
    def vfh_using_probability(self, bins=16):
        return super().vfh_using_probability(bins)


# 障害物
class Obstacle_square:
    def __init__(self, x, y, h, w):
        self.x = x
        self.y = y
        self.point = np.array([self.x, self.y]) # 左下頂点
        self.h = h
        self.w = w
    
    
    def get_arguments(self):
        return pd.DataFrame({'x': [self.x], 'y': [self.y], 'point': [self.point], 'height': [self.h], 'width': [self.w]})
# ------------------------------- function -----------------------------------------------------
# 提案分布
def distribution(distance, mean, variance):
    return 1 / math.sqrt(2 * math.pi) * math.exp(-(distance - mean) ** 2 / (2 * variance ** 2))


# グリッドマップへの追加
def create_map(map, prediction_point, current_x, current_y):
    passing_point = []
    x_min = min(prediction_point[0], current_x)
    y_min = min(prediction_point[1], current_y)
    x_max = max(prediction_point[0], current_x)
    y_max = max(prediction_point[1], current_y)
    
    if x_max != x_min:
        a = (prediction_point[1] - current_y) / (prediction_point[0] - current_x)
        b = -a * prediction_point[0] + prediction_point[1]
        for i in range(round(y_min), round(y_max)):
            for j in range(round(x_min), round(x_max)):
                if i == round(a * j + b):
                    passing_point.append([i, j])
    else:
        for i in range(round(y_min), round(y_max)):
            passing_point.append([i, x_max])
    
    for i in range(len(passing_point)):
        if -(MAP_HEIGHT / 2.0) <= passing_point[i][0] <= (MAP_HEIGHT / 2.0) and -(MAP_WIDTH/ 2.0) <= passing_point[i][1] <= (MAP_WIDTH / 2.0):
            map[CENTER_Y - 1 + passing_point[i][0]][CENTER_X - 1 + passing_point[i][1]] += 1


# グリットマップへの追加2(marker_x, marker_yは丸めて入れる)
def update_map(map, marker_x, marker_y, prediction_point, current_x, current_y):
    passing_point = []
    x_min = min(prediction_point[0], current_x)
    y_min = min(prediction_point[1], current_y)
    x_max = max(prediction_point[0], current_x)
    y_max = max(prediction_point[1], current_y)
    
    if x_max != x_min:
        a = (prediction_point[1] - current_y) / (prediction_point[0] - current_x)
        b = -a * prediction_point[0] + prediction_point[1]
        for i in range(round(y_min), round(y_max)):
            for j in range(round(x_min), round(x_max)):
                if i == round(a * j + b):
                    passing_point.append([i, j])
    else:
        for i in range(round(y_min), round(y_max)):
            passing_point.append([i, x_max])
    
    for i in range(len(passing_point)):
        if marker_y - OUTER_BOUNDARY <= passing_point[i][0] <= marker_y + OUTER_BOUNDARY and marker_x - OUTER_BOUNDARY <= passing_point[i][1] <= marker_x + OUTER_BOUNDARY:
            map[round(OUTER_BOUNDARY - (passing_point[i][0] - marker_y) - 1)][round(OUTER_BOUNDARY + (passing_point[i][1] - marker_x) - 1)] += 1

# 網羅率計算 →　変更したい
def area_coverage_calculation(map, marker):
    area_count = 0
    explored_count = 0
    for i in range(round(-OUTER_BOUNDARY), round(OUTER_BOUNDARY + 1)):
        for j in range(round(-OUTER_BOUNDARY), round(OUTER_BOUNDARY + 1)):
            distance = math.sqrt(i ** 2 + j ** 2)
            if INNER_BOUNDARY < distance < OUTER_BOUNDARY:
                area_count += 1
                if 0 <= round(OUTER_BOUNDARY - i) <= 2 * OUTER_BOUNDARY - 1 and 0 <= round(OUTER_BOUNDARY - j) <= 2 * OUTER_BOUNDARY - 1:
                    if map[round(OUTER_BOUNDARY - i)][round(OUTER_BOUNDARY - j)] >= 1:
                        explored_count += 1
    
    return explored_count / area_count * 100


# メインマーカー変換閾値
def marker_change_threshold():
    area_count = 0
    for i in range(round(-OUTER_BOUNDARY), round(OUTER_BOUNDARY + 1)):
        for j in range(round(-OUTER_BOUNDARY), round(OUTER_BOUNDARY + 1)):
            distance = math.sqrt(i ** 2 + j ** 2)
            if INNER_BOUNDARY < distance < OUTER_BOUNDARY:
                area_count += 1
    return 1.5 / area_count * 100.0


# 全体網羅率計算
def map_coverage_calculation(map):
    all_count = map.size
    explored_count = np.count_nonzero(map >= 1)
    
    return explored_count / all_count * 100.0


# 探査終了閾値
def exploration_completed_threshold(map):
    all_count = map.size
    
    return 1.5 / all_count * 100


# -------------------------------- main ---------------------------------------------------------
def main():
    # 障害物の作成
    obstacle_list = []
    obstacle1 = Obstacle_square(-100, 25, 5, 200)
    obstacle2 = Obstacle_square(-100, -30, 5, 200)
    obstacle3 = Obstacle_square(-100, -30, 60, 5)
    obstacle4 = Obstacle_square(95, -30, 60, 5)
    obstacle5 = Obstacle_square(-95, 10, 15, 40)
    obstacle6 = Obstacle_square(-95, -25, 18, 50)
    obstacle7 = Obstacle_square(-10, 15, 10, 20)
    obstacle8 = Obstacle_square(-5, -25, 25, 40)
    obstacle9 = Obstacle_square(50, 10, 5, 20)
    obstacle10 = Obstacle_square(70, 10, 15, 5)
    
    obstacle_list.append(obstacle1)
    obstacle_list.append(obstacle2)
    obstacle_list.append(obstacle3)
    obstacle_list.append(obstacle4)
    obstacle_list.append(obstacle5)
    obstacle_list.append(obstacle6)
    obstacle_list.append(obstacle7)
    obstacle_list.append(obstacle8)
    obstacle_list.append(obstacle9)
    obstacle_list.append(obstacle10)
    
    obstacle_df_list = []
    for i in range(len(obstacle_list)):
        df = pd.DataFrame()
        df = pd.DataFrame(obstacle_list[i].get_arguments())
        obstacle_df_list.append(df)
    
    # マーカーの作成
    main_marker_list = []
    marker_list = []
    marker_change_key = []
    marker_change_th = marker_change_threshold()
    collision_counter = []
    area_step = []
    marker_algorithm_key = []
    
    # Redの生成
    red_list = []
    red_df_list = []
    
    group_num = int(input('How many groups do you make? : '))
    
    for i in range(group_num):
        red_num = int(input('Please enter the number of Reds in the ' + str(i + 1) + ' group. : '))
        marker_x = int(input('Please enter the x-coordinate of the ' + str(i + 1) + ' group marker : '))
        marker_y = int(input('Please enter the y-coordinate of the ' + str(i + 1) + ' group marker : '))
        main_marker = Real_marker('marker' + str(i + 1) + '-1', marker_x, marker_y)
        red_list.append([])
        red_df_list.append([])
        marker_list.append([])
        marker_change_key.append(0)
        collision_counter.append(0)
        area_step.append(0)
        marker_algorithm_key.append(0)
        main_marker_list.append(main_marker)
    
        for j in range(red_num):
            red_id = 'red' + str(i + 1) + '-' + str(j + 1)
            red = Red(main_marker.x, main_marker.y, red_id, main_marker)
            red_list[i].append(red)
            red_df = pd.DataFrame()
            red_df = pd.DataFrame(red_list[i][j].get_arguments())
            red_df_list[i].append(red_df)
    
    grid_map = []
    X_list = []
    Y_list = []
    # グリッドマップの作成
    for i in range(group_num):
        grid_map.append(np.zeros([int(2 * OUTER_BOUNDARY), int(2 * OUTER_BOUNDARY)]))
        x = range(round(-OUTER_BOUNDARY), round(OUTER_BOUNDARY + 1))
        y = range(round(-OUTER_BOUNDARY), round(OUTER_BOUNDARY + 1))
        X, Y = np.meshgrid(x, y)
        X_list.append(X)
        Y_list.append(Y)
    
    # 全体マップの作成
    main_map = np.zeros([MAP_HEIGHT, MAP_WIDTH])
    x = range(-CENTER_X, CENTER_X + 1)
    y = range(-CENTER_Y, CENTER_Y + 1)
    X, Y = np.meshgrid(x, y)
    X_list.append(X)
    Y_list.append(Y)
    
    fig, ax = plt.subplots(1, group_num + 1)
    
    
    while True:
        # 全体網羅率の計算
        map_coverage_ratio = map_coverage_calculation(main_map)
        print('=' * 30)
        print(red_list[-1][-1].step, "step : map_coverage_ratio :", map_coverage_ratio)
        if map_coverage_ratio >= ALL_AREA_COVERAGE_THRESHOLD or len(marker_list[0]) == 100: # 仮
            print('Exploration completed.')
            break
        
        # 1群ごとの動作
        for i in range(group_num):
            area_coverage_ratio = area_coverage_calculation(grid_map[i], main_marker_list[i])
            print(red_list[i][-1].step, "step : area_coverage_ratio : ", area_coverage_ratio)
            print('-' * 30)
            if area_coverage_ratio - main_marker_list[i].coverage_ratio <= marker_change_th:
                marker_change_key[i] += 1
            else:
                marker_change_key[i] = 0
            
            main_marker_list[i].coverage_ratio = area_coverage_ratio

            # 探査中心の変換(ステップ数で変換ver)
            # --------------やるべきこと---------------------------
            # 80%を超える最適なステップ数を求める必要あり
            # 神視点バージョンも作ろう
            # 80以上, vfh_using_probability()
            # vfh_using_probability()を行なった次の変換で
            # 50以下→戻り行動, 50以上, 80未満-> vfh() or vfh_only_obstacle_density()
            # 探査向上性の方式を前に使用したマーカーの分散等から推定する方式の追加
            if main_marker_list[i].coverage_ratio >= AREA_COVERAGE_THRESHOLD:
                next_theta = math.radians(main_marker_list[i].vfh_using_probability(marker_list[i]))
                next_x = OUTER_BOUNDARY * math.cos(next_theta) + main_marker_list[i].x
                next_y = OUTER_BOUNDARY * math.sin(next_theta) + main_marker_list[i].y
                marker_list[i].append(main_marker_list[i])
                main_marker_list[i] = Virtual_marker('marker' + str(i + 1) + '-' + str(len(marker_list[i]) + 1), next_x, next_y, main_marker_list[i])
                print('=' * 30)
                print(str(i + 1) + 'group marker changed by vfh_using_probability(). (', main_marker_list[i].x, ', ', main_marker_list[i].y, ') : ', len(marker_list[i]))
                print('=' * 30)
                marker_algorithm_key[i] = 1
                marker_change_key[i] = 0
                
                # グリットマップの変更
                grid_map[i] = np.zeros([round(2 * OUTER_BOUNDARY), round(2 * OUTER_BOUNDARY)])
                ax[i].pcolormesh(X_list[i], Y_list[i], grid_map[i], cmap = 'brg', edgecolors = 'black', shading='auto')
                ax[i].set_title('marker' + str(i + 1) + ' grid_map')
                plt.pause(0.05)
                
            elif marker_change_key[i] >= 4 or area_step[i] == AREA_STEP_THRESHOLD:
                # 前のアルゴリズムがvfh_using_probability()の場合(戻り行動)
                if marker_algorithm_key[i] == 1:
                    marker_list[i].append(main_marker_list[i])
                    main_marker_list[i] = main_marker_list[i].parent
                    print('=' * 30)
                    print(str(i + 1) + 'group marker back changed. (', main_marker_list[i].x, ', ', main_marker_list[i].y, ') : ', len(marker_list[i]))
                    print('=' * 30)
                # 前のアルゴリズムがvfh_using_probability()ではない場合
                elif marker_algorithm_key[i] == 0:
                    #next_theta = main_marker_list[i].vfh(marker_list[i])
                    next_theta = math.radians(main_marker_list[i].vfh_only_obstacle_density())
                    next_x = OUTER_BOUNDARY * math.cos(next_theta) + main_marker_list[i].x
                    next_y = OUTER_BOUNDARY * math.sin(next_theta) + main_marker_list[i].y
                    marker_list[i].append(main_marker_list[i])
                    main_marker_list[i] = Virtual_marker('marker' + str(i + 1) + '-' + str(len(marker_list[i]) + 1), next_x, next_y, main_marker_list[i])
                    print('=' * 30)
                    print(str(i + 1) + 'group marker changed by vfh_using_probability(). (', main_marker_list[i].x, ', ', main_marker_list[i].y, ') : ', len(marker_list[i]))
                    print('=' * 30)
                marker_change_key[i] = 0
                marker_algorithm_key[i] = 0
                area_step[i] = 0

                # グリットマップの変更
                grid_map[i] = np.zeros([round(2 * OUTER_BOUNDARY), round(2 * OUTER_BOUNDARY)])
                ax[i].pcolormesh(X_list[i], Y_list[i], grid_map[i], cmap = 'brg', edgecolors = 'black', shading='auto')
                ax[i].set_title('marker' + str(i + 1) + ' grid_map')
                plt.pause(0.05)
            
            for j in range(AREA_COVERAGE_STEP):
                for k in range(len(red_list[i])):
                    red_list[i][k].step_motion(obstacle_list, main_marker_list[i], main_map, grid_map[i])
                    red_df_list[i][k] = pd.concat([red_df_list[i][k], red_list[i][k].get_arguments()])
                
                    if red_list[i][k].collision_flag == 1:
                        collision_counter[i] += 1
                        main_marker_list[i].calculate_collision_df(red_list[i][k].x, red_list[i][k].y)
                    
                area_step[i] += 1
        
        # 群ごとのグリッドマップ
        ax[i].pcolormesh(X_list[i], Y_list[i], grid_map[i], cmap = 'brg', edgecolors = 'black', shading='auto')
        ax[i].set_title('marker' + str(i + 1) + ' grid_map')
        # 全体のグリッドマップ
        ax[-1].pcolormesh(X_list[-1], Y_list[-1], main_map, cmap = 'brg', edgecolors = 'black', shading='auto')
        ax[-1].set_title('main grid_map')
        plt.pause(0.01)
    
    # データフレームの作成
    # Red
    for i in range(len(red_df_list)):
        for j in range(len(red_df_list[i])):
            red_df_list[i][j].to_csv(SAVE_DIRECTORY + red_list[i][j].red_id + '.csv')
    
    # Marker
    for i in range(len(marker_list)):
        for j in range(len(marker_list[i])):
            if j == 0:
                marker_df = pd.DataFrame(marker_list[i][j].get_arguments())
            else:
                marker_df = pd.concat([marker_df, marker_list[i][j].get_arguments()])
            marker_df.to_csv(SAVE_DIRECTORY + 'marker' + str(i + 1) + '.csv')
    
    # obstacle
    for i in range(len(obstacle_df_list)):
        obstacle_df_list[i].to_csv(SAVE_DIRECTORY + 'obstacle' + str(i + 1) + '.csv')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopped by keyboard input(ctrl-c)")
        sys.exit()