# 群を一つのロボットと見た時のvfhを用いた群誘導
import random
import math
import statistics
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import norm, vonmises


# -------------------------------- const parameter ---------------------------------------------------
MAX_MOVEMENT = 3.0                          # 最大移動量
MIN_MOVEMENT = 1.0                          # 最小移動量
MAX_BOIDS_MOVEMENT = 3.0                    # boids判定時の最大移動量
MIN_BOIDS_MOVEMENT = 1.0                    # boids判定時の最小移動量
OUTER_BOUNDARY = 10.0                       # マーカ-の外側境界
INNER_BOUNDARY = 0.0                        # マーカーの内側境界
MEAN = 3.0                                  # マーカーの分布平均
VARIANCE = 5.0                              # マーカーの分布標準偏差
MAP_HEIGHT = 100                             # マップの縦サイズ
MAP_WIDTH = 100                              # マップの横サイズ
CENTER_X = math.floor(MAP_WIDTH / 2)
CENTER_Y = math.floor(MAP_HEIGHT / 2)
AREA_COVERAGE_THRESHOLD = 80.0              # 領域網羅率のマーカー変換値
ALL_AREA_COVERAGE_THRESHOLD = 70.0          # マップ全体の網羅率の閾値
AREA_COVERAGE_STEP = 10
SAVE_DIRECTORY = 'csv/'                     # csvファイルの格納先フォルダ
VFH_DRIVABILITY_BIAS = 0.5
VFH_EXPLORATION_BIAS = 0.5
VFH_EXPLORATION_STD = 120
VFH_MIN_VALUE = 0.01
VFH_VONMISES_KAPPA = 2
# ----------------------------------- class ----------------------------------------------------------
# ランダムウォーク
class Random_walk:
    def __init__(self, x, y):
        self.x = random.uniform(x - OUTER_BOUNDARY, x + OUTER_BOUNDARY)
        self.y = random.uniform(y - OUTER_BOUNDARY, y + OUTER_BOUNDARY)
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
            #print("azimuth_adjustment vec_d : ", vec_d, ", vec_x : ", vec_x)
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
    
    
    def step_motion(self, obstacle_list, marker, map):
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
        create_map(map, self.point, self.x, self.y)
        
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
        cd_columns = ['x', 'y', 'azimuth', 'distance']
        self.collision_df = pd.DataFrame(columns=cd_columns)
        self.already_direction_index = []
        self.parent = None
    
    
    def get_arguments(self):
        return pd.DataFrame({'marker_id': [self.marker_id], 'x': [self.x], 'y': [self.y], 'point': [self.point], 'coverage_ratio': [self.coverage_ratio], 'mean': [self.mean], 'variance': [self.variance], 'inner_boundary': [self.inner_boundary], 'outer_boundary': [self.outer_boundary], 'parent': [self.parent]})
    
    
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
    
    
    def calculate_collision_df(self, x, y):
        collision_azimuth = 0.0
        if x != self.x:
            vec_d = np.array([x - self.x, y - self.y])
            vec_x = np.array([x - self.x, 0])
            collision_azimuth = np.rad2deg(math.acos(vec_d @ vec_x / (np.linalg.norm(vec_d) * np.linalg.norm(vec_x))))
        
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
    
    
    def vfh_using_probability(self, bins = 16):
        loc = self.calculate_mu_azimuth()
        
        histogram = []
        for i in range(bins):
            histogram.append(1)
        
        split_arg = np.rad2deg(2.0 * math.pi) / bins

        # ヒストグラムに衝突判定を割り当てる
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
            
            # 一度行った方向を省く
            stack_density = 0
            for i in range(len(self.already_direction_index)):
                stack_density += scaled_histogram[int(self.already_direction_index[i])]
                scaled_histogram[int(self.already_direction_index[i])] = 0
            
            stack_density /= bins - len(self.already_direction_index)
            
            scaled_histogram = [element + stack_density if element != 0 else element for element in scaled_histogram]
            
            indexes = np.arange(0, 16)
            select_index = np.random.choice(indexes, p=scaled_histogram)
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
            
            # 一度行った方向を省く
            stack_density = 0
            for i in range(len(self.already_direction_index)):
                stack_density += probability_density[int(self.already_direction_index[i])]
                probability_density[int(self.already_direction_index[i])] = 0
            
            stack_density /= bins - len(self.already_direction_index)
            
            probability_density = [element + stack_density if element != 0 else element for element in probability_density]
            
            indexes = np.arange(0, 16)
            select_index = np.random.choice(indexes, p=probability_density)
            self.already_direction_index.append(select_index)
        
        return (split_arg + 0.5) * select_index
    
    
    def vfh(self, bins = 16):
        loc = self.calculate_mu_azimuth()
        
        histogram = []
        for i in range(bins):
            histogram.append(1)
        
        split_arg = np.rad2deg(2.0 * math.pi) / bins

        # ヒストグラムに衝突判定を割り当てる
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
            
            # 一度行った方向を省く
            stack_density = 0
            for i in range(len(self.already_direction_index)):
                stack_density += probability_density[int(self.already_direction_index[i])]
                probability_density[int(self.already_direction_index[i])] = 0
            
            stack_density /= bins - len(self.already_direction_index)
            
            probability_density = [element + stack_density if element != 0 else element for element in probability_density]
            best_index = probability_density.index(max(probability_density))
            
            self.already_direction_index.append(best_index)
        
        return (split_arg + 0.5) * best_index

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


# 網羅率計算
def area_coverage_calculation(map, marker):
    area_count = 0
    explored_count = 0
    for i in range(round(-OUTER_BOUNDARY), round(OUTER_BOUNDARY + 1)):
        for j in range(round(-OUTER_BOUNDARY), round(OUTER_BOUNDARY + 1)):
            distance = math.sqrt(i ** 2 + j ** 2)
            if INNER_BOUNDARY < distance < OUTER_BOUNDARY:
                area_count += 1
                if 0 <= round(CENTER_Y + marker.y - i) <= MAP_HEIGHT - 1 and 0 <= round(CENTER_X + marker.x - j) <= MAP_WIDTH - 1:
                    if map[round(CENTER_Y + marker.y - i)][round(CENTER_X + marker.x - j)] >= 1:
                        explored_count += 1
    
    return explored_count / area_count * 100.0


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


def intersection_azimuth_adjustment(marker, point1, point2):
    v1 = np.array([point1[0] - marker.x, point1[1] - marker.y])
    v2 = np.array([point2[0] - marker.x, point2[1] - marker.y])
    
    theta1 = np.arctan2(v1[1], v1[0])
    theta2 = np.arctan2(v2[1], v2[0])
    
    if theta1 < 0:
        theta1 += 2 * np.pi
    if theta2 < 0:
        theta2 += 2 * np.pi
    
    if theta1 > theta2:
        tmp = theta1
        theta1 = theta2
        theta2 = tmp
    
    return theta1, theta2

# -------------------------------- main ---------------------------------------------------------

def main():
    grid_map = np.zeros([MAP_HEIGHT, MAP_WIDTH])
    x = range(-CENTER_X, CENTER_X + 1)
    y = range(-CENTER_Y, CENTER_Y + 1)
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots()
    
    # 障害物の作成
    obstacle_list = []
    obstacle1 = Obstacle_square(-50, 45, 5, 100)
    obstacle2 = Obstacle_square(-50, -50, 5, 100)
    obstacle3 = Obstacle_square(-50, -50, 100, 5)
    obstacle4 = Obstacle_square(45, -50, 100, 5)
    #obstacle5 = Obstacle_square(10, 10, 10, 30)
    obstacle_list.append(obstacle1)
    obstacle_list.append(obstacle2)
    obstacle_list.append(obstacle3)
    obstacle_list.append(obstacle4)
    #obstacle_list.append(obstacle5)
    
    obstacle_df_list = []
    for i in range(len(obstacle_list)):
        df = pd.DataFrame()
        df = pd.DataFrame(obstacle_list[i].get_arguments())
        obstacle_df_list.append(df)
    
    # 実マーカーの作成
    #marker1 = Real_marker('marker1', -30, 0)
    #marker2 = Real_marker('marker2', 30, 0)
    main_marker_list = []

    # マーカー用リスト
    marker_list = []
    
    marker_change_key = []
    marker_change_th = marker_change_threshold()
    collision_counter = []
    
    #back_key = 0
    
    # Redの生成
    red_list = []
    red_df_list = []
    
    group_num = int(input('How many groups do you make? : '))
    
    for i in range(group_num):
        red_num = int(input('Please enter the number of Reds in the ' + str(i + 1) + ' group. : '))
        marker_x = int(input('Please enter the x-coordinate of the ' + str(i + 1) + ' group marker : '))
        marker_y = int(input('Please enter the y-coordinate of the ' + str(i + 1) + ' group marker : '))
        main_marker = Real_marker('marker' + str(1 + 1) + '-1', marker_x, marker_y)
        red_list.append([])
        red_df_list.append([])
        marker_list.append([])
        marker_change_key.append(0)
        collision_counter.append(0)
        main_marker_list.append(main_marker)
        
        for j in range(red_num):
            red_id = 'red' + str(i + 1) + '-' + str(j + 1)
            red = Red(main_marker.x, main_marker.y, red_id, main_marker)
            red_list[i].append(red)
            red_df = pd.DataFrame()
            red_df = pd.DataFrame(red_list[i][j].get_arguments())
            red_df_list[i].append(red_df)
    
    while True:
        # 全体網羅率の計算
        map_coverage_ratio = map_coverage_calculation(grid_map)
        print('=' * 30)
        print(red_list[-1][-1].step, "step : map_coverage_ratio :", map_coverage_ratio)
        if map_coverage_ratio >= ALL_AREA_COVERAGE_THRESHOLD or red_list[-1][-1].step == 5000:
            print('Exploration completed.')
            break
        
        
        # 1群ごとの動作
        for i in range(group_num):
            area_coverage_ratio = area_coverage_calculation(grid_map, main_marker_list[i])
            print(red_list[i][-1].step, "step : area_coverage_ratio : ", area_coverage_ratio)
            print('-' * 30)
            if collision_counter[i] >= 1:
                if area_coverage_ratio - main_marker_list[i].coverage_ratio <= marker_change_th:
                    marker_change_key[i] += 1
                else:
                    marker_change_key[i] = 0
            
            main_marker_list[i].coverage_ratio = area_coverage_ratio
            
            # 探査中心の変換
            if main_marker_list[i].coverage_ratio >= AREA_COVERAGE_THRESHOLD:
                next_theta = main_marker_list[i].vfh_using_probability()
                #next_theta = main_marker_list[i].vfh()
                next_x = OUTER_BOUNDARY * math.cos(next_theta) + main_marker_list[i].x
                next_y = OUTER_BOUNDARY * math.sin(next_theta) + main_marker_list[i].y
                
                marker_list[i].append(main_marker_list[i])
                main_marker_list[i] = Virtual_marker('marker' + str(i + 1) + '-' + str(len(marker_list[i]) + 1), next_x, next_y, main_marker_list[i])
                
                print('=' * 30)
                print(str(i + 1) + 'group marker changed. (', main_marker_list[i].x, ', ', main_marker_list[i].y, ') : ', len(marker_list[i]))
                marker_change_key[i] = 0
            elif marker_change_key[i] >= 3 and main_marker_list[i].coverage_ratio < 80.0:
                marker_list[i].append(main_marker_list[i])
                main_marker_list[i] = main_marker_list[i].parent
                #if back_key == 0:
                #    marker_list[i].append(main_marker_list[i])
                #    main_marker_list[i] = main_marker_list[i].parent
                #    back_key = 1
                #else:
                #    next_theta = random.uniform(0.0, 2.0 * math.pi)
                #    next_x = OUTER_BOUNDARY * math.cos(next_theta) + main_marker_list[i].x
                #    next_y = OUTER_BOUNDARY * math.sin(next_theta) + main_marker_list[i].y
            
                #    marker_list[i].append(main_marker_list[i])
                #    main_marker_list[i] = Virtual_marker('marker' + str(i + 1) + '-' + str(len(marker_list[i]) + 1), next_x, next_y, main_marker_list[i])
                #    back_key = 0
                    
                print('=' * 30)
                print(str(i + 1) + 'group marker changed. (', main_marker_list[i].x, ', ', main_marker_list[i].y, ') : ', len(marker_list[i]))
                marker_change_key[i] = 0
            
            for j in range(AREA_COVERAGE_STEP):
                for k in range(len(red_list[i])):
                    red_list[i][k].step_motion(obstacle_list, main_marker_list[i], grid_map)
                    red_df_list[i][k] = pd.concat([red_df_list[i][k], red_list[i][k].get_arguments()])
                
                    if red_list[i][k].collision_flag == 1:
                        collision_counter[i] += 1
                        main_marker_list[i].calculate_collision_df(red_list[i][k].x, red_list[i][k].y)
        
        ax.pcolormesh(X, Y, grid_map, cmap = 'brg', edgecolors = 'black')
        ax.set_title('grid_map')
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