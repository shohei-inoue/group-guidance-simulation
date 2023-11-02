''' 探査領域を8領域に分割した場合'''
import random
import math
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

# -------------------------------- const parameter ---------------------------------------------------
MAX_MOVEMENT = 3.0                          # 最大移動量
MIN_MOVEMENT = 1.0                          # 最小移動量
MAX_BOIDS_MOVEMENT = 3.0                    # boids判定時の最大移動量
MIN_BOIDS_MOVEMENT = 1.0                    # boids判定時の最小移動量
OUTER_BOUNDARY = 10.0                       # マーカ-の外側境界
INNER_BOUNDARY = 5.0                        # マーカーの内側境界
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


# ----------------------------------- class ----------------------------------------------------------

# ランダムウォーク
class Random_walk:
    def __init__(self):
        self.x = random.uniform(-25, -20)
        self.y = random.uniform(-5, 10)
        #self.x = random.uniform(-math.floor(MAP_WIDTH / 2), math.floor(MAP_WIDTH / 2))
        #self.y = random.uniform(-math.floor(MAP_HEIGHT / 2), math.floor(MAP_HEIGHT / 2))
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
    
    
    def get_arguments(self):
        return pd.DataFrame({'marker_id': [self.marker_id], 'x': [self.x], 'y': [self.y], 'point': [self.point], 'coverage_ratio': [self.coverage_ratio], 'mean': [self.mean], 'variance': [self.variance], 'inner_boundary': [self.inner_boundary], 'outer_boundary': [self.outer_boundary]})


# 仮想マーカー
class Virtual_marker:
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
    
    
    def get_arguments(self):
        return pd.DataFrame({'marker_id': [self.marker_id], 'x': [self.x], 'y': [self.y], 'point': [self.point], 'coverage_ratio': [self.coverage_ratio], 'mean': [self.mean], 'variance': [self.variance], 'inner_boundary': [self.inner_boundary], 'outer_boundary': [self.outer_boundary]})


#障害物
#class Obstacle_circle:
#    def __init__(self, x, y, r):
#        self.x = x
#        self.y = y
#        self.point = np.array([self.x, self.y]) # 中心
#        self.r = r


class Obstacle_square:
    def __init__(self, x, y, h, w):
        self.x = x
        self.y = y
        self.point = np.array([self.x, self.y]) # 左下頂点
        self.h = h
        self.w = w
    
    
    def get_arguments(self):
        return pd.DataFrame({'x': [self.x], 'y': [self.y], 'point': [self.point], 'height': [self.h], 'width': [self.w]})

# Red
class Red(Random_walk):
    def __init__(self, red_id, marker, **rest):
        super().__init__(**rest)
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
        
        if (self.x - marker.x) < 0:
            if (self.y - marker.y) >= 0:
                azimuth = np.rad2deg(math.pi) - azimuth
            else:
                azimuth += np.rad2deg(math.pi)
        else:
            if (self.y - marker.y) < 0:
               azimuth = np.rad2deg(2.0 * math.pi) - azimuth
        return azimuth
    
    
    def get_arguments(self):
        return pd.DataFrame({'step': [self.step], 'x': [self.x], 'y': [self.y], 'point': [self.point], 'amount_of_movement': [self.amount_of_movement], 'direction_angle': [self.direction_angle], 'distance': [self.distance], 'azimuth': [self.azimuth], 'collision_flag': [self.collision_flag], 'boids_flag': [self.boids_flag], 'estimated_probability': [self.estimated_probability]})
    
    
    # boids判定
    def boids_judgement(self, marker):
        self.distance = np.linalg.norm(self.point - marker.point)   # add
        if self.distance > marker.outer_boundary:
            self.boids_flag = 1
        elif self.distance < marker.inner_boundary:
            self.boids_flag = 2
        else:
            self.boids_flag = 0
    
    
    # boids行動
    def boids_behavior(self, marker):
        self.direction_angle = self.azimuth    # <- self.boids_flag == 2
        if self.boids_flag == 1:
            if (self.y - marker.y) >= 0:
                self.direction_angle += np.rad2deg(math.pi)
            else:
                self.direction_angle -= np.rad2deg(math.pi)
        
        amount_of_movement = random.uniform(MIN_BOIDS_MOVEMENT, MAX_BOIDS_MOVEMENT)
        dx = amount_of_movement * math.cos(math.radians(self.direction_angle))
        dy = amount_of_movement * math.sin(math.radians(self.direction_angle))
        prediction_point = np.array([self.x + dx, self.y + dy])
        return prediction_point
    
    
    # 障害物判定 確認必要
    def obstacle_judgement(self, obstacle_list, prediction_point):
        intersection = []
        for i in range(len(obstacle_list)):
            if (obstacle_list[i].x <= prediction_point[0] <= obstacle_list[i].x + obstacle_list[i].w) and (obstacle_list[i].y <= prediction_point[1] <= obstacle_list[i].y + obstacle_list[i].h):
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
        self.direction_angle = (self.direction_angle + random.uniform(90, 270)) % np.rad2deg(math.pi * 2.0)
        amount_of_movement = random.uniform(MIN_MOVEMENT, MAX_MOVEMENT)
        dx = amount_of_movement * math.cos(math.radians(self.direction_angle))
        dy = amount_of_movement * math.sin(math.radians(self.direction_angle))
        prediction_point = np.array([self.x + dx, self.y + dy])
        return prediction_point
    
    
    # 棄却決定
    def rejection_decision(self, marker):
        # 棄却
        while True:
            direction_angle = np.rad2deg(random.uniform(0, 2.0 * math.pi))
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
        prediction_point = np.array([])
        if self.collision_flag == 1:
            prediction_point = self.avoidance_behavior()
        else:
            #boids判定
            self.boids_judgement(marker)
            if self.boids_flag != 0:
                prediction_point = self.boids_behavior(marker)
            else:
                #棄却判定
                prediction_point = self.rejection_decision(marker)
        # 障害物判定
        inspection_point = self.obstacle_judgement(obstacle_list, prediction_point)
        if self.collision_flag == 1:
            self.amount_of_movement = np.linalg.norm(inspection_point - self.point)
            self.point = inspection_point
        else:
            self.amount_of_movement = np.linalg.norm(prediction_point - self.point)
            self.point = prediction_point
        
        # grid map用コードの記述
        create_map(map, self.point, self.x, self.y)
        
        self.x = self.point[0]
        self.y = self.point[1]
        self.distance = np.linalg.norm(self.point - marker.point)
        self.azimuth = self.azimuth_adjustment(marker)
        self.step += 1


# ------------------------------- function -----------------------------------------------------


#提案分布
def distribution(distance, mean, variance):
    return 1 / math.sqrt(2 * math.pi) * math.exp(-(distance - mean) ** 2 / (2 * variance ** 2))

# グリットマップへの追加
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


# 障害物判定用領域決定関数
def area_definition(marker, red):
    theta1 = math.degrees(5.0 * math.pi / 16.0)
    theta2 = math.degrees(math.pi / 16.0)
    p1_x = OUTER_BOUNDARY * math.cos(theta1) + marker.x
    p1_y = OUTER_BOUNDARY * math.sin(theta1) + marker.y
    p2_x = OUTER_BOUNDARY * math.cos(theta2) + marker.x
    p2_y = OUTER_BOUNDARY * math.sin(theta2) + marker.y
    p3_x = OUTER_BOUNDARY * math.cos(theta2) + marker.x
    p3_y = -OUTER_BOUNDARY * math.sin(theta2) + marker.y
    p4_x = OUTER_BOUNDARY * math.cos(theta1) + marker.x
    p4_y = -OUTER_BOUNDARY * math.sin(theta1) + marker.y
    y_th1 = marker.y + (marker.y - p1_y) / (marker.x - p1_x) * (red.x - marker.x)
    y_th2 = marker.y + (marker.y - p2_y) / (marker.x - p2_x) * (red.x - marker.x)
    y_th3 = marker.y + (marker.y - p3_y) / (marker.x - p3_x) * (red.x - marker.x)
    y_th4 = marker.y + (marker.y - p4_y) / (marker.x - p4_x) * (red.x - marker.x) 
    
    if red.x - marker.x >= 0:
        if red.y <= y_th4:
            return int(6)
        elif y_th4 <= red.y <= y_th3:
            return int(7)
        elif y_th3 <= red.y <= y_th2:
            return int(0)
        elif y_th2 <= red.y <= y_th1:
            return int(1)
        elif y_th1 <= red.y:
            return int(2)
    else:
        if y_th4 <= red.y:
            return int(2)
        elif y_th3 <= red.y <= y_th4:
            return int(3)
        elif y_th2 <= red.y <= y_th3:
            return int(4)
        elif y_th1 <= red.y <= y_th2:
            return int(5)
        else:
            return int(6)

# -------------------------------- main ---------------------------------------------------------
def main():
    # グリッドマップの作成
    grid_map = np.zeros([MAP_HEIGHT, MAP_WIDTH])
    x = range(-CENTER_X, CENTER_X + 1)
    y = range(-CENTER_Y, CENTER_Y + 1)
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots()
    
    
    # 障害物の作成
    obstacle_list = []
    
    # 障害物の記述
    obstacle1 = Obstacle_square(-50, 45, 5, 100)
    obstacle2 = Obstacle_square(-50, -50, 5, 100)
    obstacle3 = Obstacle_square(-50, -50, 100, 5)
    obstacle4 = Obstacle_square(45, -50, 100, 5)
    obstacle5 = Obstacle_square(-15, 0, 50, 30)
    obstacle6 = Obstacle_square(-1, -50, 40, 30)
    obstacle_list.append(obstacle1)
    obstacle_list.append(obstacle2)
    obstacle_list.append(obstacle3)
    obstacle_list.append(obstacle4)
    obstacle_list.append(obstacle5)
    obstacle_list.append(obstacle6)
    
    obstacle_df_list = []
    for i in range(len(obstacle_list)):
        df = pd.DataFrame() 
        df = pd.DataFrame(obstacle_list[i].get_arguments())
        obstacle_df_list.append(df)
    
    
    # 実マーカーの作成
    marker1 = Real_marker('rm1', -30, 0)
    marker2 = Real_marker('rm2', 30, 0)
    marker3 = Real_marker('rm3', -30, 25)
    marker4 = Real_marker('rm4', 30, 25)
    main_marker = marker1
    main_marker_collision_list = []
    
    
    # 仮想マーカーの作成
    vmarker_list = []
    # add-------------------------
    vmarker_collision_list = []
    # ----------------------------
    vmarker_num_x = MAP_WIDTH / (OUTER_BOUNDARY)
    vmarker_num_y = MAP_HEIGHT / (OUTER_BOUNDARY)
    vmarker_df = pd.DataFrame()
    for i in range(round(vmarker_num_y)):
        for j in range(round(vmarker_num_x)):
            vmarker = Virtual_marker('vm' + str(round(i * vmarker_num_y + j + 1)), -CENTER_X + OUTER_BOUNDARY * (1 + j), CENTER_Y - OUTER_BOUNDARY * (1 + i))
            vmarker_list.append(vmarker)
            vmarker_df = pd.concat([vmarker_df, vmarker.get_arguments()])
    
    # add-----------------------------------
    for i in range(len(vmarker_list)):
        vmarker_collision_list.append([])
        for j in range(8):
          vmarker_collision_list[i].append(0)
    # ---------------------------------------
    
    
    coverage_ratio_bef = 0  # メインマーカーの前の網羅率を格納
    marker_change_key = 0
    marker_change_th = marker_change_threshold()
    
    
    # 網羅率による終了用変数
    map_coverage_ratio_bef = 0
    exploration_completed_key = 0
    exploration_completed_th = exploration_completed_threshold(grid_map)
    
    # Redの生成
    red_list = []
    red_df_list = []
    red_num = int(input('Please enter the number of red. : '))
    
    for i in range(red_num):
        red_id = 'red' + str(i + 1)
        red = Red(red_id, main_marker)
        red_list.append(red)
        red_df = pd.DataFrame()
        red_df = pd.DataFrame(red_list[i].get_arguments())
        red_df_list.append(red_df)
    
    
    while True:
        # 全体網羅率計算
        map_coverage_ratio = map_coverage_calculation(grid_map)
        print('=' * 20)
        print(red_list[0].step, " step : map_coverage_ratio : ", map_coverage_ratio)
        if map_coverage_ratio - map_coverage_ratio_bef <= exploration_completed_th:
            exploration_completed_key += 1
        else:
            exploration_completed_key = 0
        map_coverage_ratio_bef = map_coverage_ratio
        if map_coverage_ratio >= ALL_AREA_COVERAGE_THRESHOLD or exploration_completed_key >= 10:
            print("Exploration completed.")
            break
        
        # 領域網羅率計算
        area_coverage_ratio = area_coverage_calculation(grid_map, main_marker)
        print(red_list[0].step, " step : area_coverage_ratio : ", area_coverage_ratio)
        for i in range(len(vmarker_list)):
            # メインマーカーの網羅率格納
            if vmarker_list[i].marker_id == main_marker.marker_id:
                coverage_ratio_bef = vmarker_list[i].coverage_ratio # 1ステップ前の網羅率
                vmarker_list[i].coverage_ratio = area_coverage_ratio
                if vmarker_list[i].coverage_ratio - coverage_ratio_bef <= marker_change_th:
                    marker_change_key += 1
                else:
                    marker_change_key = 0
                break
        
        # 探査中心の変換
        if area_coverage_ratio >= AREA_COVERAGE_THRESHOLD or marker_change_key >= 3:
            # add---------------
            for i in range(len(vmarker_list)):
                if vmarker_list[i].marker_id == main_marker.marker_id:
                    for j in range(len(vmarker_collision_list[i])):
                        if vmarker_collision_list[i][j] >= 10:
                            main_marker_collision_list.append(j)
                break
            # ------------------
            next_main_list = []
            coverage_ratio_list = []
            for i in range(len(vmarker_list)):
                distance = math.sqrt((vmarker_list[i].x - main_marker.x) ** 2 + (vmarker_list[i].y - main_marker.y) ** 2)
                if distance <= OUTER_BOUNDARY * math.sqrt(2) and vmarker_list[i].marker_id != main_marker.marker_id:
                    # add--------------
                    x_direction = vmarker_list[i].x - main_marker.x
                    y_direction = vmarker_list[i].y - main_marker.y
                    for j in range(len(main_marker_collision_list)):
                        if main_marker_collision_list[j] == 0 and x_direction > 0 and y_direction == 0:
                            print("couldn't change marker direction +x =y : 0")
                        elif main_marker_collision_list[j] == 1 and x_direction > 0 and y_direction == 0:
                            print("couldn't change marker direction +x +y : 1")
                        elif main_marker_collision_list[j] == 2 and x_direction == 0 and y_direction > 0:
                            print("couldn't change marker direction =x +y : 2")
                        elif main_marker_collision_list[j] == 3 and x_direction < 0 and y_direction > 0:
                            print("couldn't change marker direction -x +y : 3")
                        elif main_marker_collision_list[j] == 4 and x_direction < 0 and y_direction == 0:
                            print("couldn't change marker direction -x =y : 4")
                        elif main_marker_collision_list[j] == 5 and x_direction < 0 and y_direction < 0:
                            print("couldn't change marker direction -x -y : 5")
                        elif main_marker_collision_list[j] == 6 and x_direction == 0 and y_direction < 0:
                            print("couldn't change marker direction =x -y : 6")
                        elif main_marker_collision_list[j] == 7 and x_direction > 0 and y_direction < 0:
                            print("couldn't change marker direction +x -y : 7")
                        else:
                    # -----------------
                            vmarker_list[i].coverage_ratio = area_coverage_calculation(grid_map, vmarker_list[i])
                            next_main_list.append(vmarker_list[i])
                            coverage_ratio_list.append(vmarker_list[i].coverage_ratio)
                    else:
                        vmarker_list[i].coverage_ratio = area_coverage_calculation(grid_map, vmarker_list[i])
                        next_main_list.append(vmarker_list[i])
                        coverage_ratio_list.append(vmarker_list[i].coverage_ratio)
            
            
            
            main_marker = next_main_list[coverage_ratio_list.index(min(coverage_ratio_list))]
            print('Marker changed. : (', str(main_marker.x), ', ', str(main_marker.y), ')')
            main_marker_collision_list = []
            marker_change_key = 0
            coverage_ratio_bef = 0
            
            
        for i in range(AREA_COVERAGE_STEP):
            for j in range(red_num):
                red_list[j].step_motion(obstacle_list, main_marker, grid_map)
                red_df_list[j] = pd.concat([red_df_list[j], red_list[j].get_arguments()])
                # add--------------------
                if red_list[j].collision_flag == 1:
                    for k in range(len(vmarker_list)):
                      if vmarker_list[k].marker_id == main_marker.marker_id:
                          vmarker_collision_list[k][area_definition(main_marker, red_list[j])] += 1
                # -----------------------
        
        ax.pcolormesh(X, Y, grid_map, cmap='brg', edgecolors='black')
        ax.set_title('grid map')
        plt.pause(0.01)
    
    # データフレームの作成
    #Red
    for i in range(red_num):
        red_df_list[i].to_csv(SAVE_DIRECTORY  + red_list[i].red_id + '.csv')
    # Marker
    marker1_df = pd.DataFrame(marker1.get_arguments())
    marker2_df = pd.DataFrame(marker2.get_arguments())
    marker1_df.to_csv(SAVE_DIRECTORY + 'marker1.csv')
    marker2_df.to_csv(SAVE_DIRECTORY + 'marker2.csv')
    vmarker_df.to_csv(SAVE_DIRECTORY + 'vmarker.csv')
    # obstacle
    for i in range(len(obstacle_df_list)):
        obstacle_df_list[i].to_csv(SAVE_DIRECTORY + 'obstacle' + str(i + 1) + '.csv')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopped by keyboard input(ctrl-c)")
        sys.exit()