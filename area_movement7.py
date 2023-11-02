# 擬似障害物と線形回帰, 直線と点による障害物存在確率によるフロンティアベースの誘導
import random
import math
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize_scalar

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
        self.collision_point_x = []
        self.collision_point_y = []
        self.parent = None
    
    
    def get_arguments(self):
        return pd.DataFrame({'marker_id': [self.marker_id], 'x': [self.x], 'y': [self.y], 'point': [self.point], 'coverage_ratio': [self.coverage_ratio], 'mean': [self.mean], 'variance': [self.variance], 'inner_boundary': [self.inner_boundary], 'outer_boundary': [self.outer_boundary], 'parent': [self.parent]})
    
    
    def collision_append(self, x, y):
      self.collision_point_x.append([x])
      self.collision_point_y.append([y])
    
    
    def linear_regression(self):
        X = self.collision_point_x
        Y = self.collision_point_y

        model = LinearRegression()
        
        X_var = np.var(X)
        #Y_var = np.var(Y)
        
        if X_var == 0:
            axis = 'x'
            model.fit(Y, X)
        else:
            axis = 'y'
            model.fit(X, Y)
        
        m = float(model.coef_[0])
        n = float(model.intercept_)
        
        if axis == 'x':
            print('x = my + n : x = ', m, "y", " + ", n, ".")
        else:
            print('y = mx + n : y = ', m, "x", " + ", n, ".")
        
        return m, n, axis


# 仮想マーカー
class Virtual_marker(Real_marker):
    def __init__(self,marker_id, x, y, parent):
        super().__init__(marker_id, x, y)
        self.parent = parent
    
    
    def get_arguments(self):
        arguments = super().get_arguments()
        return arguments
    
    
    def collision_append(self, x, y):
        super().collision_append(x, y)
    
    
    def linear_regression(self):
        linear_regression = super().linear_regression()
        return linear_regression


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


# 擬似障害物
class Pseudo_obstacle:
    def __init__(self, center_x, center_y):
        self.center_x = center_x
        self.center_y = center_y
        self.center_point = np.array([self.center_x, self.center_y])
        self.point_list = [np.array(self.center_x, self.center_y)]
    
    
    def get_arguments(self):
        return pd.DataFrame({'center_x': [self.center_x], 'center_y': [self.center_y], 'center_point': [self.center_point], 'height' : [self.h], 'width': [self.w]})
    
    
    def decision_center_point(self):
        mean_point = np.mean(self.point_list, axis=0)
        self.center_x = mean_point[0]
        self.center_y = mean_point[1]
        self.center_point = mean_point
    
    
    # 凸包アルゴリズム(Graham's Scan)
    def convex_hull(self):
        # 凸包を計算
        #points = np.column_stack((self.point_x_list, self.point_y_list))
        hull = ConvexHull(self.point_list)
        
        # 凸包の頂点を取得
        hull_vertices = hull.vertices
        
        # 凸包の頂点を時計回りまたは反時計回りに並べ替える
        #hull_points = points[hull_vertices]
        #if np.cross(hull_points[1] - hull_points[0], hull_points[2] - hull_points[0]) < 0:
        #    hull_points = hull_points[::-1]
        
        return hull_vertices
    
    
    # 障害物があるかの判定
    def pseudo_obstacle_judge(self, x, y):
        if has_unique_3values(self.point_list) == False:
            if (x - self.center_x) ** 2 + (y - self.center_y) ** 2 <= 1:
                return False
            else:
                return True
        else:
            hull_vertices = self.convex_hull()
        
            conditions = []
        
            for i in range(len(hull_vertices)):
                x1, y1 = hull_vertices[i]
                x2, y2 = hull_vertices[(i + 1) % len(hull_vertices)]
                a = y1 - y2
                b = x2 - x1
                c = x1 * y2 - x2 * y1

                # (x, y)が直線(ax + by + c <= 0)の左側にあることを確認
                condition = a * x + b * y + c <= 0
                conditions.append(condition)
        
            is_inside = all(conditions)
        
            if is_inside:
                # 領域内部にある
                return False
            else:
                # 領域外部にある
                return True
    
    # 障害物群への追加
    def pseudo_obstacle_area(self, x, y):
        if has_unique_3values(self.point_list) == False:
            if (x - self.center_x) ** 2 + (y - self.center_y) ** 2 <= 1:
                self.point_list.append(np.array([x, y]))
                self.decision_center_point()
                return True
            else:
                return False
        else:
            hull_vertices = self.convex_hull()
        
            conditions = []
        
            distance_threshold = 1.0
        
            for i in range(len(hull_vertices)):
                x1, y1 = hull_vertices[i]
                x2, y2 = hull_vertices[(i + 1) % len(hull_vertices)]
        
                # 凸多角形の各辺のベクトル
                edge_vector = np.array([x2 - x1, y2 -y1])
                # 指定した点から凸多角形の辺までのベクトル
                point_to_edge_vector = np.array([x - x1, y - y1])
                # ベクトル間の距離(外積の絶対値)
                distance = np.abs(np.cross(edge_vector, point_to_edge_vector)) / np.linalg.norm(edge_vector)
                # 距離が指定した閾値以上であることを確認
                condition = distance >= distance_threshold
                conditions.append(condition)
        
            is_far_away = all(conditions)
        
            if is_far_away:
                # 凸多角形の領域から {distance_threshold} より大きい距離にある
                return True
            else:
                # 凸多角形の領域から {distance_threshold} より大きい距離にない
                self.point_list.append(np.array([x, y]))
                self.decision_center_point()
                return False
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


# 探査領域と回帰直線の交点
def find_intersections(m, n, axis, a, b, r):
    print("m : ", m, ", n : ", n, ", axis : ", axis, "a : ", a, "b : ", b, ", r : ", r)
    
    A = 1 + m ** 2
    B = 2 * (m * (n - b) - a)
    C = a ** 2 + (n - b) ** 2 - r ** 2
    
    if axis == 'x':
        B = 2 * (m * (n - a) - b)
        C = (n - a) ** 2 + b ** 2 - r ** 2

    discriminant = B ** 2 - 4 * A * C

    if discriminant < 0:
        return None
    
    
    p1 = (-B + math.sqrt(discriminant)) / 2 * A
    p2 = (-B - math.sqrt(discriminant)) / 2 * A

    p3 = m * p1 + n
    p4 = m * p2 + n

    
    if axis == 'x':
        return [np.array([p3, p1]), np.array([p4, p2])]
    else:
        return [np.array([p1, p3]), np.array([p2, p4])]


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


def has_unique_3values(ndarray_list):
    if len(ndarray_list) < 3:
        return False
    
    point_list = []
    for i in range(len(ndarray_list)):
        point = ndarray_list[i].tolist()
        point_list.append(point)
    tuples = [tuple(point) for point in ndarray_list]
    
    unique_tuples = set(tuples)
    
    # ユニークな要素の数が3つ以上であればTrueを返す
    return len(unique_tuples) >= 3
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
    obstacle1 = Obstacle_square(-50, 45, 5, 100)
    obstacle2 = Obstacle_square(-50, -50, 5, 100)
    obstacle3 = Obstacle_square(-50, -50, 100, 5)
    obstacle4 = Obstacle_square(45, -50, 100, 5)
    obstacle5 = Obstacle_square(10, 10, 10, 30)
    obstacle_list.append(obstacle1)
    obstacle_list.append(obstacle2)
    obstacle_list.append(obstacle3)
    obstacle_list.append(obstacle4)
    obstacle_list.append(obstacle5)
    
    obstacle_df_list = []
    for i in range(len(obstacle_list)):
        df = pd.DataFrame()
        df = pd.DataFrame(obstacle_list[i].get_arguments())
        obstacle_df_list.append(df)
    
    # 実マーカーの作成
    marker1 = Real_marker('marker1', -30, 0)
    marker2 = Real_marker('sub_marker', 30, 0)
    main_marker = marker1
    
    # マーカー用リスト
    marker_list = []
    
    marker_change_key = 0
    marker_change_th = marker_change_threshold()
    collision_counter = 0
    
    # 擬似障害物用リスト
    pseudo_obstacle_list = []
    
    # Redの生成
    red_list = []
    red_df_list = []
    red_num = int(input('Please enter the number of red. : '))
    
    
    def next_marker_point_judgement(theta):
        # 網羅率による探査済み領域の多さ
        def f(x, y):
            area_count = 0
            explored_count = 0
        
            for i in range(round(-OUTER_BOUNDARY), round(OUTER_BOUNDARY + 1)):
                for j in range(round(-OUTER_BOUNDARY), round(OUTER_BOUNDARY + 1)):
                    distance = math.sqrt(i ** 2 + j ** 2)
                    if INNER_BOUNDARY < distance < OUTER_BOUNDARY:
                        area_count += 1
                        if  0 <= round(CENTER_Y + y - i) <= MAP_HEIGHT - 1 and 0 <= round(CENTER_X + x - j) <= MAP_WIDTH - 1:
                            if grid_map[round(CENTER_Y + y - i)][round(CENTER_X + x - j)] >= 1:
                                explored_count += 1
    
            return explored_count / area_count
    
    
        # 凸包アルゴリズムによる障害物の存在確率
        def g(x, y):
            sigma_2 = 1.0
            mu = 0.0

            for i in range(len(pseudo_obstacle_list)):
                if pseudo_obstacle_list[i].pseudo_obstacle_judge(x, y) == False:
                    return float('inf')
            else:
                counter = 0 
                d = 0
                for i in range(len(pseudo_obstacle_list)):
                    if (pseudo_obstacle_list[i].center_x - main_marker.x) ** 2 + (pseudo_obstacle_list[i].center_y - main_marker.y) ** 2 <= main_marker.outer_boundary:
                        for j in range(len(pseudo_obstacle_list[i].point_list)):
                            d += math.sqrt(x - pseudo_obstacle_list[i].point_list[j][0]) ** 2 + (y - pseudo_obstacle_list[i].point_list[j][1]) ** 2
                            counter += 1
                
                if counter == 0:
                    return 0
                else:
                    d_mean = d / counter
                    return (1 / sigma_2 * math.sqrt(2 * math.pi)) * math.exp(-((d_mean - mu) ** 2) / 2 * sigma_2)
        
        # 最適化用の重み
        a = 3
        b = 7
        
        x = OUTER_BOUNDARY * math.cos(theta) + main_marker.x
        y = OUTER_BOUNDARY * math.sin(theta) + main_marker.y
        
        return a * f(x, y) + b * g(x, y)
    
    
    for i in range(red_num):
        red_id = 'red' + str(i + 1)
        red = Red(red_id, main_marker)
        red_list.append(red)
        red_df = pd.DataFrame()
        red_df = pd.DataFrame()
        red_df = pd.DataFrame(red_list[i].get_arguments())
        red_df_list.append(red_df)
    
    while True:
        if len(marker_list) == 20:
            print("Exploration completed. marker change count 30.")
            break
        
        # 領域網羅率計算
        area_coverage_ratio = area_coverage_calculation(grid_map, main_marker)
        print(red_list[0].step, " step : area_coverage_ratio : ", area_coverage_ratio)
        print("-" * 30)
        if collision_counter >= 1:
            if area_coverage_ratio - main_marker.coverage_ratio <= marker_change_th:
                marker_change_key += 1
            else:
                marker_change_key = 0
        
        main_marker.coverage_ratio = area_coverage_ratio
        
        # 探査中心の変換
        if main_marker.coverage_ratio >= AREA_COVERAGE_THRESHOLD or marker_change_key >= 3:
            #result = minimize_scalar(next_marker_point_judgement, bounds=(0, 2.0 * math.pi), method="bounded")
            #next_theta = result.x
            #next_x = OUTER_BOUNDARY * math.cos(next_theta) + main_marker.x
            #next_y = OUTER_BOUNDARY * math.sin(next_theta) + main_marker.y
            
            if len(main_marker.collision_point_x) >= 2:
                linear_m, linear_n, axis = main_marker.linear_regression()
                intersection_point_list = find_intersections(linear_m, linear_n, axis, main_marker.x, main_marker.y, OUTER_BOUNDARY)
                print("intersection_point(交点) : ", intersection_point_list)
                if intersection_point_list == None:
                    while True:
                        result = minimize_scalar(next_marker_point_judgement, bounds=(0, 2 * np.pi), method='bounded')
                        next_theta = result.x
                        next_x = OUTER_BOUNDARY * math.cos(next_theta) + main_marker.x
                        next_y = OUTER_BOUNDARY * math.sin(next_theta) + main_marker.y
                        if main_marker.parent == None:
                            break
                        else:
                            if (next_x - main_marker.parent.x) ** 2 +(next_y - main_marker.parent.y) ** 2 >= OUTER_BOUNDARY ** 2:
                                break
                        
                else:
                    # theta1 < theta2(0~360)
                    theta1, theta2 = intersection_azimuth_adjustment(main_marker, intersection_point_list[0], intersection_point_list[1])
                    while True:
                        if theta1 + math.pi < theta2:
                            result = minimize_scalar(next_marker_point_judgement, bounds=(theta1, theta2), method='bounded')
                            next_theta = result.x
                        else:
                            next_theta_list = []
                            
                            result1 = minimize_scalar(next_marker_point_judgement, bounds=(0, theta1), method='bounded')
                            result2 = minimize_scalar(next_marker_point_judgement, bounds=(theta2, 2.0 * math.pi), method='bounded')
                            next_theta_list.append(result1.x)
                            next_theta_list.append(result2.x)
                            next_theta = random.choice(next_theta_list)
                        
                        next_x = OUTER_BOUNDARY * math.cos(next_theta) + main_marker.x
                        next_y = OUTER_BOUNDARY * math.sin(next_theta) + main_marker.y
                        if main_marker.parent == None:
                            break
                        else:
                            if (next_x - main_marker.parent.x) ** 2 +(next_y - main_marker.parent.y) ** 2 >= OUTER_BOUNDARY ** 2:
                                break
            else:
                while True:
                    result = minimize_scalar(next_marker_point_judgement, bounds=(0, 2 * math.pi), method='bounded')
                    next_theta = result.x
                    next_x = OUTER_BOUNDARY * math.cos(next_theta) + main_marker.x
                    next_y = OUTER_BOUNDARY * math.sin(next_theta) + main_marker.y
                    if main_marker.parent == None:
                        break
                    else:
                        if (next_x - main_marker.parent.x) ** 2 + (next_y - main_marker.parent.y) ** 2 >= OUTER_BOUNDARY ** 2:
                            break
            
            marker_list.append(main_marker)
            main_marker = Virtual_marker('marker' + str(len(marker_list) + 1), next_x, next_y, main_marker)
            print("=" * 30)
            print("marker changed. (", main_marker.x, ", ", main_marker.y, ") : ", len(marker_list))
            marker_change_key = 0
        
        for i in range(AREA_COVERAGE_STEP):
            for j in range(red_num):
                red_list[j].step_motion(obstacle_list, main_marker, grid_map)
                red_df_list[j] = pd.concat([red_df_list[j], red_list[j].get_arguments()])
                if red_list[j].collision_flag == 1:
                    main_marker.collision_append(red_list[j].x, red_list[j].y)
                    collision_counter +=1
                    if len(pseudo_obstacle_list) == 0:
                        pseudo_obstacle = Pseudo_obstacle(red_list[j].x, red_list[j].y)
                        pseudo_obstacle_list.append(pseudo_obstacle)
                    else:
                        for i in range(len(pseudo_obstacle_list)):
                            if pseudo_obstacle_list[i].pseudo_obstacle_area(red_list[j].x, red_list[j].y) == True:
                                break
                        else:
                            pseudo_obstacle = Pseudo_obstacle(red_list[j].x, red_list[j].y)
                            pseudo_obstacle_list.append(pseudo_obstacle)
                    
                    collision_counter +=1
        
        ax.pcolormesh(X, Y, grid_map, cmap = 'brg', edgecolors = 'black')
        ax.set_title('grid_map')
        plt.pause(0.01)
    
    
    # データフレームの作成
    # Red
    for i in range(red_num):
        red_df_list[i].to_csv(SAVE_DIRECTORY  + red_list[i].red_id + '.csv')
    
    # Marker
    for i in range(len(marker_list)):
        if i == 0: 
            marker_df = pd.DataFrame(marker_list[i].get_arguments())
        else:
            marker_df = pd.concat([marker_df, marker_list[i].get_arguments()])
        marker_df.to_csv(SAVE_DIRECTORY + 'marker.csv')
    
    # obstacle
    for i in range(len(obstacle_df_list)):
        obstacle_df_list[i].to_csv(SAVE_DIRECTORY + 'obstacle' + str(i + 1) + '.csv')
    
    # pseudo obstacle
    for i in range(len(pseudo_obstacle_list)): 
        if i == 0:
            pseudo_obstacle_df = pd.DataFrame(pseudo_obstacle_list[i].get_arguments())
        else:
            pseudo_obstacle_df = pd.concat([pseudo_obstacle_df, pseudo_obstacle_list[i].get_arguments()])
    pseudo_obstacle_df.to_csv(SAVE_DIRECTORY + 'pseudo_obstacle.csv')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopped by keyboard input(ctrl-c)")
        sys.exit()