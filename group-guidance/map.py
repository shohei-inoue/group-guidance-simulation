import numpy as np
import pandas as pd
import math
import params


# 1群の探査領域周りのグリッドマップ
def grid_map():
    map_scale = np.zeros([int(2 * params.OUTER_BOUNDARY), int(2 * params.OUTER_BOUNDARY)])
    x = range(round(-params.OUTER_BOUNDARY), round(params.OUTER_BOUNDARY + 1))
    y = range(round(-params.OUTER_BOUNDARY), round(params.OUTER_BOUNDARY + 1))
    X, Y = np.meshgrid(x, y)
    return map_scale, X, Y


# マップ全体のグリッドマップ
def overall_map():
    map_scale = np.zeros([params.MAP_HEIGHT, params.MAP_WIDTH])
    x = range(-params.CENTER_X, params.CENTER_X + 1)
    y = range(-params.CENTER_Y, params.CENTER_Y + 1)
    X, Y = np.meshgrid(x, y)
    
    return map_scale, X, Y


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
        if -(params.MAP_HEIGHT / 2.0) <= passing_point[i][0] <= (params.MAP_HEIGHT / 2.0) and -(params.MAP_WIDTH/ 2.0) <= passing_point[i][1] <= (params.MAP_WIDTH / 2.0):
            map[params.CENTER_Y - 1 + passing_point[i][0]][params.CENTER_X - 1 + passing_point[i][1]] += 1


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
        if marker_y - params.OUTER_BOUNDARY <= passing_point[i][0] <= marker_y + params.OUTER_BOUNDARY and marker_x - params.OUTER_BOUNDARY <= passing_point[i][1] <= marker_x + params.OUTER_BOUNDARY:
            map[round(params.OUTER_BOUNDARY - (passing_point[i][0] - marker_y) - 1)][round(params.OUTER_BOUNDARY + (passing_point[i][1] - marker_x) - 1)] += 1

# 網羅率計算 →　変更したい
def area_coverage_calculation(map, marker):
    area_count = 0
    explored_count = 0
    for i in range(round(-params.OUTER_BOUNDARY), round(params.OUTER_BOUNDARY + 1)):
        for j in range(round(-params.OUTER_BOUNDARY), round(params.OUTER_BOUNDARY + 1)):
            distance = math.sqrt(i ** 2 + j ** 2)
            if params.INNER_BOUNDARY < distance < params.OUTER_BOUNDARY:
                area_count += 1
                if 0 <= round(params.OUTER_BOUNDARY - i) <= 2 * params.OUTER_BOUNDARY - 1 and 0 <= round(params.OUTER_BOUNDARY - j) <= 2 * params.OUTER_BOUNDARY - 1:
                    if map[round(params.OUTER_BOUNDARY - i)][round(params.OUTER_BOUNDARY - j)] >= 1:
                        explored_count += 1
    
    return explored_count / area_count * 100


# 全体網羅率計算
def map_coverage_calculation(map, map_obstacle_size):
    all_count = map.size - map_obstacle_size
    explored_count = np.count_nonzero(map >= 1)
    
    return explored_count / all_count * 100.0


# 探査終了閾値
def exploration_completed_threshold(map):
    all_count = map.size
    
    return 1.5 / all_count * 100
