import numpy as np
import pandas as pd

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


    def obstacle_size(self):
        return self.h * self.w

#右方向に細長いマップ例1
def obstacle_map_ex1():
    obstacle_list = []
    obstacle_df_list = []
    obstacle1 = Obstacle_square(-100, 25, 5, 200)
    obstacle2 = Obstacle_square(-100, -30, 5, 200)
    obstacle3 = Obstacle_square(-100, -25, 50, 5)
    obstacle4 = Obstacle_square(95, -25, 50, 5)
    obstacle5 = Obstacle_square(-95, 10, 15, 40)
    obstacle6 = Obstacle_square(-95, -25, 18, 50)
    obstacle7 = Obstacle_square(-10, 15, 10, 20)
    obstacle8 = Obstacle_square(-5, -25, 25, 40)
    obstacle9 = Obstacle_square(50, 10, 5, 20)
    obstacle10 = Obstacle_square(70, 10, 15, 5)
    obstacle11 = Obstacle_square(-25, -20, 5, 5)
    obstacle12 = Obstacle_square(70, -10, 10, 10)
    obstacle13 = Obstacle_square(-35, 10, 10, 10)
    
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
    obstacle_list.append(obstacle11)
    obstacle_list.append(obstacle12)
    obstacle_list.append(obstacle13)
    
    map_obstacle_size = 0
    for i in range(len(obstacle_list)):
        map_obstacle_size += obstacle_list[i].obstacle_size()
    
    for i in range(len(obstacle_list)):
        df = pd.DataFrame()
        df = pd.DataFrame(obstacle_list[i].get_arguments())
        obstacle_df_list.append(df)
    
    return obstacle_list, obstacle_df_list, map_obstacle_size


def obstacle_map_ex2():
    obstacle_list = []
    obstacle_df_list = []
    obstacle1 = Obstacle_square(-100, 45, 5, 200)
    obstacle2 = Obstacle_square(-100, -50, 5, 200)
    obstacle3 = Obstacle_square(-100, -45, 90, 5)
    obstacle4 = Obstacle_square(95, -45, 90, 5)
    obstacle5 = Obstacle_square(-95, 10, 35, 40)
    obstacle6 = Obstacle_square(-95, -45, 35, 50)
    obstacle7 = Obstacle_square(-10, 5, 40, 20)
    obstacle8 = Obstacle_square(-5, -45, 25, 40)
    obstacle9 = Obstacle_square(50, 10, 10, 20)
    obstacle10 = Obstacle_square(70, 10, 35, 10)
    obstacle11 = Obstacle_square(-25, -35, 10, 10)
    obstacle12 = Obstacle_square(70, -25, 10, 10)
    obstacle13 = Obstacle_square(-35, 25, 10, 10)
    
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
    obstacle_list.append(obstacle11)
    obstacle_list.append(obstacle12)
    obstacle_list.append(obstacle13)
    
    map_obstacle_size = 0
    for i in range(len(obstacle_list)):
        map_obstacle_size += obstacle_list[i].obstacle_size()
    
    for i in range(len(obstacle_list)):
        df = pd.DataFrame()
        df = pd.DataFrame(obstacle_list[i].get_arguments())
        obstacle_df_list.append(df)
    
    return obstacle_list, obstacle_df_list, map_obstacle_size