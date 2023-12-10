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


#右方向に細長いマップ例1
def obstacle_map_ex1():
    obstacle_list = []
    obstacle_df_list = []
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
    
    for i in range(len(obstacle_list)):
        df = pd.DataFrame()
        df = pd.DataFrame(obstacle_list[i].get_arguments())
        obstacle_df_list.append(df)
    
    return obstacle_list, obstacle_df_list