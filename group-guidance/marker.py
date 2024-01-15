import numpy as np
import pandas as pd
import math
from scipy.stats import vonmises
import params
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# 実マーカー
class Real_marker:
    def __init__(self, marker_id, x, y):
        self.marker_id = marker_id
        self.x = x
        self.y = y
        self.point = np.array([self.x, self.y])
        self.coverage_ratio = 0.0
        self.mean = params.MEAN
        self.variance = params.VARIANCE
        self.inner_boundary = params.INNER_BOUNDARY
        self.outer_boundary = params.OUTER_BOUNDARY
        self.collision_df = pd.DataFrame(columns=['x', 'y', 'azimuth', 'distance'])
        self.already_direction_index = []
        self.label = 0
        self.parent = None
    
    
    def get_arguments(self):
        return pd.DataFrame({'marker_id': [self.marker_id], 'x': [self.x], 'y': [self.y], 'point': [self.point], 'coverage_ratio': [self.coverage_ratio], 'mean': [self.mean], 'variance': [self.variance], 'inner_boundary': [self.inner_boundary], 'outer_boundary': [self.outer_boundary], 'label': [self.label], 'parent': [self.parent]})
    
    
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
    
    
    # 初期位置に対しての角度
    def calculate_mu_azimuth3(self, init_x, init_y):
        azimuth = 0.0
        if init_x != self.x:
            vec_d = np.array([self.x - init_x, self.y - init_y])
            vec_x = np.array([self.x - init_x, 0])
            azimuth = np.rad2deg(math.acos(vec_d @ vec_x / (np.linalg.norm(vec_d) * np.linalg.norm(vec_x))))
        
        if self.x - init_x < 0:
            if self.y - init_y >= 0:
                azimuth = np.rad2deg(math.pi) - azimuth
            else:
                azimuth += np.rad2deg(math.pi)
        else:
            if self.y - init_y < 0:
                azimuth = np.rad2deg(2.0 * math.pi) - azimuth
        
        return azimuth
    
    
    # 探査領域ごとの障害物判定をデータフレームに格納する
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
    def vfh_using_probability(self, bins=params.VFH_BINS):
        loc = self.calculate_mu_azimuth()
        #print("loc : ", loc)
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
        #    print("histogram[", str(i), "] : ", histogram[i])
        
        total_histogram = np.sum(histogram)
        normalize_histogram = np.array(histogram) / total_histogram
        
        scaled_histogram = params.VFH_MIN_VALUE + normalize_histogram * (1 - params.VFH_MIN_VALUE)
        scaled_histogram = scaled_histogram / np.sum(scaled_histogram)
        
        if loc == None:
            scaled_histogram = scaled_histogram.tolist()
            select_index = np.random.choice(np.arange(0, 16), p=scaled_histogram)
        else:
            probability_of_exploration_rate = [0 for i in range(bins)]
            for i in range(bins):
                bin_range_start = vonmises.cdf(loc = np.deg2rad(loc), kappa = params.VFH_VONMISES_KAPPA, x = np.deg2rad(i * split_arg))
                bin_range_end = vonmises.cdf(loc = np.deg2rad(loc), kappa = params.VFH_VONMISES_KAPPA, x = np.deg2rad((i + 1) * split_arg))
                probability_of_exploration_rate[i] = abs(bin_range_end - bin_range_start)
            
            probability_of_exploration_rate = np.array(probability_of_exploration_rate)
            probability_density = params.VFH_DRIVABILITY_BIAS * scaled_histogram + params.VFH_EXPLORATION_BIAS * probability_of_exploration_rate
            
            probability_density = probability_density.tolist()
            
            select_index = np.random.choice(np.arange(0, 16), p=probability_density)
            self.already_direction_index.append(select_index)
        
       #print("select_index : ", select_index, " arg : ", split_arg * (select_index + 0.5))
        
        return split_arg * (select_index + 0.5)
    
    
    def vfh_using_probability2(self, marker_list, bins = params.VFH_BINS):
        # 探査向上性が最も高くなる位置の算出
        loc = self.calculate_mu_azimuth2(marker_list)
        
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
        
        # ヒストグラムを逆数にし, 探査不可能性から探査可能性へのヒストグラムへ変換
        for i in range(bins):
            histogram[i] = 1 / histogram[i]
        
        # ヒストグラムの正規化
        total_histogram = np.sum(histogram)
        normalized_histogram = np.array(histogram) / total_histogram
        
        scaled_histogram = params.VFH_MIN_VALUE + normalized_histogram * (1 - params.VFH_MIN_VALUE)
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
                bin_range_start = vonmises.cdf(loc = np.deg2rad(loc), kappa = params.VFH_VONMISES_KAPPA, x = np.deg2rad(i * split_arg))
                bin_range_end = vonmises.cdf(loc = np.deg2rad(loc), kappa = params.VFH_VONMISES_KAPPA, x = np.deg2rad((i + 1) * split_arg))
                probability_of_exploration_rate[i] = abs(bin_range_end - bin_range_start)
        
            probability_of_exploration_rate = np.array(probability_of_exploration_rate)
            probability_density = params.VFH_DRIVABILITY_BIAS * scaled_histogram + params.VFH_EXPLORATION_BIAS * probability_of_exploration_rate
            
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
        
        return split_arg * (select_index + 0.5)
    
    
    # VFHから確率分布を作成し移動先を決定する(走行可能性による確率密度分布)
    def vfh_using_probability_only_obstacle_density(self, bins=params.VFH_BINS):
        histogram = [1 for _ in range(bins)]
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
        
        scaled_histogram = params.VFH_MIN_VALUE + normalized_histogram * (1 - params.VFH_MIN_VALUE)
        scaled_histogram = scaled_histogram / np.sum(scaled_histogram)
        
        scaled_histogram = scaled_histogram.tolist()
        select_index = np.random.choice(np.arange(0, 16), p=scaled_histogram)
        self.already_direction_index.append(select_index)
        
        return split_arg * (select_index + 0.5)
    
    
    # VFHにより最も効率の良いとされる方向を決定する(探査向上性と走行可能性)
    def vfh(self, bins=params.VFH_BINS):
        loc = self.calculate_mu_azimuth()
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
        
        scaled_histogram = params.VFH_MIN_VALUE + normalized_histogram * (1 - params.VFH_MIN_VALUE)
        scaled_histogram = scaled_histogram / np.sum(scaled_histogram)
        
        if loc == None:
            scaled_histogram = scaled_histogram.tolist()
            best_index = scaled_histogram.index(max(scaled_histogram))
        else:
            probability_of_exploration_rate = []
            for i in range(bins):
                probability_of_exploration_rate.append(0)
            
            for i in range(bins):
                bin_range_start = vonmises.cdf(loc = np.deg2rad(loc), kappa = params.VFH_VONMISES_KAPPA, x = np.deg2rad(i * split_arg))
                bin_range_end = vonmises.cdf(loc = np.deg2rad(loc), kappa = params.VFH_VONMISES_KAPPA, x = np.deg2rad((i + 1) * split_arg))
                probability_of_exploration_rate[i] = abs(bin_range_end - bin_range_start)
        
            probability_of_exploration_rate = np.array(probability_of_exploration_rate)
            probability_density = params.VFH_DRIVABILITY_BIAS * scaled_histogram + params.VFH_EXPLORATION_BIAS * probability_of_exploration_rate
            
            probability_density = probability_density.tolist()
            best_index = probability_density.index(max(probability_density))
        
        return split_arg * (best_index + 0.5)
    
    
    # VFHにより最も効率の良いとされる方向を決定する(走行可能性)
    def vfh_only_obstacle_density(self, bins=params.VFH_BINS):
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
            print("histogram[", str(i), '] : ', histogram[i])
        
        total_histogram = np.sum(histogram)
        normalized_histogram = np.array(histogram) / total_histogram
        
        scaled_histogram = params.VFH_MIN_VALUE + normalized_histogram * (1 - params.VFH_MIN_VALUE)
        scaled_histogram = scaled_histogram / np.sum(scaled_histogram)
        scaled_histogram = scaled_histogram.tolist()
        
        best_index = scaled_histogram.index(max(scaled_histogram))
        print("best_index : ", best_index, " arg : ", split_arg * (best_index + 0.5))
        
        return split_arg * (best_index + 0.5)


    # 深さ優先探索を用い, 走行可能性により方向を決定する
    def vfh_dfs(self, init_x, init_y, bins=params.VFH_BINS):
        histogram = []
        center_distance = np.linalg.norm(np.array([self.x - init_x, self.y - init_y]))
        split_arg = np.rad2deg(2.0 * math.pi) / bins
        
        for i in range(bins):
            bin_x = self.x + params.OUTER_BOUNDARY * math.cos(math.radians(split_arg * (i + 0.5)))
            bin_y = self.y + params.OUTER_BOUNDARY * math.sin(math.radians(split_arg * (i + 0.5)))
            bin_distance = np.linalg.norm(np.array([bin_x - init_x, bin_y - init_y]))
            histogram.append([i, bin_distance, 1])
        
        for i in range(self.collision_df.shape[0]):
            collision_azimuth_value = self.collision_df['azimuth'].iloc[i]
            
            for j in range(bins):
                bin_range_start = j * split_arg
                bin_range_end = (j + 1) * split_arg
                
                if bin_range_start <= collision_azimuth_value < bin_range_end:
                    histogram[j][-1] += 1
                    break
            
        for i in range(bins):
            print("histogram[", str(i), "] : ", histogram[i][-1])
        
        # ラベル付け
        label_list = []
        label_key = 0
        for i in range(bins):
            if label_key == 0:
                if histogram[i][-1] <= params.DFS_THRESHOLD:
                    label_list.append([histogram[i][0]])
                    label_key = 1
            else: # label_key == 1
                if histogram[i][-1] <= params.DFS_THRESHOLD:
                    label_list[-1].append(histogram[i][0])
                else:
                    label_key = 0
        
        if label_list != []:
            if len(label_list[0]) != bins:
                if 0 in label_list[0] and bins - 1 in label_list[-1]:
                    label_list[0] += label_list[-1]
                    del label_list[-1]
        
        if len(label_list) >= 2:
            self.label = 2
            for i in range(len(label_list)):
                if len(label_list[i]) >= 6:
                    self.label = 3
                    break
        else:
            self.label = 1
            print("s_t = ",  str(len(label_list)))
            for i in range(len(label_list)):
                if len(label_list[i]) >= 6:
                    self.label = 3
        
        for i in range(len(self.already_direction_index)):
            histogram[int(self.already_direction_index[i])][-1] = float('inf')
        
        next_candidate = []
        for i in range(bins):
            if histogram[i][1] > center_distance:
                next_candidate.append(histogram[i])
        
        next_candidate.sort(key=custom_sort)
        
        if next_candidate == []:
            None
        else:
            self.already_direction_index.append(next_candidate[0][0])
            if next_candidate[0][-1] <= params.DFS_THRESHOLD:
                return split_arg * (next_candidate[0][0] + 0.5)


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
    
    
    def calculate_mu_azimuth2(self, marker_list):
        return super().calculate_mu_azimuth2(marker_list)
    
    
    def calculate_mu_azimuth3(self, init_x, init_y):
        return super().calculate_mu_azimuth3(init_x, init_y)
    
    
    def calculate_collision_df(self, x, y):
        return super().calculate_collision_df(x, y)
    
    
    def vfh(self, bins=params.VFH_BINS):
        return super().vfh(bins)
    
    
    def vfh_using_probability(self, bins=params.VFH_BINS):
        return super().vfh_using_probability(bins)
    
    
    def vfh_only_obstacle_density(self, bins=params.VFH_BINS):
        return super().vfh_only_obstacle_density(bins)
    
    
    def vfh_using_probability_only_obstacle_density(self, bins=params.VFH_BINS):
        return super().vfh_using_probability_only_obstacle_density(bins)
    
    
    def vfh_dfs(self, init_x, init_y, bins=params.VFH_BINS):
        return super().vfh_dfs(init_x, init_y, bins)


# メインマーカー変換閾値
def marker_change_threshold():
    area_count = 0
    for i in range(round(-params.OUTER_BOUNDARY), round(params.OUTER_BOUNDARY + 1)):
        for j in range(round(-params.OUTER_BOUNDARY), round(params.OUTER_BOUNDARY + 1)):
            distance = math.sqrt(i ** 2 + j ** 2)
            if params.INNER_BOUNDARY < distance < params.OUTER_BOUNDARY:
                area_count += 1
    return 1.5 / area_count * 100.0


# vfh_dfs用のソートキーの比較関数
def custom_sort(item):
    return(item[2], -item[1])