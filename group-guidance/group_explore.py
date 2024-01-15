import math
import pandas as pd
import matplotlib.pyplot as plt
import sys
import route_determination
import func_group_dfs
import marker
import red_cls
import obstacle
import params
import map


def main():
    # 障害物マップの取得
    obstacle_list, obstacle_df_list, obstacle_size = obstacle.obstacle_map_ex2()
    
    # 探査用マーカーリストの作成
    marker_change_th = marker.marker_change_threshold()
    explore_main_marker_list     = []
    explore_marker_list          = []
    explore_marker_change_key    = []
    explore_collision_counter    = []
    explore_area_step            = []
    explore_marker_algorithm_key = []
    explore_marker_counter = []
    explore_back_key = []

    # 探査用redリストの作成
    explore_red_list = []
    explore_red_df_list = []

    group_num, marker_list = func_group_dfs.func_group_dfs()
    route_list, red_num_list = route_determination.route_determinate(group_num, marker_list)
    route_key = []
    route_count = []
    
    for i in range(len(route_list)):
        red_num = red_num_list[i]
        explore_red_list.append([])
        explore_red_df_list.append([])
        explore_marker_list.append([])
        explore_marker_change_key.append(0)
        explore_collision_counter.append(0)
        explore_area_step.append(0)
        explore_main_marker_list.append(route_list[i][0])
        explore_marker_algorithm_key.append(0)
        explore_marker_counter.append(1)
        explore_back_key.append(0)
        route_key.append(0)
        route_count.append(1)
        
        for j in range(red_num):
            red_id = 'ex_red' + str(i + 1) + '-' + str(j + 1)
            red  = red_cls.Red(explore_main_marker_list[i].x, explore_main_marker_list[i].y, red_id, explore_main_marker_list[i])
            explore_red_list[i].append(red)
            red_df = pd.DataFrame()
            red_df = pd.DataFrame(explore_red_list[i][j].get_arguments())
            explore_red_df_list[i].append(red_df)
    
    # グリッドマップの作成
    grid_map = []
    X_list   = []
    Y_list   = []
    for i in range(len(route_list)):
        get_map = map.grid_map()
        grid_map.append(get_map[0])
        X_list.append(get_map[1])
        Y_list.append(get_map[2])
    get_map = map.overall_map()
    grid_map.append(get_map[0])
    X_list.append(get_map[1])
    Y_list.append(get_map[2])
    
    fig, ax = plt.subplots(1, len(route_list) + 1)
    
    
    # 探査開始
    while True:
        # 全体網羅率の計算
        map_coverage_ratio = map.map_coverage_calculation(grid_map[-1], obstacle_size)
        print('#' * 30)
        print(explore_red_list[-1][-1].step, 'step : map_coverage_ratio :', map_coverage_ratio)
        print('#' * 30)
        
        # 誘導終了判定
        #if explore_red_list[-1][-1].step == 5000:
        if map_coverage_ratio >= params.ALL_AREA_COVERAGE_THRESHOLD:
            print('Exploration completed.')
            break
        
        # 群ごとの動作
        for i in range(len(route_list)):
            area_coverage_ratio = map.area_coverage_calculation(grid_map[i], explore_main_marker_list[i])
            print('=' * 30)
            print(explore_area_step[i], 'step(total ', explore_red_list[i][-1].step, ' step) : area_coverage_ratio : ', area_coverage_ratio)
            print('=' * 30)
            if route_key[i] == 0:
                if explore_area_step[i] == 10:
                    explore_marker_list[i].append(explore_main_marker_list[i])
                    explore_marker_counter[i] += 1
                    explore_main_marker_list[i] = route_list[i][route_count[i]]
                    print('=' * 10, 'marker', str(i + 1), ' change', '=' * 10)
                    print(str(i + 1) + 'group marker back changed. (', explore_main_marker_list[i].x, ', ', explore_main_marker_list[i].y, ') : ', str(explore_marker_counter[i]))
                    route_count[i] += 1
                    explore_area_step[i] = 0
                    if explore_main_marker_list[i] == route_list[i][-1]:
                        route_key[i] = 1
            else:
                # 網羅率向上によるストレスカウント
                if area_coverage_ratio - explore_main_marker_list[i].coverage_ratio <= marker_change_th:
                    explore_marker_change_key[i] += 1
                else:
                    explore_marker_change_key[i] = 0
                
                explore_main_marker_list[i].coverage_ratio = area_coverage_ratio

                # 網羅率が閾値を超えた場合の探査中心の変換
                if explore_main_marker_list[i].coverage_ratio >= params.AREA_COVERAGE_THRESHOLD:
                    if len(explore_main_marker_list[i].already_direction_index) == params.VFH_BINS:
                        explore_marker_list[i].append(explore_main_marker_list[i])
                        explore_main_marker_list[i] = explore_main_marker_list[i].parent
                        print('=' * 10, 'marker ', str(i + 1), ' change', '='* 10)
                        print(str(i + 1) + 'group already_direction_index is max. group marker back changed. (', explore_main_marker_list[i].x, ', ', explore_main_marker_list[i].y, ') : ', str(explore_marker_counter[i] - 1))
                        print('=' * 30)
                    else:
                        next_theta = math.radians(explore_main_marker_list[i].vfh_using_probability())
                        next_x = params.OUTER_BOUNDARY * math.cos(next_theta) + explore_main_marker_list[i].x
                        next_y = params.OUTER_BOUNDARY * math.sin(next_theta) + explore_main_marker_list[i].y
                        explore_marker_list[i].append(explore_main_marker_list[i])
                        explore_marker_counter[i] += 1
                        explore_main_marker_list[i] = marker.Virtual_marker('ex_marker' + str(i + 1) + '-' + str(explore_marker_counter[i]), next_x, next_y, explore_main_marker_list[i])
                        print('=' * 10, 'marker', str(i + 1), ' change', '=' * 10)
                        print(str(i + 1) + 'group marker changed by vfh_using_probability(). (', explore_main_marker_list[i].x, ', ', explore_main_marker_list[i].y, ') : ', len(explore_marker_list[i]))
                        print('=' * 30)
                    explore_marker_algorithm_key[i] = 1
                    explore_marker_change_key[i] = 0
                    
                    # グリッドマップの更新
                    grid_map[i], _, _ = map.grid_map()
                    ax[i].pcolormesh(X_list[i], Y_list[i], grid_map[i], cmap = 'brg', edgecolors = 'black', shading='auto')
                    ax[i].set_title('marker' + str(i + 1) + 'grid_map')
                    plt.pause(0.05)
                
                # 網羅率が閾値を超えなかった場合とストレスカウントが閾値を超えた場合の探査中心の変換
                elif explore_marker_change_key[i] >= 4: #or explore_area_step[i] == params.AREA_STEP_THRESHOLD:
                    # 前回の探査中心の変換が確率密度を用いたものだった場合
                    #if explore_marker_algorithm_key[i] == 1:
                    explore_marker_list[i].append(explore_main_marker_list[i])
                    explore_main_marker_list[i] = explore_main_marker_list[i].parent
                    print('=' * 10, 'marker', str(i + 1), ' change', '=' * 10)
                    print(str(i + 1) + 'group marker back changed. (', explore_main_marker_list[i].x, ', ', explore_main_marker_list[i].y, ') : ', str(explore_marker_counter[i] - 1))
                    print('=' * 30)
                    
                    # 前回の探査中心の変換が確率密度を用いなかった場合
                    #elif explore_marker_algorithm_key[i] == 0:
                    #    next_theta = math.radians(explore_main_marker_list[i].vfh_only_obstacle_density())
                    #    next_x = params.OUTER_BOUNDARY * math.cos(next_theta) + explore_main_marker_list[i].x
                    #    next_y = params.OUTER_BOUNDARY * math.sin(next_theta) + explore_main_marker_list[i].y
                    #    explore_main_marker_list[i] = marker.Virtual_marker('marker' + str(i + 1) + '-' + str(len(explore_marker_list[i]) + 1), next_x, next_y, explore_main_marker_list[i])
                    #    print('=' * 10, 'marker', str(i + 1), ' change', '=' * 10)
                    #    print(str(i + 1) + 'group marker changed by vfh(). (', explore_main_marker_list[i].x, ', ', explore_main_marker_list[i].y, ') : ', len(explore_marker_list[i]))
                    #    print('=' * 30)
                    explore_marker_change_key[i] = 0
                    explore_marker_algorithm_key[i] = 0
                    explore_area_step[i] = 0
                    
                    # グリッドマップの更新
                    grid_map[i], _, _ = map.grid_map()
                    ax[i].pcolormesh(X_list[i], Y_list[i], grid_map[i], cmap = 'brg', edgecolors = 'black', shading='auto')
                    ax[i].set_title('marker' + str(i + 1) + ' grid_map')
                    plt.pause(0.05)
                
            for j in range(params.AREA_COVERAGE_STEP):
                for k in range(len(explore_red_list[i])):
                    explore_red_list[i][k].step_motion(obstacle_list, explore_main_marker_list[i], grid_map[-1], grid_map[i])
                    explore_red_df_list[i][k] = pd.concat([explore_red_df_list[i][k], explore_red_list[i][k].get_arguments()])
                    
                    if explore_red_list[i][k].collision_flag == 1:
                        explore_collision_counter[i] += 1
                        explore_main_marker_list[i].calculate_collision_df(explore_red_list[i][k].x, explore_red_list[i][k].y)
                
                explore_area_step[i] += 1
            
            # 群ごとのグリッドマップ
            ax[i].pcolormesh(X_list[i], Y_list[i], grid_map[i], cmap = 'brg', edgecolors = 'black', shading='auto')
            ax[i].set_title('marker' + str(i + 1) + ' grid_map')
            # 全体のグリッドマップ
            ax[-1].pcolormesh(X_list[-1], Y_list[-1], grid_map[-1], cmap = 'brg', edgecolors = 'black', shading='auto')
            ax[-1].set_title('main grid_map')
            plt.pause(0.05)
    
    # データフレームの作成
    # Red
    for i in range(len(explore_red_df_list)):
        for j in range(len(explore_red_df_list[i])):
            explore_red_df_list[i][j].to_csv(params.SAVE_DIRECTORY + explore_red_list[i][j].red_id + '.csv')
    
    # Marker
    for i in range(len(explore_marker_list)):
        for j in range(len(explore_marker_list[i])):
            if j == 0:
                marker_df = pd.DataFrame(explore_marker_list[i][j].get_arguments())
            else:
                marker_df = pd.concat([marker_df, explore_marker_list[i][j].get_arguments()])
            marker_df.to_csv(params.SAVE_DIRECTORY + 'ex_marker' + str(i + 1) + '.csv')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopped by keyboard input(ctrl-c)")
        sys.exit()