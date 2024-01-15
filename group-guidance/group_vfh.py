import math
import pandas as pd
import matplotlib.pyplot as plt
import sys
import map
import marker
import red_cls
import params
import obstacle

def main():
    # 障害物マップの取得
    obstacle_list, obstacle_df_list, obstacle_size = obstacle.obstacle_map_exs()
    
    # マーカー用リストの作成
    marker_change_th = marker.marker_change_threshold()
    main_marker_list     = []
    marker_list          = []
    marker_change_key    = []
    collision_counter    = []
    area_step            = []
    marker_algorithm_key = []
    
    # Red用リストの作成
    red_list    = []
    red_df_list = []
    
    # 群数の指定と群の作成
    #group_num = int(input('How many groups do you make? : '))
    group_num = 1
    for i in range(group_num):
        #red_num = int(input('Please enter the number of Reds in the ' + str(i + 1) + ' group. : '))
        red_num = 10
        #marker_x = int(input('Please enter the x-coordinate of the ' + str(i + 1) + ' group marker : '))
        #marker_y = int(input('Please enter the y-coordinate of the ' + str(i + 1) + ' group marker : '))
        marker_x = 0
        marker_y = 0
        main_marker = marker.Real_marker('marker' + str(i + 1) + '-1', marker_x, marker_y)
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
            red = red_cls.Red(main_marker.x, main_marker.y, red_id, main_marker)
            red_list[i].append(red)
            red_df = pd.DataFrame()
            red_df = pd.DataFrame(red_list[i][j].get_arguments())
            red_df_list[i].append(red_df)
    
    # グリッドマップの作成
    grid_map = []
    X_list   = []
    Y_list   = []
    for i in range(group_num):
        get_map = map.grid_map()
        grid_map.append(get_map[0])
        X_list.append(get_map[1])
        Y_list.append(get_map[2])
    get_map = map.overall_map()
    grid_map.append(get_map[0])
    X_list.append(get_map[1])
    Y_list.append(get_map[2])
    
    fig, ax = plt.subplots(1, group_num + 1)
    
    # 誘導開始
    while True:
        # 全体網羅率の計算
        map_coverage_ratio = map.map_coverage_calculation(grid_map[-1], obstacle_size)
        print('#' * 30)
        print(red_list[-1][-1].step, 'step : map_coverage_ratio :', map_coverage_ratio)
        print('#' * 30)
        
        # 誘導終了判定
        #if map_coverage_ratio >= params.ALL_AREA_COVERAGE_THRESHOLD or red_list[-1][-1].step == 10000: # 仮置き
        if red_list[-1][-1].step == 5000:
            print('Exploration completed.')
            break
        
        # 群ごとの動作
        for i in range(group_num):
            # 探査領域網羅率の計算
            area_coverage_ratio = map.area_coverage_calculation(grid_map[i], main_marker_list[i])
            print('=' * 30)
            print(area_step[i], 'step(total ', red_list[i][-1].step, ' step) : area_coverage_ratio : ', area_coverage_ratio)
            print('=' * 30)
            # 網羅率向上によるストレスカウント
            if area_coverage_ratio - main_marker_list[i].coverage_ratio <= marker_change_th:
                marker_change_key[i] += 1
            else:
                marker_change_key[i] = 0
            
            main_marker_list[i].coverage_ratio = area_coverage_ratio
            
            # 網羅率が閾値を超えた場合の探査中心の変換
            if main_marker_list[i].coverage_ratio >= params.AREA_COVERAGE_THRESHOLD:
                if len(main_marker_list[i].already_direction_index) == params.VFH_BINS:
                    marker_list[i].append(main_marker_list[i])
                    main_marker_list[i] = main_marker_list[i].parent
                    print('=' * 10, 'marker ', str(i + 1), ' change', '='* 10)
                    print(str(i + 1) + 'already_direction_index is max. group marker back changed. (', main_marker_list[i].x, ', ', main_marker_list[i].y, ') : ', len(marker_list[i]))
                    print('=' * 30)
                else:
                    next_theta = math.radians(main_marker_list[i].vfh_using_probability())
                    #next_theta = math.radians(main_marker_list[i].vfh_using_probability_only_obstacle_density())
                    next_x = params.OUTER_BOUNDARY * math.cos(next_theta) + main_marker_list[i].x
                    next_y = params.OUTER_BOUNDARY * math.sin(next_theta) + main_marker_list[i].y
                    marker_list[i].append(main_marker_list[i])
                    main_marker_list[i] = marker.Virtual_marker('marker' + str(i + 1) + '-' + str(len(marker_list[i]) + 1), next_x, next_y, main_marker_list[i])
                    print('=' * 10, 'marker', str(i + 1), ' change', '=' * 10)
                    print(str(i + 1) + 'group marker changed by vfh_using_probability(). (', main_marker_list[i].x, ', ', main_marker_list[i].y, ') : ', len(marker_list[i]))
                    print('=' * 30)
                marker_algorithm_key[i] = 1
                marker_change_key[i] = 0
                
                # グリットマップの更新
                grid_map[i], _, _ = map.grid_map()
                ax[i].pcolormesh(X_list[i], Y_list[i], grid_map[i], cmap = 'brg', edgecolors = 'black', shading='auto')
                ax[i].set_title('marker' + str(i + 1) + 'grid_map')
                plt.pause(0.05)
            
            # 網羅率が閾値を超えなかった場合とストレスカウントが閾値を超えた場合の探査中心の変換
            elif marker_change_key[i] >= 4:# or area_step[i] == params.AREA_STEP_THRESHOLD:
                # 前回の探査中心の変換が確率密度を用いたものだった場合
                #if marker_algorithm_key[i] == 1:
                marker_list[i].append(main_marker_list[i])
                main_marker_list[i] = main_marker_list[i].parent
                print('=' * 10, 'marker', str(i + 1), ' change', '=' * 10)
                print(str(i + 1) + 'group marker back changed. (', main_marker_list[i].x, ', ',main_marker_list[i].y, ') : ', len(marker_list[i]))
                print('=' * 30)
                
                # 前回の探査中心の変換が確率密度を用いなかった場合
                #elif marker_algorithm_key[i] == 0:
                #    next_theta = math.radians(main_marker_list[i].vfh_only_obstacle_density())
                #    next_x = params.OUTER_BOUNDARY * math.cos(next_theta) + main_marker_list[i].x
                #    next_y = params.OUTER_BOUNDARY * math.sin(next_theta) + main_marker_list[i].y
                #    main_marker_list[i] = marker.Virtual_marker('marker' + str(i + 1) + '-' + str(len(marker_list[i]) + 1), next_x, next_y, main_marker_list[i])
                #    print('=' * 10, 'marker', str(i + 1), ' change', '=' * 10)
                #    print(str(i + 1) + 'group marker changed by vfh(). (', main_marker_list[i].x, ', ', main_marker_list[i].y, ') : ', len(marker_list[i]))
                #    print('=' * 30)
                marker_change_key[i] = 0
                marker_algorithm_key[i] = 0
                area_step[i] = 0
                
                # グリッドマップの更新
                grid_map[i], _, _ = map.grid_map()
                ax[i].pcolormesh(X_list[i], Y_list[i], grid_map[i], cmap = 'brg', edgecolors = 'black', shading='auto')
                ax[i].set_title('marker' + str(i + 1) + ' grid_map')
                plt.pause(0.05)
            
            for j in range(params.AREA_COVERAGE_STEP):
                for k in range(len(red_list[i])):
                    red_list[i][k].step_motion(obstacle_list, main_marker_list[i], grid_map[-1], grid_map[i])
                    red_df_list[i][k] = pd.concat([red_df_list[i][k], red_list[i][k].get_arguments()])
                    
                    if red_list[i][k].collision_flag == 1:
                        collision_counter[i] += 1
                        main_marker_list[i].calculate_collision_df(red_list[i][k].x, red_list[i][k].y)
                
                area_step[i] += 1
        
        # 群ごとのグリッドマップ
        ax[i].pcolormesh(X_list[i], Y_list[i], grid_map[i], cmap = 'brg', edgecolors = 'black', shading='auto')
        ax[i].set_title('marker' + str(i + 1) + ' grid_map')
        # 全体のグリッドマップ
        ax[-1].pcolormesh(X_list[-1], Y_list[-1], grid_map[-1], cmap = 'brg', edgecolors = 'black', shading='auto')
        ax[-1].set_title('main grid_map')
        plt.pause(0.05)
    
    # データフレームの作成
    # Red
    for i in range(len(red_df_list)):
        for j in range(len(red_df_list[i])):
            red_df_list[i][j].to_csv(params.SAVE_DIRECTORY + 'ex_' + red_list[i][j].red_id + '.csv')

    # Marker
    for i in range(len(marker_list)):
        for j in range(len(marker_list[i])):
            if j == 0:
                marker_df = pd.DataFrame(marker_list[i][j].get_arguments())
            else:
                marker_df = pd.concat([marker_df, marker_list[i][j].get_arguments()])
            marker_df.to_csv(params.SAVE_DIRECTORY + 'ex_marker' + str(i + 1) + '.csv')
    
    # obstacle
    for i in range(len(obstacle_df_list)):
        obstacle_df_list[i].to_csv(params.SAVE_DIRECTORY + 'obstacle' + str(i + 1) + '.csv')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopped by keyboard input(ctrl-c)")
        sys.exit()