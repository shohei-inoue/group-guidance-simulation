import math
import pandas as pd
import matplotlib.pyplot as plt
import sys
import map
import marker
import red_cls
import params
import obstacle


# ここの部分だけをシミュレーションしたければmain()に変更
def func_group_dfs():
    # 障害物マップの取得
    obstacle_list, obstacle_df_list, obstacle_size = obstacle.obstacle_map_ex2()
    
    # マーカー用リストの作成
    main_marker_list     = []
    marker_list          = []
    collision_counter    = []
    area_step            = []
    marker_counter = []
    back_key = []

    # Red用リストの作成
    red_list    = []
    red_df_list = []
    
    # 群数の指定と群の作成
    group_num = int(input('How many groups do you make? : '))
    for i in range(group_num):
        red_num = int(input('Please enter the number of Reds in the ' + str(i + 1) + ' group. : '))
        marker_x = int(input('Please enter the x-coordinate of the ' + str(i + 1) + ' group marker : '))
        marker_y = int(input('Please enter the y-coordinate of the ' + str(i + 1) + ' group marker : '))
        main_marker = marker.Real_marker('marker' + str(i + 1) + '-1', marker_x, marker_y)
        red_list.append([])
        red_df_list.append([])
        marker_list.append([])
        collision_counter.append(0)
        area_step.append(0)
        main_marker_list.append(main_marker)
        marker_counter.append(1)
        back_key.append(0)
        
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
        if red_list[-1][-1].step == 4000:
            print('Exploration completed.')
            break
        
        # 群ごとの動作
        for i in range(group_num):
            # 探査領域網羅率の計算
            area_coverage_ratio = map.area_coverage_calculation(grid_map[i], main_marker_list[i])
            print('=' * 30)
            print(area_step[i], 'step(total ', red_list[i][-1].step, ' step) : area_coverage_ratio : ', area_coverage_ratio)
            print('=' * 30)
            main_marker_list[i].coverage_ratio = area_coverage_ratio
            
            # 探査中心の変換
            if area_step[i] == 100:
                next_theta = main_marker_list[i].vfh_dfs(-90, 0) # 仮置き
                # 戻り行動
                if next_theta == None or main_marker_list[i].coverage_ratio < 50:
                    main_marker_list[i].label = 1
                    if back_key[i] != 1:
                        marker_list[i].append(main_marker_list[i])
                    main_marker_list[i] = main_marker_list[i].parent
                    back_key[i] = 1
                    print('=' * 10, 'marker', str(i + 1), 'change', '=' * 10)
                    print(str(i + 1) + 'group marker back changed. (', main_marker_list[i].x, ', ', main_marker_list[i].y, ') : ', len(marker_list[i]))
                    print('=' * 30)
                else:
                    next_theta = math.radians(next_theta)
                    next_x = params.OUTER_BOUNDARY * math.cos(next_theta) + main_marker_list[i].x
                    next_y = params.OUTER_BOUNDARY * math.sin(next_theta) + main_marker_list[i].y
                    if back_key[i] != 1:
                        marker_list[i].append(main_marker_list[i])
                    back_key[i] = 0
                    marker_counter[i] += 1
                    main_marker_list[i] = marker.Virtual_marker('marker' + str(i + 1) + '-' + str(marker_counter[i]), next_x, next_y, main_marker_list[i])
                    print('=' * 10, 'marker', str(i + 1), ' change', '=' * 10)
                    print(str(i + 1) + 'group marker changed by vfh_dfs(). (', main_marker_list[i].x, ', ', main_marker_list[i].y, ') : ', len(marker_list[i]))
                    print('=' * 30)
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
            red_df_list[i][j].to_csv(params.SAVE_DIRECTORY + red_list[i][j].red_id + '.csv')

    # Marker
    for i in range(len(marker_list)):
        for j in range(len(marker_list[i])):
            if j == 0:
                marker_df = pd.DataFrame(marker_list[i][j].get_arguments())
            else:
                marker_df = pd.concat([marker_df, marker_list[i][j].get_arguments()])
            marker_df.to_csv(params.SAVE_DIRECTORY + 'marker' + str(i + 1) + '.csv')
    
    # obstacle
    for i in range(len(obstacle_df_list)):
        obstacle_df_list[i].to_csv(params.SAVE_DIRECTORY + 'obstacle' + str(i + 1) + '.csv')
    
    return group_num, marker_list