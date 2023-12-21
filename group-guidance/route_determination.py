def route_determinate(group_num, marker_list):
    
    # 探査用ルートの作成
    route_key = 0
    wide_area_list = []
    for i in range(group_num):
        for j in range(len(marker_list[i])):
            if route_key == 0:
                if marker_list[i][j].label == 3:
                    wide_area_list.append([marker_list[i][j]])
                    route_key = 1
            else:
                if marker_list[i][j].label != 3:
                    route_key = 0
                else:
                    wide_area_list[-1].append(marker_list[i][j])
    
    length_list = []
    for i in range(len(wide_area_list)):
        length_list.append(len(wide_area_list[i]))
    
    red_num_result = normalize_list(length_list, target_sum=30, lower_bound=5)
    
    print("Original List:", length_list)
    print("Normalized List (Sum = 30, Lower Bound = 5):", red_num_result)
    
    wide_area_exploration_list = []
    for i in range(len(wide_area_list)):
        explore_point = wide_area_list[i][len(wide_area_list[i]) // 2]
        wide_area_exploration_list.append(explore_point)

    exploration_route_list = []
    for i in range(len(wide_area_exploration_list)):
        exploration_route_point = wide_area_exploration_list[i]
        exploration_route_list.append([])
        while True:
            exploration_route_list[i].append(exploration_route_point)
            if exploration_route_point.parent != None:
                exploration_route_point = exploration_route_point.parent
            else:
                break
    
    exploration_route_list.reverse()
    for i in range(len(exploration_route_list)):
        exploration_route_list[i].reverse()
    
    return exploration_route_list, red_num_result


def normalize_list(lst, target_sum = 30, lower_bound = 5):
    current_sum = sum(lst)
    
    if current_sum == 0:
        # リストが全て0の場合は正規化の意味がないので何もせずに返す
        return lst
    
    scaling_factor = target_sum / current_sum

    normalized_list = [max(int(element * scaling_factor), lower_bound) for element in lst]
    
    # 正規化後の合計が30にならない場合, 調整が必要
    adjustment = target_sum - sum(normalized_list)
    if adjustment != 0:
        # 正規化後の合計が30になるように, 調整が必要な要素に調整を行う
        for i in range(len(lst)):
            if normalized_list[i] + adjustment >= lower_bound:
                normalized_list[i] += adjustment
                break

    return normalized_list