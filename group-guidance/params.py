import math

# -------------------------------- const parameter ---------------------------------------------------
MAX_MOVEMENT = 3.0                          # 最大移動量
MIN_MOVEMENT = 2.0                          # 最小移動量
MAX_BOIDS_MOVEMENT = 3.0                    # boids判定時の最大移動量
MIN_BOIDS_MOVEMENT = 2.0                    # boids判定時の最小移動量
OUTER_BOUNDARY = 10.0                       # マーカ-の外側境界
INNER_BOUNDARY = 0.0                        # マーカーの内側境界
MEAN = 0.0                                  # マーカーの分布平均
VARIANCE = 10.0                              # マーカーの分布標準偏差
MAP_HEIGHT = 60                             # マップの縦サイズ
MAP_WIDTH = 200                              # マップの横サイズ
CENTER_X = math.floor(MAP_WIDTH / 2)
CENTER_Y = math.floor(MAP_HEIGHT / 2)
AREA_COVERAGE_THRESHOLD = 80.0              # 領域網羅率のマーカー変換値
AREA_STEP_THRESHOLD = 140                   # 領域探査のステップ上限
ALL_AREA_COVERAGE_THRESHOLD = 70.0          # マップ全体の網羅率の閾値
AREA_COVERAGE_STEP = 10
SAVE_DIRECTORY = 'csv/'                     # csvファイルの格納先フォルダ
# ------------ VFHに使用するパラメータ -----------------------
VFH_DRIVABILITY_BIAS = 0.5
VFH_EXPLORATION_BIAS = 0.5
VFH_EXPLORATION_STD = 120
VFH_MIN_VALUE = 0.01
VFH_VONMISES_KAPPA = 2
VFH_BINS = 16