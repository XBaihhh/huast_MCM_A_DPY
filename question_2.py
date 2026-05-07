import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

# ===================== 全局参数定义 =====================
# 1. 车辆参数
VEHICLE_PARAMS = {
    # 燃油车
    'F1': {'type': 'fuel', 'load': 3000, 'volume': 13.5, 'count': 60, 'start_cost': 400},
    'F2': {'type': 'fuel', 'load': 1500, 'volume': 10.8, 'count': 50, 'start_cost': 400},
    'F3': {'type': 'fuel', 'load': 1250, 'volume': 6.5, 'count': 50, 'start_cost': 400},
    # 新能源车
    'E1': {'type': 'electric', 'load': 3000, 'volume': 15, 'count': 10, 'start_cost': 400},
    'E2': {'type': 'electric', 'load': 1250, 'volume': 8.5, 'count': 15, 'start_cost': 400},
}

# 2. 成本参数
COST_PARAMS = {
    'fuel_price': 7.61,  # 元/L
    'electric_price': 1.64,  # 元/kWh
    'carbon_price': 0.65,  # 元/kg
    'early_wait_cost': 20,  # 元/小时
    'late_punish_cost': 50,  # 元/小时
    'restrict_wait_cost': 20,  # 限行等待成本 元/小时
    'violate_punish': 1e8,  # 违规惩罚成本
}

# 3. 速度与时段参数
SPEED_PARAMS = {
    'smooth': {'time': [(9, 10), (13, 15)], 'mu': 55.3, 'sigma2': 0.12},
    'normal': {'time': [(10, 11.5), (15, 17)], 'mu': 35.4, 'sigma2': 5.22},
    'congest': {'time': [(8, 9), (11.5, 13)], 'mu': 9.8, 'sigma2': 4.72},
}
# 静态调度取均值速度
SPEED_MEAN = {
    'smooth': 55.3,
    'normal': 35.4,
    'congest': 9.8,
}

# 4. 能耗公式
def calc_FPK(v):
    # 燃油车百公里油耗 L/100km
    return 0.0025 * (v ** 2) - 0.2554 * v + 31.75

def calc_EPK(v):
    # 新能源车百公里电耗 kWh/100km
    return 0.0014 * (v ** 2) - 0.12 * v + 36.19

# 能耗满载系数
FUEL_LOAD_FACTOR = 0.4    # 满载比空载高40%
ELECTRIC_LOAD_FACTOR = 0.35  # 满载比空载高35%

# 5. 碳排放系数
CARBON_FACTOR = {
    'fuel': 2.547,     # kg/L
    'electric': 0.501, # kg/kWh
}

# 6. 服务与限行参数
SERVICE_TIME = 20 / 60  # 服务时间20分钟，转小时
RESTRICT_TIME = (8, 16) # 限行时段8:00-16:00
GREEN_ZONE_RADIUS = 10  # 绿区半径10km
DEPOT_ID = 0             # 配送中心ID

# 7. ALNS算法参数
ALNS_PARAMS = {
    'max_iter': 1000,     
    'init_temp': 1000,
    'cooling_rate': 0.995,
    'rho_A': 0.15,          # NA类客户移除概率衰减因子
    'weight_update': 0.1,   # 算子权重更新系数
    'segment_size': 50,     # 权重更新段大小
}

# ===================== 数据读取与预处理 =====================
def load_and_preprocess_data():
    """
    读取4个数据文件，按客户ID聚合订单，完成客户分类
    """
    # 读取数据文件
    coord_df = pd.read_excel('./data/客户坐标信息.xlsx')
    order_df = pd.read_excel('./data/订单信息.xlsx')
    timewindow_df = pd.read_excel('./data/时间窗.xlsx')
    distance_df = pd.read_excel('./data/距离矩阵.xlsx', index_col=0)

    # 数据清洗
    order_df["重量"] = pd.to_numeric(order_df["重量"], errors="coerce").fillna(0.0)
    order_df["体积"] = pd.to_numeric(order_df["体积"], errors="coerce").fillna(0.0)
    distance_df = distance_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    
    # ===================== 按客户ID聚合订单 =====================
    print(f"原始订单行数：{len(order_df)}")
    order_agg = order_df.groupby('目标客户编号').agg(
        总重量=('重量', 'sum'),
        总体积=('体积', 'sum')
    ).reset_index()
    print(f"聚合后客户数：{len(order_agg)}")
    # ========================================================================
    
    # 合并客户数据
    depot_info = coord_df[coord_df['类型'] == '配送中心'].iloc[0]
    customers = coord_df[coord_df['类型'] == '客户'].copy()

    # 1. 合并聚合后的订单数据
    customers = customers.merge(
        order_agg, 
        left_on='ID',           
        right_on='目标客户编号', 
        how='left'
    )
    # 重命名列以便后续使用
    customers.rename(columns={'总重量': '重量', '总体积': '体积'}, inplace=True)
    
    # 2. 合并时间窗数据
    customers = customers.merge(
        timewindow_df, 
        left_on='ID',           
        right_on='客户编号',    
        how='left'
    )
    
    # 时间格式转换：HH:MM → 小时数值
    def time_to_hour(time_str):
        if pd.isna(time_str):
            return np.nan
        if isinstance(time_str, pd.Timestamp):
            return time_str.hour + time_str.minute / 60
        try:
            h, m = map(int, str(time_str).split(':'))
            return h + m / 60
        except:
            return np.nan
    
    customers['ETi'] = customers['开始时间'].apply(time_to_hour)
    customers['LTi'] = customers['结束时间'].apply(time_to_hour)
    
    # 计算绿区标记g_i
    customers['g_i'] = customers.apply(
        lambda x: 1 if (x['X (km)']**2 + x['Y (km)']**2) <= GREEN_ZONE_RADIUS**2 else 0,
        axis=1
    )
    
    # 客户四分类
    def classify_customer(row):
        if row['g_i'] == 0:
            return 'ND'
        LTi = row['LTi']
        ETi = row['ETi']
        if pd.isna(LTi) or pd.isna(ETi):
            return 'ND'
        if LTi < RESTRICT_TIME[1]:
            return 'NA'
        elif ETi < RESTRICT_TIME[1] < LTi:
            return 'NB'
        elif ETi >= RESTRICT_TIME[1]:
            return 'NC'
        return 'ND'
    
    customers['type'] = customers.apply(classify_customer, axis=1)
    distance_matrix = distance_df.to_numpy()
    
    # 输出统计信息
    print(f"数据预处理完成，共{len(customers)}个客户")
    print(f"客户分类：NA={sum(customers['type']=='NA')}, NB={sum(customers['type']=='NB')}, NC={sum(customers['type']=='NC')}, ND={sum(customers['type']=='ND')}")
    
    return customers, distance_matrix, depot_info

# ===================== 前置可行性校验 =====================
def feasibility_check(customers):
    na_customers = customers[customers['type'] == 'NA']
    total_na_weight = na_customers['重量'].sum()
    total_na_volume = na_customers['体积'].sum()
    
    total_electric_weight = 0
    total_electric_volume = 0
    for v_type, params in VEHICLE_PARAMS.items():
        if params['type'] == 'electric':
            total_electric_weight += params['load'] * params['count']
            total_electric_volume += params['volume'] * params['count']
    
    print(f"NA类客户总需求：重量={total_na_weight:.2f}kg，体积={total_na_volume:.4f}m³")
    print(f"新能源车总运力：重量={total_electric_weight:.2f}kg，体积={total_electric_volume:.4f}m³")
    
    if total_na_weight > total_electric_weight or total_na_volume > total_electric_volume:
        print("【警告】NA类客户总需求超过新能源车总运力，无可行解！")
        return False
    print("【通过】可行性校验")
    return True

# ===================== 辅助函数：行驶时间计算 =====================
def calc_drive_time(distance, start_time):
    remaining_distance = distance
    current_time = start_time
    total_drive_time = 0
    
    while remaining_distance > 1e-6:
        current_hour = current_time % 24
        speed = SPEED_MEAN['smooth']
        period_end = 24
        
        for period, params in SPEED_PARAMS.items():
            for (t_start, t_end) in params['time']:
                if t_start <= current_hour < t_end:
                    speed = SPEED_MEAN[period]
                    period_end = t_end
                    break
            if speed != SPEED_MEAN['smooth']: break
        
        time_in_period = period_end - current_hour
        distance_in_period = speed * time_in_period
        
        if distance_in_period >= remaining_distance:
            drive_time = remaining_distance / speed
            total_drive_time += drive_time
            current_time += drive_time
            remaining_distance = 0
        else:
            total_drive_time += time_in_period
            current_time += time_in_period
            remaining_distance -= distance_in_period
    
    return total_drive_time, current_time

# ===================== 核心函数：单路径成本计算 =====================
def calc_route_cost(vehicle_type, route, customers, distance_matrix):
    cost_detail = {
        'start_cost': 0, 'energy_cost': 0, 'carbon_cost': 0,
        'early_wait_cost': 0, 'late_punish_cost': 0,
        'restrict_wait_cost': 0, 'violate_punish': 0
    }
    is_violate = False
    vehicle_params = VEHICLE_PARAMS[vehicle_type]
    is_fuel = (vehicle_params['type'] == 'fuel')
    
    cost_detail['start_cost'] = vehicle_params['start_cost']
    if len(route) <= 2:
        return cost_detail['start_cost'], cost_detail, [], is_violate
    
    # 路径总需求
    total_weight = 0
    total_volume = 0
    for c_id in route[1:-1]:
        c_match = customers[customers['ID'] == c_id]
        if c_match.empty: continue
        c = c_match.iloc[0]
        w = c['重量'] if pd.notna(c['重量']) else 0
        v = c['体积'] if pd.notna(c['体积']) else 0
        total_weight += w
        total_volume += v
        
    if total_weight > vehicle_params['load'] or total_volume > vehicle_params['volume']:
        cost_detail['violate_punish'] += 1e6
        is_violate = True
    
    # 逐节点模拟
    current_load = total_weight
    current_time = 8.0
    
    for i in range(1, len(route)):
        prev_node = route[i-1]
        curr_node = route[i]
        if curr_node == 0: break
        
        # 1. 行驶
        dist_val = distance_matrix[prev_node][curr_node]
        distance = dist_val if (pd.notna(dist_val) and dist_val > 0) else 0.1
        _, arrive_time = calc_drive_time(distance, current_time)
        current_time = arrive_time
        
        # 2. 客户信息
        c_match = customers[customers['ID'] == curr_node]
        if c_match.empty: continue
        c = c_match.iloc[0]
        c_type = c['type'] if pd.notna(c['type']) else 'ND'
        ETi = c['ETi'] if pd.notna(c['ETi']) else 8.0
        LTi = c['LTi'] if pd.notna(c['LTi']) else 20.0
        g_i = c['g_i'] if pd.notna(c['g_i']) else 0
        
        if c_type == 'NA' and is_fuel:
            cost_detail['violate_punish'] += COST_PARAMS['violate_punish']
            is_violate = True
        
        # 3. 限行等待
        restrict_wait = 0
        if is_fuel and g_i == 1:
            if current_time < RESTRICT_TIME[1]:
                restrict_wait = RESTRICT_TIME[1] - current_time
                current_time = RESTRICT_TIME[1]
                cost_detail['restrict_wait_cost'] += restrict_wait * COST_PARAMS['restrict_wait_cost']
        
        # 4. 时间窗
        early_wait = max(ETi - current_time, 0)
        late_punish = max(current_time - LTi, 0)
        current_time += early_wait
        cost_detail['early_wait_cost'] += early_wait * COST_PARAMS['early_wait_cost']
        cost_detail['late_punish_cost'] += late_punish * COST_PARAMS['late_punish_cost']
        
        # 5. 服务
        current_time += SERVICE_TIME
        
        # 6. 能耗改用实际行驶时段的速度，不再固定用smooth
        load_ratio = current_load / vehicle_params['load'] if vehicle_params['load'] > 0 else 0
        
        # 根据当前时间判断实际行驶时段，取对应速度
        current_hour = current_time % 24
        if 8 <= current_hour < 9 or 11.5 <= current_hour < 13:
            actual_speed = SPEED_MEAN['congest']
        elif 9 <= current_hour < 10 or 13 <= current_hour < 15:
            actual_speed = SPEED_MEAN['smooth']
        else:
            actual_speed = SPEED_MEAN['normal']
        
        if is_fuel:
            fpk = calc_FPK(actual_speed) if actual_speed > 0 else 31.75
            actual_fpk = fpk * (1 + FUEL_LOAD_FACTOR * load_ratio)
            fuel_consumption = actual_fpk * distance / 100
            cost_detail['energy_cost'] += fuel_consumption * COST_PARAMS['fuel_price']
            cost_detail['carbon_cost'] += fuel_consumption * CARBON_FACTOR['fuel'] * COST_PARAMS['carbon_price']
        else:
            epk = calc_EPK(actual_speed) if actual_speed > 0 else 36.19
            actual_epk = epk * (1 + ELECTRIC_LOAD_FACTOR * load_ratio)
            electric_consumption = actual_epk * distance / 100
            cost_detail['energy_cost'] += electric_consumption * COST_PARAMS['electric_price']
            cost_detail['carbon_cost'] += electric_consumption * CARBON_FACTOR['electric'] * COST_PARAMS['carbon_price']
        
        # 7. 卸货
        w = c['重量'] if pd.notna(c['重量']) else 0
        current_load -= w
    
    for k in cost_detail:
        if pd.isna(cost_detail[k]): cost_detail[k] = 0
            
    total_cost = sum(cost_detail.values())
    return total_cost, cost_detail, [], is_violate

# ===================== 初始解生成：快速贪心算法 =====================
def greedy_insertion_initial(vehicle_type, customer_list, customers, distance_matrix, depot_id=0):
    if not customer_list: return [depot_id, depot_id], []
    vehicle_params = VEHICLE_PARAMS[vehicle_type]
    
    route = [depot_id, depot_id]
    assigned = []
    current_load, current_vol = 0, 0
    
    random.shuffle(customer_list)
    
    for c in customer_list:
        c_row = customers[customers['ID'] == c].iloc[0]
        c_w = c_row['重量'] if pd.notna(c_row['重量']) else 0
        c_v = c_row['体积'] if pd.notna(c_row['体积']) else 0
        
        if current_load + c_w > vehicle_params['load'] or current_vol + c_v > vehicle_params['volume']:
            continue
            
        best_pos, best_delta = -1, float('inf')
        for pos in range(1, len(route)):
            prev_n = route[pos-1]
            next_n = route[pos]
            delta = distance_matrix[prev_n][c] + distance_matrix[c][next_n] - distance_matrix[prev_n][next_n]
            if delta < best_delta:
                best_delta, best_pos = delta, pos
        
        if best_pos != -1:
            route.insert(best_pos, c)
            assigned.append(c)
            current_load += c_w
            current_vol += c_v
    
    return route, assigned

def generate_initial_solution(customers, distance_matrix):
    solution = []
    used_vehicles = {v:0 for v in VEHICLE_PARAMS.keys()}
    
    # 第一层 NA
    print("=== 第一层：NA类客户分配 ===")
    na_customers = customers[customers['type'] == 'NA']['ID'].tolist()
    for v_type in ['E1', 'E2']:
        while used_vehicles[v_type] < VEHICLE_PARAMS[v_type]['count'] and na_customers:
            route, assigned = greedy_insertion_initial(v_type, na_customers, customers, distance_matrix)
            if not assigned: break
            total_cost, cost_detail, _, is_violate = calc_route_cost(v_type, route, customers, distance_matrix)
            if not is_violate:
                solution.append({'vehicle_type': v_type, 'route': route, 'cost_detail': cost_detail, 'total_cost': total_cost})
                used_vehicles[v_type] += 1
                na_customers = [c for c in na_customers if c not in assigned]
    
    # 第二层 NB
    print("=== 第二层：NB类客户分配 ===")
    nb_customers = customers[customers['type'] == 'NB']['ID'].tolist()
    for v_type in ['E1', 'E2']:
        while used_vehicles[v_type] < VEHICLE_PARAMS[v_type]['count'] and nb_customers:
            route, assigned = greedy_insertion_initial(v_type, nb_customers, customers, distance_matrix)
            if not assigned: break
            total_cost, cost_detail, _, is_violate = calc_route_cost(v_type, route, customers, distance_matrix)
            if not is_violate:
                solution.append({'vehicle_type': v_type, 'route': route, 'cost_detail': cost_detail, 'total_cost': total_cost})
                used_vehicles[v_type] += 1
                nb_customers = [c for c in nb_customers if c not in assigned]
    
    # 第三层 其他
    print("=== 第三层：剩余客户混合分配 ===")
    remaining_customers = nb_customers + customers[customers['type'].isin(['NC', 'ND'])]['ID'].tolist()
    for v_type in ['E1', 'E2', 'F1', 'F2', 'F3']:
        while used_vehicles[v_type] < VEHICLE_PARAMS[v_type]['count'] and remaining_customers:
            route, assigned = greedy_insertion_initial(v_type, remaining_customers, customers, distance_matrix)
            if not assigned: break
            total_cost, cost_detail, _, is_violate = calc_route_cost(v_type, route, customers, distance_matrix)
            if not is_violate:
                solution.append({'vehicle_type': v_type, 'route': route, 'cost_detail': cost_detail, 'total_cost': total_cost})
                used_vehicles[v_type] += 1
                remaining_customers = [c for c in remaining_customers if c not in assigned]

    print(f"初始解生成成功！剩余未分配：{len(remaining_customers)}")
    print(f"初始解总成本：{sum([r['total_cost'] for r in solution]):.2f}元")
    return solution, used_vehicles

# ===================== ALNS算法 =====================
class ALNS:
    def __init__(self, customers, distance_matrix, init_solution, init_used_vehicles):
        self.customers = customers
        self.distance_matrix = distance_matrix
        self.current_solution = deepcopy(init_solution)
        self.best_solution = deepcopy(init_solution)
        self.current_cost = sum([r['total_cost'] for r in self.current_solution])
        self.best_cost = self.current_cost
        self.used_vehicles = init_used_vehicles
        self.params = ALNS_PARAMS
        self.temp = self.params['init_temp']
        self.customer_type_map = dict(zip(customers['ID'], customers['type']))
        print("ALNS初始化完成")
    
    def random_removal(self, num_remove=3):
        all_customers = []
        customer_weights = []  # 权重列表
        
        for route in self.current_solution:
            for c in route['route'][1:-1]:
                all_customers.append(c)
                # 类型A客户权重乘以rho_A=0.15
                if self.customer_type_map.get(c) == 'NA':
                    customer_weights.append(self.params['rho_A'])
                else:
                    customer_weights.append(1.0)
        
        if not all_customers: return []
        
        num_remove = min(num_remove, len(all_customers))
        # 加权随机采样，替代原来的均匀采样
        removed = random.choices(all_customers, weights=customer_weights, k=num_remove)
        removed = list(set(removed))  # 去重
        
        new_sol = []
        for r in self.current_solution:
            new_r = [0] + [c for c in r['route'][1:-1] if c not in removed] + [0]
            if len(new_r) > 2:
                cost, detail, _, _ = calc_route_cost(r['vehicle_type'], new_r, self.customers, self.distance_matrix)
                new_sol.append({'vehicle_type': r['vehicle_type'], 'route': new_r, 'cost_detail': detail, 'total_cost': cost})
            else:
                self.used_vehicles[r['vehicle_type']] -= 1
        self.current_solution = new_sol
        return removed
    
    # 最差移除算子
    def worst_removal(self, num_remove=3):
        """最差移除：移除边际成本最高的客户，类型A低概率纳入"""
        customer_cost = {}
        for route in self.current_solution:
            for c in route['route'][1:-1]:
                # 计算移除该客户的成本变化
                new_r = [0] + [x for x in route['route'][1:-1] if x != c] + [0]
                orig_cost = route['total_cost']
                new_cost, _, _, _ = calc_route_cost(route['vehicle_type'], new_r, self.customers, self.distance_matrix)
                # 类型A成本乘以衰减因子
                weight = self.params['rho_A'] if self.customer_type_map.get(c) == 'NA' else 1.0
                customer_cost[c] = (orig_cost - new_cost) * weight
        
        sorted_cust = sorted(customer_cost.items(), key=lambda x: x[1], reverse=True)
        num_remove = min(num_remove, len(sorted_cust))
        if num_remove == 0:
            return []
        removed = [c for c, _ in sorted_cust[:num_remove]]
        
        # 后续逻辑同random_removal
        new_sol = []
        for r in self.current_solution:
            new_r = [0] + [c for c in r['route'][1:-1] if c not in removed] + [0]
            if len(new_r) > 2:
                cost, detail, _, _ = calc_route_cost(r['vehicle_type'], new_r, self.customers, self.distance_matrix)
                new_sol.append({'vehicle_type': r['vehicle_type'], 'route': new_r, 'cost_detail': detail, 'total_cost': cost})
            else:
                self.used_vehicles[r['vehicle_type']] -= 1
        self.current_solution = new_sol
        return removed
    
    def greedy_repair(self, removed_customers):
        for c in removed_customers:
            c_type = self.customer_type_map.get(c, 'ND')
            best_cost = float('inf')
            best_idx, best_pos = None, None
            
            for idx, route in enumerate(self.current_solution):
                v_type = route['vehicle_type']
                v_params = VEHICLE_PARAMS[v_type]
                if c_type == 'NA' and v_params['type'] == 'fuel': continue
                
                # 简单载重检查
                current_w = sum([self.customers[self.customers['ID']==x]['重量'].iloc[0] for x in route['route'][1:-1]])
                current_v = sum([self.customers[self.customers['ID']==x]['体积'].iloc[0] for x in route['route'][1:-1]])
                c_w = self.customers[self.customers['ID']==c]['重量'].iloc[0]
                c_v = self.customers[self.customers['ID']==c]['体积'].iloc[0]
                if current_w + c_w > v_params['load'] or current_v + c_v > v_params['volume']: continue
                
                for pos in range(1, len(route['route'])):
                    new_r = route['route'][:pos] + [c] + route['route'][pos:]
                    new_cost, _, _, is_v = calc_route_cost(v_type, new_r, self.customers, self.distance_matrix)
                    if not is_v and new_cost < best_cost:
                        best_cost = new_cost
                        best_idx, best_pos = idx, pos
            
            # 尝试新建
            if best_idx is None:
                avail = ['E1', 'E2'] if c_type == 'NA' else ['E1', 'E2', 'F1', 'F2', 'F3']
                for vt in avail:
                    if self.used_vehicles[vt] < VEHICLE_PARAMS[vt]['count']:
                        new_r = [0, c, 0]
                        cost, detail, _, is_v = calc_route_cost(vt, new_r, self.customers, self.distance_matrix)
                        if not is_v:
                            self.current_solution.append({'vehicle_type': vt, 'route': new_r, 'cost_detail': detail, 'total_cost': cost})
                            self.used_vehicles[vt] += 1
                            break
            else:
                r = self.current_solution[best_idx]
                new_r = r['route'][:best_pos] + [c] + r['route'][best_pos:]
                cost, detail, _, _ = calc_route_cost(r['vehicle_type'], new_r, self.customers, self.distance_matrix)
                self.current_solution[best_idx] = {'vehicle_type': r['vehicle_type'], 'route': new_r, 'cost_detail': detail, 'total_cost': cost}
        return

    def run(self):
        print("=== ALNS开始运行 ===")
        cost_history = [self.best_cost]
        for iter in tqdm(range(self.params['max_iter']), desc="ALNS迭代"):
            temp_sol = deepcopy(self.current_solution)
            temp_used = deepcopy(self.used_vehicles)
            temp_cost = self.current_cost
            
            # 随机选择移除算子：50%概率用random，50%概率用worst
            if random.random() < 0.5:
                removed = self.random_removal(num_remove=max(2, int(len(self.customers)*0.03)))
            else:
                removed = self.worst_removal(num_remove=max(2, int(len(self.customers)*0.03)))
            
            self.greedy_repair(removed)
            
            new_cost = sum([r['total_cost'] for r in self.current_solution])
            delta = new_cost - temp_cost
            has_v = any([r['cost_detail']['violate_punish'] > 0 for r in self.current_solution])
            
            accept = False
            if not has_v:
                if delta < 0:
                    accept = True
                    if new_cost < self.best_cost:
                        self.best_solution = deepcopy(self.current_solution)
                        self.best_cost = new_cost
                else:
                    prob = math.exp(-delta / self.temp) if self.temp > 0 else 0
                    if random.random() < prob: accept = True
            
            if not accept:
                self.current_solution = temp_sol
                self.used_vehicles = temp_used
                self.current_cost = temp_cost
            else:
                self.current_cost = new_cost
            
            self.temp *= self.params['cooling_rate']
            cost_history.append(self.best_cost)
            
        print(f"ALNS结束，最优成本：{self.best_cost:.2f}")
        return self.best_solution, self.best_cost, cost_history

# ===================== 结果分析与主函数 =====================
def analyze_result(solution, customers):
    print("\n===================== 结果分析 =====================")
    total_cost = sum([r['total_cost'] for r in solution])
    print(f"总成本：{total_cost:.2f}元")
    
    vehicle_usage = {v:0 for v in VEHICLE_PARAMS.keys()}
    for r in solution:
        vehicle_usage[r['vehicle_type']] += 1
    print(f"车辆使用：{vehicle_usage}")
    
    total_carbon = 0
    for r in solution:
        total_carbon += (r['cost_detail']['carbon_cost'] / COST_PARAMS['carbon_price'])
    print(f"碳排放：{total_carbon:.2f}kg")
    return

# ===================== 结果分析函数 =====================
def detailed_analyze_result(best_solution, init_solution, init_cost, customers, distance_matrix):
    print("\n" + "="*80)
    print(" " * 20 + "【超详细结果分析报告】")
    print("="*80)
    
    # ===================== 1. 最优解总成本构成明细 =====================
    print("\n【1/6】最优解总成本构成明细")
    print("-" * 60)
    
    total_detail = {
        'start_cost': 0, 'energy_cost': 0, 'carbon_cost': 0,
        'early_wait_cost': 0, 'late_punish_cost': 0,
        'restrict_wait_cost': 0, 'violate_punish': 0
    }
    total_carbon = 0
    fuel_carbon = 0
    elec_carbon = 0
    
    for route_info in best_solution:
        v_type = route_info['vehicle_type']
        detail = route_info['cost_detail']
        for key in total_detail.keys():
            total_detail[key] += detail[key]
        
        # 分类统计碳排放
        route_carbon = detail['carbon_cost'] / COST_PARAMS['carbon_price'] if COST_PARAMS['carbon_price'] > 0 else 0
        total_carbon += route_carbon
        if VEHICLE_PARAMS[v_type]['type'] == 'fuel':
            fuel_carbon += route_carbon
        else:
            elec_carbon += route_carbon
    
    best_cost = sum(total_detail.values())
    
    # 打印成本明细
    cost_names = {
        'start_cost': '车辆启动成本',
        'energy_cost': '能耗成本（油费+电费）',
        'carbon_cost': '碳排放成本',
        'early_wait_cost': '时间窗早到等待成本',
        'late_punish_cost': '时间窗晚到惩罚成本',
        'restrict_wait_cost': '限行政策等待成本',
        'violate_punish': '违规惩罚成本（应为0）'
    }
    
    print(f"✅ 最优解总成本：{best_cost:.2f} 元")
    print(f"✅ 总碳排放量：{total_carbon:.2f} kg")
    print("\n各项成本明细：")
    for key, value in total_detail.items():
        percentage = (value / best_cost * 100) if best_cost > 0 else 0
        print(f"  - {cost_names[key]}：{value:.2f} 元，占比 {percentage:.2f}%")
    
    # 碳排放分类
    print("\n碳排放分类统计：")
    print(f"  - 燃油车碳排放：{fuel_carbon:.2f} kg，占比 {(fuel_carbon/total_carbon*100):.2f}%" if total_carbon > 0 else "  - 燃油车碳排放：0 kg")
    print(f"  - 新能源车碳排放：{elec_carbon:.2f} kg，占比 {(elec_carbon/total_carbon*100):.2f}%" if total_carbon > 0 else "  - 新能源车碳排放：0 kg")
    
    # ===================== 2. 车辆使用结构深度分析 =====================
    print("\n【2/6】车辆使用结构深度分析")
    print("-" * 60)
    
    vehicle_usage = {v:0 for v in VEHICLE_PARAMS.keys()}
    vehicle_load_info = {v: {'total_load': 0, 'total_vol': 0, 'max_load': 0, 'max_vol': 0} for v in VEHICLE_PARAMS.keys()}
    
    for route_info in best_solution:
        v_type = route_info['vehicle_type']
        vehicle_usage[v_type] += 1
        
        # 计算该路径的总载重与容积
        route = route_info['route']
        route_load = 0
        route_vol = 0
        for c_id in route[1:-1]:
            c_match = customers[customers['ID'] == c_id]
            if c_match.empty: continue
            c = c_match.iloc[0]
            w = c['重量'] if pd.notna(c['重量']) else 0
            v = c['体积'] if pd.notna(c['体积']) else 0
            route_load += w
            route_vol += v
        
        vehicle_load_info[v_type]['total_load'] += route_load
        vehicle_load_info[v_type]['total_vol'] += route_vol
        vehicle_load_info[v_type]['max_load'] = max(vehicle_load_info[v_type]['max_load'], route_load)
        vehicle_load_info[v_type]['max_vol'] = max(vehicle_load_info[v_type]['max_vol'], route_vol)
    
    print("车型使用数量统计：")
    total_vehicles = sum(vehicle_usage.values())
    for v_type, count in vehicle_usage.items():
        params = VEHICLE_PARAMS[v_type]
        usage_percentage = (count / params['count'] * 100) if params['count'] > 0 else 0
        total_percentage = (count / total_vehicles * 100) if total_vehicles > 0 else 0
        print(f"  - {v_type}（{params['type']}）：启用 {count} 辆，可用 {params['count']} 辆，使用率 {usage_percentage:.2f}%，占总车辆 {total_percentage:.2f}%")
        if count > 0:
            avg_load = vehicle_load_info[v_type]['total_load'] / count
            avg_vol = vehicle_load_info[v_type]['total_vol'] / count
            load_util = (avg_load / params['load'] * 100) if params['load'] > 0 else 0
            vol_util = (avg_vol / params['volume'] * 100) if params['volume'] > 0 else 0
            print(f"    平均载重：{avg_load:.2f} kg（利用率 {load_util:.2f}%），平均容积：{avg_vol:.4f} m³（利用率 {vol_util:.2f}%）")
    
    # ===================== 3. 每辆车的完整调度方案 =====================
    print("\n【3/6】每辆车的完整调度方案")
    print("-" * 60)
    
    for idx, route_info in enumerate(best_solution):
        v_type = route_info['vehicle_type']
        route = route_info['route']
        print(f"\n🚗 车辆 {idx+1}（车型：{v_type}，类型：{VEHICLE_PARAMS[v_type]['type']}）")
        print(f"  行驶路径：{' -> '.join(map(str, route))}")
        
        # 重新计算到达时间与详细成本
        _, detail, _, _ = calc_route_cost(v_type, route, customers, distance_matrix)
        print(f"  单车辆成本：{sum(detail.values()):.2f} 元")
        print(f"  成本明细：启动 {detail['start_cost']:.2f} + 能耗 {detail['energy_cost']:.2f} + 碳排 {detail['carbon_cost']:.2f} + 早等 {detail['early_wait_cost']:.2f} + 晚罚 {detail['late_punish_cost']:.2f} + 限行等 {detail['restrict_wait_cost']:.2f}")
        
        # 打印服务的客户类型
        cust_types = []
        for c_id in route[1:-1]:
            c_match = customers[customers['ID'] == c_id]
            if c_match.empty: continue
            cust_types.append(c_match.iloc[0]['type'])
        print(f"  服务客户类型：{', '.join(cust_types)}")
    
    # ===================== 4. 客户服务情况统计 =====================
    print("\n【4/6】客户服务情况统计")
    print("-" * 60)
    
    # 统计原始客户分类
    orig_type_count = {
        'NA': sum(customers['type'] == 'NA'),
        'NB': sum(customers['type'] == 'NB'),
        'NC': sum(customers['type'] == 'NC'),
        'ND': sum(customers['type'] == 'ND'),
    }
    
    # 统计服务的客户分类
    served_cust = set()
    served_type_count = {'NA': 0, 'NB': 0, 'NC': 0, 'ND': 0}
    for route_info in best_solution:
        for c_id in route_info['route'][1:-1]:
            served_cust.add(c_id)
            c_match = customers[customers['ID'] == c_id]
            if c_match.empty: continue
            c_type = c_match.iloc[0]['type']
            served_type_count[c_type] += 1
    
    print(f"总客户数：{len(customers)}，已服务客户数：{len(served_cust)}，服务覆盖率：{(len(served_cust)/len(customers)*100):.2f}%")
    print("\n各类客户服务情况：")
    for t in ['NA', 'NB', 'NC', 'ND']:
        coverage = (served_type_count[t] / orig_type_count[t] * 100) if orig_type_count[t] > 0 else 100
        print(f"  - {t}类客户：原始 {orig_type_count[t]} 个，已服务 {served_type_count[t]} 个，覆盖率 {coverage:.2f}%")
    
    # ===================== 5. 初始解 vs 最优解对比 =====================
    print("\n【5/6】初始解 vs 最优解对比")
    print("-" * 60)
    
    cost_drop = init_cost - best_cost
    cost_drop_percentage = (cost_drop / init_cost * 100) if init_cost > 0 else 0
    
    print(f"初始解总成本：{init_cost:.2f} 元")
    print(f"最优解总成本：{best_cost:.2f} 元")
    print(f"成本下降幅度：{cost_drop:.2f} 元，下降比例 {cost_drop_percentage:.2f}%")
    
    # 车辆使用对比
    init_vehicle_count = sum([1 for _ in init_solution])
    best_vehicle_count = len(best_solution)
    print(f"初始解车辆数：{init_vehicle_count} 辆")
    print(f"最优解车辆数：{best_vehicle_count} 辆")
    print(f"车辆数变化：{best_vehicle_count - init_vehicle_count} 辆")
    
    print("\n" + "="*80)
    print(" " * 20 + "【详细结果分析报告结束】")
    print("="*80)
    return

# ===================== 主函数 =====================
def main():
    # 1. 数据读取与预处理
    customers, distance_matrix, depot_info = load_and_preprocess_data()
    
    # 2. 可行性校验
    if not feasibility_check(customers):
        return
    
    # 3. 初始解生成
    print("\n" + "="*80)
    init_solution, init_used_vehicles = generate_initial_solution(customers, distance_matrix)
    init_cost = sum([r['total_cost'] for r in init_solution])
    print(f"初始解生成完成，初始总成本：{init_cost:.2f} 元")
    print("="*80)
    
    # 4. ALNS算法运行
    alns = ALNS(customers, distance_matrix, init_solution, init_used_vehicles)
    best_solution, best_cost, cost_history = alns.run()
    
    # 5. 基础结果分析
    analyze_result(best_solution, customers)
    
    # 6. 结果分析
    detailed_analyze_result(best_solution, init_solution, init_cost, customers, distance_matrix)
    
    # 7. 收敛曲线绘图
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, linewidth=2, color='#1f77b4')
    plt.title('Cost Convergence Curve (ALNS Optimization)', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Total Cost (Yuan)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    return

if __name__ == "__main__":
    main()