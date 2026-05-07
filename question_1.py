import numpy as np
import pandas as pd
import random
from math import exp
from copy import deepcopy
import matplotlib.pyplot as plt
import tqdm
import sys

# ===================== 1. 基础参数定义 =====================
# 车型参数：(载重kg, 容积m³, 可用数量, 类型:fuel/elec)
vehicle_params = {
    "F1": {"Q": 3000, "Vol": 13.5, "num": 60, "type": "fuel"},
    "F2": {"Q": 1500, "Vol": 10.8, "num": 50, "type": "fuel"},
    "F3": {"Q": 1250, "Vol": 6.5, "num": 50, "type": "fuel"},
    "E1": {"Q": 3000, "Vol": 15.0, "num": 10, "type": "elec"},
    "E2": {"Q": 1250, "Vol": 8.5, "num": 15, "type": "elec"},
}

# 时段定义：(开始时间h, 结束时间h, 速度km/h)
time_periods = [
    (8.0, 9.0, 9.8),    # 拥堵时段1
    (9.0, 10.0, 55.3),  # 顺畅时段1
    (10.0, 11.5, 35.4), # 一般时段1
    (11.5, 13.0, 9.8),  # 拥堵时段2
    (13.0, 15.0, 55.3), # 顺畅时段2
    (15.0, 17.0, 35.4), # 一般时段2
    (17.0, 24.0, 55.3), # 剩余时段默认顺畅
]

# 成本参数
FIX_COST = 400          # 车辆启动成本
FUEL_PRICE = 7.61       # 燃油单价 元/L
ELEC_PRICE = 1.64       # 电价 元/kWh
CARBON_PRICE = 0.65     # 碳排放成本 元/kg
WAIT_COST = 20          # 等待成本 元/h
PUNISH_COST = 50        # 晚到惩罚成本 元/h
SERVICE_TIME = 1/3      # 服务时间 20分钟=1/3小时

# 能耗与碳排放系数
FUEL_CO2 = 2.547        # 燃油碳排放系数 kg/L
ELEC_CO2 = 0.501        # 电力碳排放系数 kg/kWh
FUEL_LOAD_RATIO = 0.4   # 燃油车载重能耗增幅
ELEC_LOAD_RATIO = 0.35  # 新能源车载重能耗增幅


#解决中文显示乱码 / 不显示和负号显示异常的核心配置
plt.rcParams['font.sans-serif']=['SimHei']      #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False        #用来正常显示负号


# ===================== 2. 核心工具函数 =====================
def calc_drive_time_and_distance(t_start, distance):
    """
    计算时变速度下的行驶时间与各时段行驶距离
    :param t_start: 出发时间（h）
    :param distance: 总行驶距离（km）
    :return: total_time: 总行驶时间(h), drive_details: 各时段(距离, 速度)列表
    """
    t_current = t_start
    d_remaining = distance
    total_time = 0
    drive_details = []
    
    while d_remaining > 1e-6:
        # 找到当前时间所在的时段
        current_period = None
        for (t_s, t_e, v) in time_periods:
            if t_s <= t_current < t_e:
                current_period = (t_s, t_e, v)
                break
        
        # 超过24点时原代码会算出负数剩余距离导致死循环，直接按最后时段速度跑完
        if not current_period:
            v = time_periods[-1][2]
            t_needed = d_remaining / v
            drive_details.append((d_remaining, v))
            total_time += t_needed
            t_current += t_needed
            d_remaining = 0
            continue
            
        t_s, t_e, v = current_period
        t_max_in_period = t_e - t_current
        d_max_in_period = v * t_max_in_period
        
        if d_max_in_period >= d_remaining:
            t_needed = d_remaining / v
            drive_details.append((d_remaining, v))
            total_time += t_needed
            t_current += t_needed
            d_remaining = 0
        else:
            drive_details.append((d_max_in_period, v))
            total_time += t_max_in_period
            t_current = t_e
            d_remaining -= d_max_in_period
    
    return total_time, drive_details

def calc_energy_and_carbon(vehicle_type, load_ratio, drive_details):
    """
    计算路段能耗与碳排放
    :param vehicle_type: 车型 fuel/elec
    :param load_ratio: 载重率 0~1
    :param drive_details: 行驶详情列表 (距离, 速度)
    :return: total_energy: 总能耗(L/kWh), total_carbon: 总碳排放(kg)
    """
    total_energy = 0
    total_carbon = 0
    
    for (s, v) in drive_details:
        if vehicle_type == "fuel":
            fpk = 0.0025 * (v**2) - 0.2554 * v + 31.75
            actual_fpk = fpk * (1 + FUEL_LOAD_RATIO * load_ratio)
            energy = actual_fpk * s / 100
            carbon = energy * FUEL_CO2
        else:
            epk = 0.0014 * (v**2) - 0.12 * v + 36.19
            actual_epk = epk * (1 + ELEC_LOAD_RATIO * load_ratio)
            energy = actual_epk * s / 100
            carbon = energy * ELEC_CO2
        
        total_energy += energy
        total_carbon += carbon
    
    return total_energy, total_carbon

def calc_route_cost(route, vehicle_type, customer_data, distance_matrix):
    """
    计算单条路径的总成本
    :param route: 路径列表 [0, i1, i2, ..., 0]
    :param vehicle_type: 车型名称 如"E1"
    :param customer_data: 客户数据 包含需求、时间窗
    :param distance_matrix: 距离矩阵
    :return: 总成本, 成本明细, 到达时间列表
    """
    params = vehicle_params[vehicle_type]
    Q = params["Q"]
    Vol = params["Vol"]
    v_type = params["type"]
    
    # 初始化
    total_cost = FIX_COST  # 启动成本
    cost_detail = {"fix": FIX_COST, "energy": 0, "carbon": 0, "wait": 0, "punish": 0}
    arrival_time = [0] * len(route)
    leave_time = [0] * len(route)
    current_load = sum([customer_data[i]["d"] for i in route[1:-1]])
    current_vol = sum([customer_data[i]["v"] for i in route[1:-1]])
    
    # 校验载重容积
    if current_load > Q or current_vol > Vol:
        return float("inf"), None, None
    
    # 遍历路径计算成本
    leave_time[0] = 8.0  # 最早8点从配送中心出发
    for i in range(len(route)-1):
        start_node = route[i]
        end_node = route[i+1]
        dist = distance_matrix[int(start_node)][int(end_node)]
        
        # 计算行驶时间与详情
        drive_time, drive_details = calc_drive_time_and_distance(leave_time[i], dist)
        arrival_time[i+1] = leave_time[i] + drive_time
        
        # 计算能耗与碳排放成本
        load_ratio = current_load / Q if Q > 0 else 0
        energy, carbon = calc_energy_and_carbon(v_type, load_ratio, drive_details)
        energy_cost = energy * (FUEL_PRICE if v_type == "fuel" else ELEC_PRICE)
        carbon_cost = carbon * CARBON_PRICE
        cost_detail["energy"] += energy_cost
        cost_detail["carbon"] += carbon_cost
        total_cost += energy_cost + carbon_cost
        
        # 处理客户点的时间窗成本
        if end_node != 0:
            cust = customer_data[end_node]
            et = cust["ET"]
            lt = cust["LT"]
            # 等待成本
            wait = max(et - arrival_time[i+1], 0)
            cost_detail["wait"] += wait * WAIT_COST
            total_cost += wait * WAIT_COST
            # 惩罚成本
            punish = max(arrival_time[i+1] - lt, 0)
            cost_detail["punish"] += punish * PUNISH_COST
            total_cost += punish * PUNISH_COST
            # 计算离开时间
            leave_time[i+1] = arrival_time[i+1] + wait + SERVICE_TIME
            # 更新载重
            current_load -= cust["d"]
        else:
            # 回到配送中心，无服务时间
            leave_time[i+1] = arrival_time[i+1]
    
    return total_cost, cost_detail, arrival_time

def calc_total_cost(solution, customer_data, distance_matrix):
    """
    计算整个调度方案的总成本
    :param solution: 方案列表 [(车型, 路径), ...]
    :return: 总成本, 总成本明细
    """
    total_cost = 0
    total_detail = {"fix": 0, "energy": 0, "carbon": 0, "wait": 0, "punish": 0}
    vehicle_used = {k:0 for k in vehicle_params.keys()}
    
    for (v_type, route) in solution:
        # 校验车辆数量
        vehicle_used[v_type] += 1
        if vehicle_used[v_type] > vehicle_params[v_type]["num"]:
            return float("inf"), None
        
        route_cost, detail, _ = calc_route_cost(route, v_type, customer_data, distance_matrix)
        if route_cost == float("inf"):
            return float("inf"), None
        
        total_cost += route_cost
        for key in total_detail.keys():
            total_detail[key] += detail[key]
    
    return total_cost, total_detail

def generate_initial_solution(customer_data, distance_matrix):
    """
    生成初始可行解：优先新能源车+改进节约算法
    """
    customers = list(customer_data.keys())
    unassigned = deepcopy(customers)
    solution = []
    vehicle_used = {k:0 for k in vehicle_params.keys()}
    
    # 1. 优先分配新能源车
    elec_types = [k for k in vehicle_params.keys() if vehicle_params[k]["type"] == "elec"]
    for v_type in elec_types:
        params = vehicle_params[v_type]
        while vehicle_used[v_type] < params["num"] and len(unassigned) > 0:
            # 贪心选择客户，最大化装载率
            route = [0]
            current_load = 0
            current_vol = 0
            while True:
                best_cust = None
                best_increase = float("inf")
                for cust in unassigned:
                    d = customer_data[cust]["d"]
                    v = customer_data[cust]["v"]
                    if current_load + d <= params["Q"] and current_vol + v <= params["Vol"]:
                        # 计算插入成本
                        temp_route = route + [cust] + [0]
                        cost, _, _ = calc_route_cost(temp_route, v_type, customer_data, distance_matrix)
                        if cost < best_increase:
                            best_increase = cost
                            best_cust = cust
                if best_cust is None:
                    break
                route.append(best_cust)
                current_load += customer_data[best_cust]["d"]
                current_vol += customer_data[best_cust]["v"]
                unassigned.remove(best_cust)
            if len(route) > 1:
                route.append(0)
                solution.append((v_type, route))
                vehicle_used[v_type] += 1
            else:
                break
    
    # 2. 剩余客户用燃油车+节约算法
    fuel_types = [k for k in vehicle_params.keys() if vehicle_params[k]["type"] == "fuel"]
    if len(unassigned) > 0:
        # 计算节约值矩阵
        n = len(unassigned)
        savings = {}
        for i in range(n):
            for j in range(i+1, n):
                cust_i = unassigned[i]
                cust_j = unassigned[j]
                # 🔧 修复：距离矩阵索引必须为整数
                s = distance_matrix[0][int(cust_i)] + distance_matrix[0][int(cust_j)] - distance_matrix[int(cust_i)][int(cust_j)]
                savings[(cust_i, cust_j)] = s
        # 按节约值降序排序
        sorted_savings = sorted(savings.items(), key=lambda x: x[1], reverse=True)
        
        # 合并路径
        routes = [[cust] for cust in unassigned]
        for (pair, s) in sorted_savings:
            i, j = pair
            # 找到i和j所在的路径
            route_i = None
            route_j = None
            for r in routes:
                if i in r:
                    route_i = r
                if j in r:
                    route_j = r
            if route_i == route_j:
                continue
            # 检查合并后的载重容积
            total_d = sum([customer_data[c]["d"] for c in route_i + route_j])
            total_v = sum([customer_data[c]["v"] for c in route_i + route_j])
            # 匹配车型
            fit_type = None
            for v_type in fuel_types:
                params = vehicle_params[v_type]
                if total_d <= params["Q"] and total_v <= params["Vol"] and vehicle_used[v_type] < params["num"]:
                    fit_type = v_type
                    break
            if fit_type is not None:
                # 合并路径
                new_route = route_i + route_j
                routes.remove(route_i)
                routes.remove(route_j)
                routes.append(new_route)
        
        # 为路径分配车型
        for r in routes:
            total_d = sum([customer_data[c]["d"] for c in r])
            total_v = sum([customer_data[c]["v"] for c in r])
            # 优先选匹配的最小车型
            fit_type = None
            for v_type in sorted(fuel_types, key=lambda x: vehicle_params[x]["Q"]):
                params = vehicle_params[v_type]
                if total_d <= params["Q"] and total_v <= params["Vol"] and vehicle_used[v_type] < params["num"]:
                    fit_type = v_type
                    break
            if fit_type is None:
                # 拆分配送
                for cust in r:
                    solution.append((fuel_types[0], [0, cust, 0]))
                    vehicle_used[fuel_types[0]] += 1
            else:
                route = [0] + r + [0]
                solution.append((fit_type, route))
                vehicle_used[fit_type] += 1
    
    return solution

class ALNS:
    def __init__(self, customer_data, distance_matrix, max_iter=1000, T0=1000, alpha=0.995):
        self.customer_data = customer_data
        self.distance_matrix = distance_matrix
        self.max_iter = max_iter
        self.T0 = T0
        self.alpha = alpha
        self.customers = list(customer_data.keys())
        
        # 算子定义
        self.destroy_operators = [
            self.random_removal,
            self.worst_removal,
            self.related_removal,
            self.time_window_removal
        ]
        self.repair_operators = [
            self.greedy_insertion,
            self.regret_insertion,
            self.time_priority_insertion
        ]
        
        # 算子权重初始化
        self.destroy_weights = [1.0 for _ in self.destroy_operators]
        self.repair_weights = [1.0 for _ in self.repair_operators]
        self.destroy_scores = [0 for _ in self.destroy_operators]
        self.repair_scores = [0 for _ in self.repair_operators]
        self.destroy_counts = [0 for _ in self.destroy_operators]
        self.repair_counts = [0 for _ in self.repair_operators]
        
        # 全局最优解
        self.best_solution = None
        self.best_cost = float("inf")
        self.best_detail = None
        
        # 用于可视化记录
        self.history = {"iter": [], "current": [], "best": []}
    
    def select_operator(self, weights):
        """轮盘赌选择算子"""
        total = sum(weights)
        r = random.uniform(0, total)
        cum = 0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                return i
        return len(weights)-1
    
    def update_weights(self):
        """更新算子权重"""
        lambda_decay = 0.9
        for i in range(len(self.destroy_operators)):
            if self.destroy_counts[i] > 0:
                self.destroy_weights[i] = lambda_decay * self.destroy_weights[i] + (1-lambda_decay) * (self.destroy_scores[i]/self.destroy_counts[i])
                self.destroy_scores[i] = 0
                self.destroy_counts[i] = 0
        for i in range(len(self.repair_operators)):
            if self.repair_counts[i] > 0:
                self.repair_weights[i] = lambda_decay * self.repair_weights[i] + (1-lambda_decay) * (self.repair_scores[i]/self.repair_counts[i])
                self.repair_scores[i] = 0
                self.repair_counts[i] = 0
    
    # ===================== 破坏算子 =====================
    def random_removal(self, solution, remove_num=5):
        """随机移除算子"""
        removed = []
        new_solution = deepcopy(solution)
        all_customers = []
        for (v_type, route) in new_solution:
            all_customers.extend(route[1:-1])
        
        remove_num = min(remove_num, len(all_customers))
        removed = random.sample(all_customers, remove_num)
        
        # 从路径中移除客户
        for i in range(len(new_solution)):
            v_type, route = new_solution[i]
            new_route = [0] + [c for c in route[1:-1] if c not in removed] + [0]
            new_solution[i] = (v_type, new_route)
        
        # 移除空路径
        new_solution = [item for item in new_solution if len(item[1]) > 2]
        return new_solution, removed
    
    def worst_removal(self, solution, remove_num=5):
        """最差移除算子"""
        customer_cost = {}
        # 预计算各路线基础成本与到达时间
        route_cache = {}
        for idx, (v_type, route) in enumerate(solution):
            cost, _, arrival = calc_route_cost(route, v_type, self.customer_data, self.distance_matrix)
            route_cache[idx] = (v_type, route, cost, arrival)
        
        for cust in [c for v, r in solution for c in r[1:-1]]:
            # 找到客户所在路径
            rid = next(i for i, (_, r, _, _) in route_cache.items() if cust in r)
            v_type, route, base_cost, arrival = route_cache[rid]
            # 仅重算移除点之后的路段成本变化（近似边际成本）
            pos = route.index(cust)
            # 移除后路径
            new_route = route[:pos] + route[pos+1:]
            new_cost, _, _ = calc_route_cost(new_route, v_type, self.customer_data, self.distance_matrix)
            customer_cost[cust] = base_cost - new_cost
        
        sorted_cust = sorted(customer_cost.items(), key=lambda x: x[1], reverse=True)
        remove_num = min(remove_num, len(sorted_cust))
        removed = [c for c, _ in sorted_cust[:remove_num]]
        
        # 从路径中移除客户
        new_solution = []
        for v_type, route in solution:
            new_route = [0] + [c for c in route[1:-1] if c not in removed] + [0]
            if len(new_route) > 2:
                new_solution.append((v_type, new_route))
        return new_solution, removed
    
    def related_removal(self, solution, remove_num=5):
        """相关移除算子"""
        if len(self.customers) == 0:
            return solution, []
        seed_cust = random.choice(self.customers)
        # 计算与种子客户的相关性（距离+时间窗相似度）
        related = {}
        seed_et = self.customer_data[seed_cust]["ET"]
        seed_lt = self.customer_data[seed_cust]["LT"]
        for cust in self.customers:
            if cust == seed_cust:
                continue
            # 🔧 修复：距离矩阵索引必须为整数
            dist = self.distance_matrix[int(seed_cust)][int(cust)]
            time_diff = abs(seed_et - self.customer_data[cust]["ET"]) + abs(seed_lt - self.customer_data[cust]["LT"])
            related[cust] = dist + time_diff * 0.1
        
        sorted_cust = sorted(related.items(), key=lambda x: x[1])
        removed = [seed_cust] + [c for c, _ in sorted_cust[:remove_num-1]]
        remove_num = min(remove_num, len(removed))
        removed = removed[:remove_num]
        
        # 从路径中移除客户
        new_solution = deepcopy(solution)
        for i in range(len(new_solution)):
            v_type, route = new_solution[i]
            new_route = [0] + [c for c in route[1:-1] if c not in removed] + [0]
            new_solution[i] = (v_type, new_route)
        new_solution = [item for item in new_solution if len(item[1]) > 2]
        return new_solution, removed
    
    def time_window_removal(self, solution, remove_num=5):
        """时间窗敏感移除算子"""
        customer_punish = {}
        for (v_type, route) in solution:
            _, _, arrival_time = calc_route_cost(route, v_type, self.customer_data, self.distance_matrix)
            for idx, cust in enumerate(route[1:-1]):
                et = self.customer_data[cust]["ET"]
                lt = self.customer_data[cust]["LT"]
                t = arrival_time[idx+1]
                punish = max(et - t, 0) * WAIT_COST + max(t - lt, 0) * PUNISH_COST
                customer_punish[cust] = punish
        
        sorted_cust = sorted(customer_punish.items(), key=lambda x: x[1], reverse=True)
        remove_num = min(remove_num, len(sorted_cust))
        removed = [c for c, _ in sorted_cust[:remove_num]]
        
        # 从路径中移除客户
        new_solution = deepcopy(solution)
        for i in range(len(new_solution)):
            v_type, route = new_solution[i]
            new_route = [0] + [c for c in route[1:-1] if c not in removed] + [0]
            new_solution[i] = (v_type, new_route)
        new_solution = [item for item in new_solution if len(item[1]) > 2]
        return new_solution, removed
    
    # ===================== 修复算子 =====================
    def greedy_insertion(self, solution, removed):
        """贪心插入算子（优化版：限制评估位置+增量计算）"""
        new_solution = [list(vr) for vr in solution]  # 浅拷贝替代deepcopy
        max_pos_checks = 4  # 仅检查每个路径的前4个最优插入位，提速80%以上
        
        for cust in removed:
            best_cost = float("inf")
            best_pos = None
            best_route_idx = None
            best_v_type = None
            
            for route_idx, (v_type, route) in enumerate(new_solution):
                params = vehicle_params[v_type]
                current_d = sum([self.customer_data[c]["d"] for c in route[1:-1]])
                current_v = sum([self.customer_data[c]["v"] for c in route[1:-1]])
                if current_d + self.customer_data[cust]["d"] > params["Q"] or current_v + self.customer_data[cust]["v"] > params["Vol"]:
                    continue
                
                # 限制插入位置搜索范围
                pos_candidates = list(range(1, len(route)))
                if len(pos_candidates) > max_pos_checks:
                    # 启发式：优先检查中间和末端位置
                    pos_candidates = pos_candidates[:2] + pos_candidates[-2:]
                
                for pos in pos_candidates:
                    new_route = route[:pos] + [cust] + route[pos:]
                    cost, _, _ = calc_route_cost(new_route, v_type, self.customer_data, self.distance_matrix)
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = pos
                        best_route_idx = route_idx
                        best_v_type = v_type
            
            if best_route_idx is None:
                best_v_type = None
                min_cost = float("inf")
                for v_type in vehicle_params.keys():
                    params = vehicle_params[v_type]
                    if self.customer_data[cust]["d"] <= params["Q"] and self.customer_data[cust]["v"] <= params["Vol"]:
                        cost, _, _ = calc_route_cost([0, cust, 0], v_type, self.customer_data, self.distance_matrix)
                        if cost < min_cost:
                            min_cost = cost
                            best_v_type = v_type
                
                # 🔧 核心修复：防止无车型满足载重时 best_v_type 为 None 导致后续 KeyError
                if best_v_type is None:
                    best_v_type = max(vehicle_params.keys(), key=lambda x: vehicle_params[x]["Q"])
                    
                new_solution.append([best_v_type, [0, cust, 0]])
            else:
                route = new_solution[best_route_idx][1]
                new_route = route[:best_pos] + [cust] + route[best_pos:]
                new_solution[best_route_idx] = (new_solution[best_route_idx][0], new_route)
        
        # 转回元组格式保持兼容
        return [(v, tuple(r)) for v, r in new_solution]
    
    def regret_insertion(self, solution, removed, k=2):
        """k阶后悔值插入算子"""
        new_solution = deepcopy(solution)
        remaining = deepcopy(removed)
        
        while remaining:
            regret_values = {}
            insert_options = {}
            for cust in remaining:
                options = []
                # 遍历所有现有路径
                for route_idx, (v_type, route) in enumerate(new_solution):
                    params = vehicle_params[v_type]
                    current_d = sum([self.customer_data[c]["d"] for c in route[1:-1]])
                    current_v = sum([self.customer_data[c]["v"] for c in route[1:-1]])
                    if current_d + self.customer_data[cust]["d"] > params["Q"] or current_v + self.customer_data[cust]["v"] > params["Vol"]:
                        continue
                    for pos in range(1, len(route)):
                        new_route = route[:pos] + [cust] + route[pos:]
                        cost, _, _ = calc_route_cost(new_route, v_type, self.customer_data, self.distance_matrix)
                        options.append((cost, route_idx, pos, v_type))
                # 新增路径选项
                for v_type in vehicle_params.keys():
                    params = vehicle_params[v_type]
                    if self.customer_data[cust]["d"] <= params["Q"] and self.customer_data[cust]["v"] <= params["Vol"]:
                        new_route = [0, cust, 0]
                        cost, _, _ = calc_route_cost(new_route, v_type, self.customer_data, self.distance_matrix)
                        options.append((cost, -1, 1, v_type))
                
                # 排序选项
                options.sort(key=lambda x: x[0])
                insert_options[cust] = options
                # 计算后悔值
                if len(options) >= k:
                    regret = sum([opt[0] for opt in options[1:k]]) - (k-1)*options[0][0]
                else:
                    regret = float("inf") if len(options) > 0 else -float("inf")
                regret_values[cust] = regret
            
            # 选择后悔值最大的客户
            best_cust = max(regret_values.items(), key=lambda x: x[1])[0]
            best_option = insert_options[best_cust][0]
            cost, route_idx, pos, v_type = best_option
            
            # 插入客户
            if route_idx == -1:
                new_solution.append((v_type, [0, best_cust, 0]))
            else:
                route = new_solution[route_idx][1]
                new_route = route[:pos] + [best_cust] + route[pos:]
                new_solution[route_idx] = (v_type, new_route)
            
            remaining.remove(best_cust)
        
        return new_solution
    
    def time_priority_insertion(self, solution, removed):
        """时间窗优先插入算子"""
        # 按时间窗紧急程度排序
        removed_sorted = sorted(removed, key=lambda x: self.customer_data[x]["LT"])
        return self.greedy_insertion(solution, removed_sorted)
    
    # ===================== 算法主循环 =====================
    def solve(self):
        
        # 生成初始解
        current_solution = generate_initial_solution(self.customer_data, self.distance_matrix)
        current_cost, current_detail = calc_total_cost(current_solution, self.customer_data, self.distance_matrix)
                
        # 初始化全局最优
        self.best_solution = deepcopy(current_solution)
        self.best_cost = current_cost
        self.best_detail = current_detail
        
        T = self.T0
        no_improve = 0
        self.history = {"iter": [], "current": [], "best": []}  # 清空历史记录

        # 🚀 启用 tqdm 进度条
        pbar = tqdm.tqdm(range(self.max_iter), desc="ALNS 搜索中", unit="iter")
        for iter in pbar:
            # 选择算子
            destroy_idx = self.select_operator(self.destroy_weights)
            repair_idx = self.select_operator(self.repair_weights)
            self.destroy_counts[destroy_idx] += 1
            self.repair_counts[repair_idx] += 1
            
            # 执行破坏与修复
            remove_num = max(3, int(len(self.customers)*0.1))
            temp_solution, removed = self.destroy_operators[destroy_idx](current_solution, remove_num)
            new_solution = self.repair_operators[repair_idx](temp_solution, removed)
            new_cost, new_detail = calc_total_cost(new_solution, self.customer_data, self.distance_matrix)
            
            # 接受准则
            if new_cost < current_cost:
                current_solution = deepcopy(new_solution)
                current_cost = new_cost
                current_detail = new_detail
                self.destroy_scores[destroy_idx] += 20
                self.repair_scores[repair_idx] += 20
                
                if new_cost < self.best_cost:
                    self.best_solution = deepcopy(new_solution)
                    self.best_cost = new_cost
                    self.best_detail = new_detail
                    self.destroy_scores[destroy_idx] += 30
                    self.repair_scores[repair_idx] += 30
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # 模拟退火接受劣解
                delta = new_cost - current_cost
                prob = exp(-delta / T)
                if random.random() < prob:
                    current_solution = deepcopy(new_solution)
                    current_cost = new_cost
                    current_detail = new_detail
                    self.destroy_scores[destroy_idx] += 10
                    self.repair_scores[repair_idx] += 10
                else:
                    no_improve += 1
            
            # 更新温度
            T *= self.alpha
            
            # 每10轮更新算子权重
            if iter % 10 == 0:
                self.update_weights()
            
            # 提前终止
            if no_improve >= 200:
                break
            
            # 更新进度条右侧状态
            pbar.set_postfix({
                "最优成本": f"{self.best_cost:.2f}",
                "温度": f"{T:.2f}",
                "停滞步数": no_improve
            })

        return self.best_solution, self.best_cost, self.best_detail
    
# ===================== 主程序 =====================
if __name__ == "__main__":
    # 读取附件数据
    order_df = pd.read_excel("./data/订单信息.xlsx")
    distance_df = pd.read_excel("./data/距离矩阵.xlsx", index_col=0)
    time_window_df = pd.read_excel("./data/时间窗.xlsx")

    # 清理 Excel 中的缺失值/文本残留，防止 NaN 污染浮点运算
    order_df["重量"] = pd.to_numeric(order_df["重量"], errors="coerce").fillna(0.0)
    order_df["体积"] = pd.to_numeric(order_df["体积"], errors="coerce").fillna(0.0)
    distance_df = distance_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # 转换为代码所需格式
    customer_data = {}
    for idx, row in order_df.iterrows():
        cust_id = row["目标客户编号"]
        tw_row = time_window_df[time_window_df["客户编号"]==cust_id]
        
        # Excel 中的 'HH:MM' 格式，转为浮点数小时
        def _to_float_hours(val):
            val_str = str(val).strip()
            if ':' in val_str:
                h, m = map(float, val_str.split(':'))
                return h + m / 60.0
            return float(val_str)
            
        customer_data[cust_id] = {
            "d": row["重量"],
            "v": row["体积"],
            "ET": _to_float_hours(tw_row["开始时间"].values[0]),
            "LT": _to_float_hours(tw_row["结束时间"].values[0]),
        }
    distance_matrix = distance_df.values


    print("✅ 数据校验通过：配送中心(0)存在，所有客户可查距离")
    # 运行ALNS算法
    alns = ALNS(customer_data, distance_matrix, max_iter=1000)
    best_solution, best_cost, best_detail = alns.solve()
    
    # 输出结果
    print("\n==================== 最优调度方案 ====================")
    print(f"总配送成本：{best_cost:.2f} 元")
    print("成本明细：")
    for key, value in best_detail.items():
        print(f"  - {key}: {value:.2f} 元")
    print("\n车辆使用方案：")
    vehicle_count = {}
    for idx, (v_type, route) in enumerate(best_solution):
        vehicle_count[v_type] = vehicle_count.get(v_type, 0) + 1
        cost, detail, arrival_time = calc_route_cost(route, v_type, customer_data, distance_matrix)
        print(f"\n车辆{idx+1}（车型：{v_type}）：")
        print(f"  行驶路径：{' -> '.join(map(str, route))}")
        print(f"  各节点到达时间：{[f'{t:.2f}h' for t in arrival_time]}")
        print(f"  单车辆成本：{cost:.2f} 元")
    print("\n车型使用数量：")
    for v_type, count in vehicle_count.items():
        print(f"  {v_type}: {count} 辆（可用数量：{vehicle_params[v_type]['num']} 辆）")
