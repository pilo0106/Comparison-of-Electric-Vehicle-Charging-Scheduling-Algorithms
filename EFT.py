import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# 設定隨機種子
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ChargingEnvEFTWithLimit:
    def __init__(self, num_vehicles, num_stations, station_piles, start_times, 
                 battery_capacities, charging_rates, vehicle_positions, station_positions):
        self.num_vehicles = num_vehicles
        self.num_stations = num_stations
        self.station_piles = station_piles
        self.start_times = start_times
        self.battery_capacities = battery_capacities
        self.charging_rates = charging_rates
        self.vehicle_positions = vehicle_positions
        self.station_positions = station_positions
        # 計算原始充電時間
        self.calculated_charging_durations = np.ceil(
            (self.battery_capacities / np.array([random.choice(charging_rates) 
                                                for _ in range(num_vehicles)])) * 60)
        # 設定期望充電時間上限
        self.expected_charging_durations = np.random.uniform(30, 60, size=num_vehicles)
        # 實際充電時間取 min(計算值, 期望上限)
        self.charging_durations = np.minimum(self.calculated_charging_durations, 
                                           self.expected_charging_durations)
        self.reset()

    def reset(self):
        self.df = pd.DataFrame({
            'Vehicle': [f"Vehicle {i+1}" for i in range(self.num_vehicles)],
            'Start Time': self.start_times,
            'Charging Duration': self.charging_durations,
            'Expected Duration': self.expected_charging_durations,  # 新增期望時間欄位
            'Calculated Duration': self.calculated_charging_durations,  # 新增計算時間欄位
            'Station': -1,
            'Pile': -1,
            'Wait Time': 0,
            'End Time': 0,
            'Arrival Time': 0,
            'Interrupted': False  # 新增是否被中斷的標記
        })
        self.station_schedules = [[] for _ in range(self.num_stations)]
        for i, piles in enumerate(self.station_piles):
            self.station_schedules[i] = [[] for _ in range(piles)]

    def calculate_distance(self, vehicle_idx, station_idx):
        return np.linalg.norm(np.array(self.vehicle_positions[vehicle_idx]) - 
                            np.array(self.station_positions[station_idx]))

    def find_eft(self, vehicle_idx):
        min_finish_time = float('inf')
        best_station, best_pile = -1, -1
        best_start_time, best_finish_time = None, None
        best_arrival_time = None
        
        vehicle_start = self.df.loc[vehicle_idx, 'Start Time']
        charging_duration = self.df.loc[vehicle_idx, 'Charging Duration']

        for station_idx in range(self.num_stations):
            distance = self.calculate_distance(vehicle_idx, station_idx)
            arrival_time = vehicle_start + distance / (40 / 60)  # 速度 40 km/h
            for pile_idx in range(len(self.station_schedules[station_idx])):
                schedule = self.station_schedules[station_idx][pile_idx]
                last_end_time = max((e for _, _, e in schedule), default=0)
                start_time = max(arrival_time, last_end_time)
                finish_time = start_time + charging_duration
                
                if finish_time < min_finish_time:
                    min_finish_time = finish_time
                    best_station, best_pile = station_idx, pile_idx
                    best_start_time, best_finish_time = start_time, finish_time
                    best_arrival_time = arrival_time

        if best_station == -1:
            return -1, -1, None, None, None
        return best_station, best_pile, best_start_time, best_finish_time, best_arrival_time

    def schedule_vehicles(self):
        vehicle_order = self.df['Start Time'].argsort()
        unassigned_count = 0
        interrupted_count = 0
        for vehicle_idx in vehicle_order:
            station_idx, pile_idx, start_time, finish_time, arrival_time = self.find_eft(vehicle_idx)
            if station_idx != -1 and start_time is not None:
                wait_time = start_time - arrival_time
                # 檢查是否超過期望時間
                calculated_duration = self.df.at[vehicle_idx, 'Calculated Duration']
                expected_duration = self.df.at[vehicle_idx, 'Expected Duration']
                if calculated_duration > expected_duration:
                    self.df.at[vehicle_idx, 'Interrupted'] = True
                    interrupted_count += 1
                self.df.at[vehicle_idx, 'Station'] = station_idx
                self.df.at[vehicle_idx, 'Pile'] = pile_idx
                self.df.at[vehicle_idx, 'Wait Time'] = wait_time
                self.df.at[vehicle_idx, 'End Time'] = finish_time
                self.df.at[vehicle_idx, 'Arrival Time'] = arrival_time
                self.station_schedules[station_idx][pile_idx].append((vehicle_idx, start_time, finish_time))
                if vehicle_idx < 10:  # 只打印前10輛車的數據
                    print(f"Vehicle {vehicle_idx + 1}: Arrival {arrival_time:.2f}, "
                          f"Start {start_time:.2f}, Finish {finish_time:.2f}, "
                          f"Interrupted: {self.df.at[vehicle_idx, 'Interrupted']}")
            else:
                self.df.at[vehicle_idx, 'Wait Time'] = 1440
                unassigned_count += 1
        print(f"未分配車輛數量: {unassigned_count}")
        print(f"因超過期望時間而中斷的車輛數量: {interrupted_count}")

    def display_vehicle_schedule(self):
        print("\n車輛排程信息:")
        for vehicle_idx, row in self.df.iterrows():
            station = row['Station']
            if station == -1:
                print(f"  Vehicle {vehicle_idx + 1}: 未排程")
            else:
                dist = self.calculate_distance(vehicle_idx, station)
                interrupted = " (中斷)" if row['Interrupted'] else ""
                print(f"  Vehicle {vehicle_idx + 1}: Station {station + 1}, Pile {row['Pile'] + 1}, "
                      f"Start Time {row['Start Time']:.2f} min, End Time {row['End Time']:.2f} min, "
                      f"Wait Time {row['Wait Time']:.2f} min, Distance {dist:.2f} km, "
                      f"Expected Duration {row['Expected Duration']:.2f}{interrupted}")

    def calculate_idle_times(self):
        total_idle_time = 0
        station_idle_times = []
        for station in self.station_schedules:
            station_idle_time = 0
            for schedule in station:
                if not schedule:
                    continue
                time_segments = sorted([(start, end) for _, start, end in schedule])
                merged_segments = []
                for segment in time_segments:
                    if not merged_segments or segment[0] > merged_segments[-1][1]:
                        merged_segments.append(segment)
                    else:
                        merged_segments[-1] = (
                            merged_segments[-1][0],
                            max(merged_segments[-1][1], segment[1]),
                        )
                idle_time = 0
                current_time = 0
                for start, end in merged_segments:
                    if start > current_time:
                        idle_time += start - current_time
                    current_time = end
                station_idle_time += idle_time
            station_idle_times.append(station_idle_time)
            total_idle_time += station_idle_time
        print("\n每個充電站的總閒置時間:")
        for idx, idle_time in enumerate(station_idle_times):
            print(f"  充電站 {idx + 1} 總閒置時間: {idle_time:.2f} 分鐘")
        print(f"\n所有充電站的總閒置時間為: {total_idle_time:.2f} 分鐘")

    def analyze_station_load(self):
        print("\n充電站負載分析:")
        for station_idx, station in enumerate(self.station_schedules):
            total_busy_time = sum(end - start for pile in station for _, start, end in pile)
            num_piles = len(station)
            utilization = total_busy_time / (1440 * num_piles) if num_piles > 0 else 0
            queue_length = sum(len(pile) for pile in station)
            print(f"  Station {station_idx + 1}: 佔用率 {utilization:.2%}, 總排隊數量 {queue_length}")

    def plot_station_schedules(self):
        for station_idx, station in enumerate(self.station_schedules):
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.set_title(f"Charging Schedule for Station {station_idx + 1}")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Pile")
            colors = plt.cm.tab10.colors
            for pile_idx, schedule in enumerate(station):
                for vehicle_idx, start, end in schedule:
                    color = 'red' if self.df.at[vehicle_idx, 'Interrupted'] else colors[vehicle_idx % len(colors)]
                    ax.barh(
                        pile_idx, end - start, left=start, height=0.4,
                        color=color, edgecolor="black"
                    )
            ax.set_xlim(0, 1440)
            ax.set_ylim(-0.5, len(station) + 0.5)
            ax.set_yticks(range(len(station)))
            ax.set_yticklabels([f"Pile {i + 1}" for i in range(len(station))])
            ax.grid(axis="x", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.show()

# 主程序
if __name__ == "__main__":
    seed = 42
    set_seed(seed)
    num_vehicles = 1000
    num_stations = 5
    station_piles = [5, 2, 4, 3, 4]
    start_times = np.random.randint(0, 1440, size=num_vehicles)
    battery_capacities = np.random.randint(28, 96, size=num_vehicles)
    charging_rates = [200, 350, 150, 360, 100]
    vehicle_positions = np.random.randint(0, 100, size=(num_vehicles, 2))
    station_positions = np.random.randint(0, 100, size=(num_stations, 2))

    env = ChargingEnvEFTWithLimit(num_vehicles, num_stations, station_piles, start_times, 
                                 battery_capacities, charging_rates, vehicle_positions, station_positions)
    env.schedule_vehicles()
    env.display_vehicle_schedule()
    assigned_vehicles = env.df[env.df['Wait Time'] != 1440]
    total_wait_time = assigned_vehicles['Wait Time'].sum()
    average_wait_time = total_wait_time / len(assigned_vehicles) if len(assigned_vehicles) > 0 else 0
    print(f"\n所有車輛的總等待時間為: {total_wait_time:.2f} 分鐘")
    print(f"所有車輛的平均等待時間為: {average_wait_time:.2f} 分鐘")
    print(f"最大等待時間為: {assigned_vehicles['Wait Time'].max():.2f} 分鐘")
    print(f"等待時間超過60分鐘的車輛數: {(assigned_vehicles['Wait Time'] > 60).sum()}")
    print(f"已分配車輛數量: {len(assigned_vehicles)}")
    print(f"等待時間最小值: {assigned_vehicles['Wait Time'].min():.2f} 分鐘")
    print(f"等待時間中位數: {assigned_vehicles['Wait Time'].median():.2f} 分鐘")
    env.calculate_idle_times()
    env.analyze_station_load()
    env.plot_station_schedules()