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

# 定义多充电站环境
class ChargingEnvPSO:
    def __init__(self, num_vehicles, num_stations, station_piles, start_times, vehicle_positions, 
                 station_positions, vehicle_battery_capacities, expected_charging_durations, assigned_stations):
        self.num_vehicles = num_vehicles
        self.num_stations = num_stations
        self.station_piles = station_piles
        self.start_times = start_times
        self.vehicle_positions = vehicle_positions
        self.station_positions = station_positions
        self.vehicle_battery_capacities = vehicle_battery_capacities
        self.expected_charging_durations = expected_charging_durations
        self.assigned_stations = assigned_stations
        self.charging_rates = [200, 350, 150, 360, 100]  # 預定義充電速率（kW）
        self.max_wait_time = 60  # 最大等待時間上限（分鐘）
        self.reset()

    def reset(self):
        self.charging_durations = np.array([
            min(
                self.vehicle_battery_capacities[i] / self.charging_rates[self.assigned_stations[i]] * 60,
                self.expected_charging_durations[i]
            ) for i in range(self.num_vehicles)
        ])
        self.df = pd.DataFrame({
            'Vehicle': [f"Vehicle {i+1}" for i in range(self.num_vehicles)],
            'Start Time': self.start_times.astype(np.float64),
            'Charging Duration': self.charging_durations.astype(np.float64),
            'Station': pd.Series(-1, index=range(self.num_vehicles), dtype=np.int64),
            'Pile': pd.Series(-1, index=range(self.num_vehicles), dtype=np.int64),
            'Wait Time': pd.Series(0.0, index=range(self.num_vehicles), dtype=np.float64),
            'End Time': pd.Series(0.0, index=range(self.num_vehicles), dtype=np.float64)
        })
        self.station_schedules = [[] for _ in range(self.num_stations)]
        for i, piles in enumerate(self.station_piles):
            self.station_schedules[i] = [[] for _ in range(piles)]

    def calculate_distance(self, vehicle_idx, station_idx):
        return np.linalg.norm(np.array(self.vehicle_positions[vehicle_idx]) - np.array(self.station_positions[station_idx]))

    def is_conflict(self, start_time, end_time, schedule):
        for _, s, e in schedule:
            if not (end_time <= s or start_time >= e):
                return True
        return False

    def objective_function(self, positions):
        total_cost = 0
        self.reset()

        for vehicle_idx, station_idx in enumerate(positions):
            if station_idx >= self.num_stations:
                self.df.at[vehicle_idx, 'Wait Time'] = self.max_wait_time
                total_cost += self.max_wait_time * 2.0
                continue

            station_idx = int(station_idx)
            best_pile, min_finish_time = -1, float('inf')

            distance = self.calculate_distance(vehicle_idx, station_idx)
            traffic_time = distance / 40
            arrival_time = self.start_times[vehicle_idx] + traffic_time

            for pile_idx in range(len(self.station_schedules[station_idx])):
                schedule = self.station_schedules[station_idx][pile_idx]
                last_end_time = max((e for _, _, e in schedule), default=0)
                available_time = max(arrival_time, last_end_time)
                finish_time = available_time + self.charging_durations[vehicle_idx]

                if finish_time < min_finish_time and not self.is_conflict(available_time, finish_time, schedule):
                    min_finish_time = finish_time
                    best_pile = pile_idx

            if best_pile != -1:
                last_end_time = max((e for _, _, e in self.station_schedules[station_idx][best_pile]), default=0)
                start_time = max(arrival_time, last_end_time)
                wait_time = min(max(0, start_time - arrival_time), self.max_wait_time)
                finish_time = start_time + self.charging_durations[vehicle_idx]

                self.df.at[vehicle_idx, 'Station'] = station_idx
                self.df.at[vehicle_idx, 'Pile'] = best_pile
                self.df.at[vehicle_idx, 'Wait Time'] = wait_time
                self.df.at[vehicle_idx, 'End Time'] = finish_time
                self.station_schedules[station_idx][best_pile].append((vehicle_idx, start_time, finish_time))

                alpha, beta = 2.0, 0.3
                total_cost += alpha * wait_time + beta * traffic_time
            else:
                self.df.at[vehicle_idx, 'Wait Time'] = self.max_wait_time
                total_cost += self.max_wait_time * 2.0

        station_idle_times, total_idle_time = self.calculate_idle_time()
        gamma = 0.1
        total_cost += gamma * total_idle_time * 0.1

        return total_cost

    def pso(self, num_particles, max_iter, seed):
        set_seed(seed)
        particles = np.random.uniform(0, self.num_stations, size=(num_particles, self.num_vehicles))
        velocities = np.zeros_like(particles)
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([self.objective_function(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        w, c1, c2 = 0.5, 2, 2

        for iteration in range(max_iter):
            print(f"\n訓練回合 {iteration + 1}/{max_iter} 開始...")
            for i in range(num_particles):
                velocities[i] = (
                    w * velocities[i] +
                    c1 * random.random() * (personal_best_positions[i] - particles[i]) +
                    c2 * random.random() * (global_best_position - particles[i])
                )
                particles[i] = np.clip(particles[i] + velocities[i], 0, self.num_stations - 1)

                score = self.objective_function(particles[i])
                if score < personal_best_scores[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_scores[i] = score

                    if score < global_best_score:
                        global_best_position = particles[i]
                        global_best_score = score

            print(f"訓練回合 {iteration + 1}/{max_iter} 完成，最佳分數: {global_best_score:.2f}")

        return global_best_position

    def calculate_idle_time(self):
        total_idle_time = 0
        station_idle_times = []

        for station in self.station_schedules:
            station_idle_time = 0
            for schedule in station:
                if not schedule:
                    station_idle_time += 1440
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
                if current_time < 1440:
                    idle_time += 1440 - current_time
                station_idle_time += idle_time
            station_idle_times.append(station_idle_time)
            total_idle_time += station_idle_time

        return station_idle_times, total_idle_time

    def calculate_average_wait_time(self):
        """計算所有車輛的平均等待時間"""
        return self.df['Wait Time'].mean()

    def display_vehicle_schedule(self):
        print("\n車輛排程信息:")
        for vehicle_idx, row in self.df.iterrows():
            station = row['Station']
            if station == -1:
                print(f"  Vehicle {vehicle_idx + 1}: 未排程")
            else:
                dist = self.calculate_distance(vehicle_idx, station)
                print(f"  Vehicle {vehicle_idx + 1}: Station {station + 1}, Pile {row['Pile'] + 1}, "
                      f"Start Time {row['Start Time']:.2f}, End Time {row['End Time']:.2f}, "
                      f"Wait Time {row['Wait Time']:.2f}, "
                      f"Vehicle Position {self.vehicle_positions[vehicle_idx]}, Station Position {self.station_positions[station]}, "
                      f"Distance {dist:.2f}")

    def display_station_idle_times(self):
        station_idle_times, total_idle_time = self.calculate_idle_time()
        print("\n每個充電站所有充電樁的總閒置時間:")
        for station_idx, idle_time in enumerate(station_idle_times):
            print(f"  充電站 {station_idx + 1} 總閒置時間: {idle_time:.2f} 分鐘")
        print(f"\n所有充電站的總閒置時間為: {total_idle_time:.2f} 分鐘")

    def display_total_wait_time(self):
        total_wait_time = self.df['Wait Time'].sum()
        print(f"\n所有車輛的總等待時間為: {total_wait_time:.2f} 分鐘")

    def plot_station_schedules(self):
        for station_idx, station in enumerate(self.station_schedules):
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.set_title(f"Charging Schedule for Station {station_idx + 1}")
            ax.set_xlabel("Time (minutes)")
            ax.set_ylabel("Pile")
            colors = plt.cm.tab10.colors

            for pile_idx, schedule in enumerate(station):
                for vehicle_idx, start, end in schedule:
                    ax.barh(pile_idx, end - start, left=start, height=0.4,
                            color=colors[vehicle_idx % len(colors)], edgecolor="black",
                            label=f"Vehicle {vehicle_idx + 1}" if f"Vehicle {vehicle_idx + 1}" not in [patch.get_label() for patch in ax.patches] else "")

            ax.set_xlim(0, 1440)
            ax.set_ylim(-0.5, len(station) + 0.5)
            ax.set_yticks(range(len(station)))
            ax.set_yticklabels([f"Pile {i + 1}" for i in range(len(station))])
            ax.grid(axis="x", linestyle="--", alpha=0.7)

            handles, labels = ax.get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles))
            ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper right", title="Vehicles", fontsize="small")

            plt.tight_layout()
            plt.show()

# 主程序
if __name__ == "__main__":
    seed = 42
    set_seed(seed)

    num_vehicles = 1000
    num_stations = 5
    station_piles = [5, 2, 4, 3, 4]
    start_times = np.linspace(0, 1440, num_vehicles)
    vehicle_positions = np.random.randint(0, 100, size=(num_vehicles, 2))
    station_positions = np.random.randint(0, 100, size=(num_stations, 2))
    vehicle_battery_capacities = np.random.uniform(28, 95, size=num_vehicles)
    expected_charging_durations = np.random.uniform(30, 60, size=num_vehicles)
    assigned_stations = np.random.choice(num_stations, num_vehicles)

    env = ChargingEnvPSO(num_vehicles, num_stations, station_piles, start_times, vehicle_positions, 
                        station_positions, vehicle_battery_capacities, expected_charging_durations, assigned_stations)
    
    best_positions = env.pso(num_particles=300, max_iter=200, seed=seed)
    final_cost = env.objective_function(best_positions)
    
    env.display_vehicle_schedule()
    env.display_total_wait_time()
    average_wait_time = env.calculate_average_wait_time()
    print(f"平均等待時間: {average_wait_time:.2f} 分鐘")
    env.display_station_idle_times()
    env.plot_station_schedules()
    print(f"\n最終成本: {final_cost:.2f}")