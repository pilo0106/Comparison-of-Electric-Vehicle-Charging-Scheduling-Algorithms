import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple

# 定义多充电站环境
class ChargingEnv:
    def __init__(self, num_vehicles, num_stations, station_piles, start_times, vehicle_positions, station_positions, vehicle_battery_capacities, expected_charging_durations, assigned_stations):
        self.num_vehicles = num_vehicles
        self.num_stations = num_stations
        self.station_piles = station_piles
        self.start_times = start_times
        self.vehicle_positions = vehicle_positions
        self.station_positions = station_positions
        self.charging_rates = [200, 350, 150, 360, 100]  # 预定义充电速率（kW）
        self.unallocated_vehicles = []
        self.vehicle_battery_capacities = vehicle_battery_capacities
        self.expected_charging_durations = expected_charging_durations
        self.assigned_stations = assigned_stations
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
            'Vehicle': [f"Vehicle {i}" for i in range(self.num_vehicles)],
            'Start Time': self.start_times,
            'Charging Duration': self.charging_durations,
            'Station': -1,
            'Pile': -1,
            'Wait Time': 0,
            'End Time': self.start_times + self.charging_durations
        })
        self.station_schedules = [
            [[] for _ in range(piles)] for piles in self.station_piles
        ]
        self.unallocated_vehicles = []
        self.current_vehicle = 0
        return self.get_state()

    def calculate_distance(self, vehicle_idx, station_idx):
        vehicle_pos = self.vehicle_positions[vehicle_idx]
        station_pos = self.station_positions[station_idx]
        return np.linalg.norm(np.array(vehicle_pos) - np.array(station_pos))

    def get_next_available_time(self, schedule):
        return max((end for _, _, end in schedule), default=0)

    def get_state(self):
        vehicle = self.df.iloc[self.current_vehicle]
        distances = [self.calculate_distance(self.current_vehicle, i) for i in range(self.num_stations)]
        next_available_times = [
            self.get_next_available_time(schedule) 
            for station in self.station_schedules 
            for schedule in station
        ]
        return np.array(
            [vehicle['Start Time'], vehicle['Charging Duration']] + distances + next_available_times,
            dtype=np.float32
        )

    def is_conflict(self, start_time, end_time, schedule):
        for _, s, e in schedule:
            if not (end_time <= s or start_time >= e):
                return True
        return False

    def get_valid_actions(self):
        valid_actions = []
        vehicle = self.df.iloc[self.current_vehicle]
        start_time = vehicle['Start Time']
        duration = vehicle['Charging Duration']
        action_size = sum(self.station_piles)  # 確保與 DQN 的動作空間一致

        pile_offset = 0  # 用於計算每個充電站的動作索引偏移
        for station_idx in range(self.num_stations):
            for pile_idx in range(self.station_piles[station_idx]):
                schedule = self.station_schedules[station_idx][pile_idx]
                distance = self.calculate_distance(self.current_vehicle, station_idx)
                traffic_time = distance / 40
                arrival_time = start_time + traffic_time
                available_time = max(arrival_time, self.get_next_available_time(schedule))
                end_time = available_time + duration

                if not self.is_conflict(available_time, end_time, schedule):
                    action_idx = pile_offset + pile_idx  # 基於偏移量計算動作索引
                    valid_actions.append(action_idx)

            pile_offset += self.station_piles[station_idx]  # 更新偏移量

        if not valid_actions:
            min_load_station = min(
                range(self.num_stations), 
                key=lambda s: sum(len(p) for p in self.station_schedules[s])
            )
            pile_idx = min(
                range(self.station_piles[min_load_station]), 
                key=lambda p: len(self.station_schedules[min_load_station][p])
            )
            pile_offset = sum(self.station_piles[:min_load_station])  # 計算該站的偏移量
            action_idx = pile_offset + pile_idx
            valid_actions.append(action_idx)

        return valid_actions

    def step(self, action):
        vehicle = self.df.iloc[self.current_vehicle]
        
        # 根據 action 解析 station_idx 和 pile_idx
        pile_offset = 0
        station_idx = 0
        for i, piles in enumerate(self.station_piles):
            if action < pile_offset + piles:
                station_idx = i
                pile_idx = action - pile_offset
                break
            pile_offset += piles

        start_time = vehicle['Start Time']
        duration = vehicle['Charging Duration']
        distance = self.calculate_distance(self.current_vehicle, station_idx)
        traffic_time = distance / 40
        arrival_time = start_time + traffic_time

        schedule = self.station_schedules[station_idx][pile_idx]
        available_time = max(arrival_time, self.get_next_available_time(schedule))
        total_wait_time = min(max(0, available_time - arrival_time), self.max_wait_time)

        if not self.is_conflict(available_time, available_time + duration, schedule):
            self.df.at[self.current_vehicle, 'Station'] = station_idx
            self.df.at[self.current_vehicle, 'Pile'] = pile_idx
            self.df.at[self.current_vehicle, 'Wait Time'] = total_wait_time
            self.df.at[self.current_vehicle, 'Start Time'] = available_time
            self.df.at[self.current_vehicle, 'End Time'] = available_time + duration
            self.station_schedules[station_idx][pile_idx].append((self.current_vehicle, available_time, available_time + duration))

            alpha = 2.0
            beta = 0.3
            gamma = 0.1
            station_idle_times, _ = self.calculate_idle_time()
            idle_time_penalty = sum(station_idle_times) * 0.1
            reward = -(alpha * total_wait_time + beta * traffic_time) - gamma * idle_time_penalty
             
            if total_wait_time + traffic_time < 3:
                reward += 1000
            elif total_wait_time + traffic_time < 5:
                reward += 500
            elif total_wait_time + traffic_time < 10:
                reward += 100
            elif total_wait_time + traffic_time < 20:
                reward += 50
            elif total_wait_time + traffic_time < 40:
                reward += 10
        else:
            reward = -100
            self.unallocated_vehicles.append((self.current_vehicle, "No Available Pile"))

        self.current_vehicle += 1
        done = self.current_vehicle >= self.num_vehicles
        next_state = self.get_state() if not done else None
        return next_state, reward, done

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

# DQN 模型
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        self.fc4 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.ln3(self.fc3(x)))
        return self.fc4(x)

# 经验回放
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

    def push(self, *args):
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 訓練 DQN (Double DQN)
def train_dqn(env, num_episodes=1000, target_update=10):
    state_size = 2 + env.num_stations + sum(env.station_piles)
    action_size = sum(env.station_piles)
    policy_net = DQN(state_size, action_size).to(device)
    target_net = DQN(state_size, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)  # 降低學習率
    criterion = nn.MSELoss()
    memory = ReplayMemory(2000)
    gamma = 0.99
    epsilon = 0.7
    epsilon_decay = 0.995  # 加快衰減
    epsilon_min = 0.01
    batch_size = 32

    rewards_per_episode = []
    wait_times_per_episode = []
    idle_times_per_episode = []

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        state = torch.FloatTensor(env.reset()).to(device)
        total_reward = 0

        for t in range(env.num_vehicles):
            valid_actions = env.get_valid_actions()
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q_values = policy_net(state.unsqueeze(0))
                    valid_q_values = torch.tensor(
                        [q_values.squeeze(0)[action].item() for action in valid_actions],
                        device=device
                    )
                    action = valid_actions[torch.argmax(valid_q_values).item()]

            next_state, reward, done = env.step(action)
            total_reward += reward
            next_state = torch.FloatTensor(next_state).to(device) if next_state is not None else None
            memory.push(state, action, reward, next_state, done)
            state = next_state

            if len(memory) > batch_size:
                minibatch = memory.sample(batch_size)
                batch = memory.transition(*zip(*minibatch))
                state_batch = torch.stack(batch.state)
                action_batch = torch.tensor(batch.action, dtype=torch.long).to(device)
                reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
                non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

                # Double DQN
                next_state_actions = policy_net(non_final_next_states).argmax(1, keepdim=True)
                next_state_values = torch.zeros(batch_size, device=device)
                next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_state_actions).squeeze()
                expected_q_values = reward_batch + (gamma * next_state_values)
                q_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
                loss = criterion(q_values, expected_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        rewards_per_episode.append(total_reward)
        wait_times_per_episode.append(env.df["Wait Time"].mean())
        _, total_idle_time = env.calculate_idle_time()
        idle_times_per_episode.append(total_idle_time)

        if episode % 100 == 0:  # 更頻繁打印進度
            print(f"Episode {episode}/{num_episodes}, Reward: {total_reward:.2f}, "
                  f"Avg Wait Time: {wait_times_per_episode[-1]:.2f}, Total Idle Time: {total_idle_time:.2f}")

    return policy_net, rewards_per_episode, wait_times_per_episode, idle_times_per_episode

# 显示所有车辆的排程信息
def display_vehicle_schedule(env):
    print("\n车辆排程信息:")
    for vehicle_idx, row in env.df.iterrows():
        if row['Station'] == -1:
            print(f"  Vehicle {vehicle_idx + 1}: 未排程")
        else:
            vehicle_pos = env.vehicle_positions[vehicle_idx]
            station_idx = row['Station']
            station_pos = env.station_positions[station_idx]
            distance = np.linalg.norm(np.array(vehicle_pos) - np.array(station_pos))
            print(f"  Vehicle {vehicle_idx + 1}: Station {station_idx + 1}, Pile {row['Pile'] + 1}, "
                  f"Start Time {row['Start Time']:.2f}, End Time {row['End Time']:.2f}, "
                  f"Vehicle Position {vehicle_pos}, Station Position {station_pos}, Distance {distance:.2f}")

# 显示未排程车辆
def display_unallocated_vehicles(env):
    if len(env.unallocated_vehicles) > 0:
        print("\n未排程車輛:")
        for vehicle_idx, reason in env.unallocated_vehicles:
            print(f"  Vehicle {vehicle_idx + 1}: {reason}")
    else:
        print("\n所有車輛均已成功排程")

# 绘制甘特图
def plot_station_schedules(env):
    for station_idx, station in enumerate(env.station_schedules):
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.set_title(f"Charging Schedule for Station {station_idx + 1}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Pile")
        colors = plt.cm.tab10.colors
        for pile_idx, schedule in enumerate(station):
            for vehicle_idx, start, end in schedule:
                ax.barh(
                    pile_idx, end - start, left=start, height=0.4,
                    color=colors[vehicle_idx % len(colors)], edgecolor="black",
                    label=f"Vehicle {vehicle_idx + 1}" if f"Vehicle {vehicle_idx + 1}" not in [patch.get_label() for patch in ax.patches] else ""
                )
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

def calculate_average_wait_time(env):
    return env.df['Wait Time'].mean()

def display_station_idle_times(env):
    station_idle_times, total_idle_time = env.calculate_idle_time()
    print("\n每個充電站所有充電樁的總閒置時間:")
    for station_idx, idle_time in enumerate(station_idle_times):
        print(f"  充電站 {station_idx + 1} 總閒置時間: {idle_time:.2f} 分鐘")
    print(f"\n所有充電站的總閒置時間為: {total_idle_time:.2f} 分鐘")

def display_total_wait_time(env):
    total_wait_time = env.df['Wait Time'].sum()
    print(f"\n所有車輛的總等待時間為: {total_wait_time:.2f} 分鐘")

# 設定種子碼的函數
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    set_seed(seed)

    # 初始化參數
    num_vehicles = 1000
    num_stations = 5
    station_piles = [5, 2, 4, 3, 4]
    # 分散到達時間，避免過於集中
    start_times = np.linspace(0, 1440, num_vehicles)
    vehicle_positions = np.random.randint(0, 100, size=(num_vehicles, 2))
    station_positions = np.random.randint(0, 100, size=(num_stations, 2))
    vehicle_battery_capacities = np.random.uniform(50, 95, size=num_vehicles)
    expected_charging_durations = np.random.uniform(30, 60, size=num_vehicles)
    assigned_stations = np.random.choice(num_stations, num_vehicles)

    env = ChargingEnv(num_vehicles, num_stations, station_piles, start_times, vehicle_positions, station_positions, vehicle_battery_capacities, expected_charging_durations, assigned_stations)

    policy_net, rewards, wait_times, idle_times = train_dqn(env, num_episodes=1000)

    display_vehicle_schedule(env)
    display_unallocated_vehicles(env)
    average_wait_time = calculate_average_wait_time(env)
    print(f"平均等待時間: {average_wait_time:.2f} 分鐘")
    display_total_wait_time(env)
    display_station_idle_times(env)

    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    axs[0].plot(rewards, label="Total Reward", color='b')
    axs[0].set_title("Total Reward per Episode")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Reward")
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(wait_times, label="Average Wait Time", color='r')
    axs[1].set_title("Average Wait Time per Episode")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Wait Time")
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(idle_times, label="Total Idle Time", color='g')
    axs[2].set_title("Total Idle Time per Episode")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Idle Time")
    axs[2].legend()
    axs[2].grid()

    plt.tight_layout()
    plt.show()

    plot_station_schedules(env)