import numpy as np  # numpy : 데이터 처리 및 수학적 연산
import random  # random : 데이터 샘플링이나 에이전트의 행동을 무작위로 선택
import tensorflow as tf  # tensorflow : 신경망 모델을 구축하고 학습
from tensorflow.keras import layers, models  # keras :  layers와 models를 임포트하여 신경망의 층(layer)과 모델을 정의
import matplotlib.pyplot as plt  # Matplotlib : 데이터 시각화를 위한 라이브러리
import datetime  # 파이썬에서 날짜와 시간 관련 기능을 제공.


# 환경 설정
class HVACEnvironment:  # HVAC(Heating, Ventilation, and Air Conditioning) 클래스 정의
    def __init__(self, initial_temperature=22, temperature_increase_rate=0.01):  # 초기온도 22도, 온도 상승 속도 0.01 설정
        self.temperature = initial_temperature  # 현재 온도 저장
        self.target_temperature = 20  # 목표온도 20도로 설정
        self.cost_per_degree = 1  # 온도 1도 조절하는 데 드는 비용
        self.season = 'summer'  # 계절 기본값 설정
        self.temperature_increase_rate = temperature_increase_rate  # 온도 상승 속도 저장
        self.energy_usage = 0  # 에너지 사용량 초기화
        self.user_preference = random.uniform(18, 24)  # 18도에서 24도 사이의 랜덤한 값을 생성하여 사용자 선호 온도로 설정

    def reset(self):  # reset : 객체의 상태를 초기화
        self.temperature = 22  # 현재 온도를 22도로 초기화
        self.season = random.choice(['summer', 'winter'])  # 여름 또는 겨울 중 하나를 랜덤으로 선택하여 계절을 설정
        self.energy_usage = 0  # 에너지 사용량 초기화
        return self.temperature  # 현재 온도(22도)를 반환

    def step(self, action):  # step : HVAC 시스템의 상태를 업데이트하고 특정 행동(action)에 따라 환경을 변화시키는 역할
        # 계절에 따른 외부 온도 변화
        if self.season == 'summer':
            external_temperature = np.random.normal(loc=30, scale=5)  # 여름일 경우, 외부 온도는 평균 30도, 표준편차 5도의 정규 분포를 따르는 난수로 설정
        # np.random.normal(loc=30, scale=5)는 NumPy 라이브러리의 함수를 사용하여 정규 분포에서 랜덤한 값을 생성
        else:
            external_temperature = np.random.normal(loc=0, scale=5)  # 현재 계절이 겨울일 경우, 외부 온도는 평균 0도, 표준편차 5도의 정규 분포를 따르는 난수로 설정

        # Action에 따른 온도 변화
        if action == 0:  # 2도 낮추기
            self.temperature = max(0, self.temperature - 2)
            self.energy_usage += 2  # 온도를 2도 낮추는 데 소요된 에너지를 기록 -> 에너지 사용량이 2만큼 증가
        elif action == 1:  # 1도 낮추기
            self.temperature = max(0, self.temperature - 1)
            self.energy_usage += 1  # 에너지 사용량 증가
        elif action == 2:  # 유지하기
            pass
        elif action == 3:  # 1도 높이기
            self.temperature = min(40, self.temperature + 1)
            self.energy_usage += 1  # 에너지 사용량 증가
        elif action == 4:  # 2도 높이기
            self.temperature = min(40, self.temperature + 2)
            self.energy_usage += 2  # 에너지 사용량 증가

        # 내부 온도 상승
        self.temperature += (external_temperature - self.temperature) * self.temperature_increase_rate
          # 내부온도 = (외부온도 - 현재내부온도) * 온도증가율 로 업데이트
          # 외부온도 > 내부온도 : 양수 -> 내부온도 상승,
          # 외부온도 < 내부온도 : 음수 -> 내부온도 감소.

        # 비용 계산 (에너지 효율성 고려)
        cost = abs(self.target_temperature - self.temperature) * self.cost_per_degree
        energy_cost = 0.5 * abs(external_temperature - self.target_temperature)  # 외부온도가 너무 높거나 낮으면 그만큼 에너지 사용량이 증가하는 거니까 에너지사용량 비용 추가.
        total_cost = cost + energy_cost + self.energy_usage  # 에너지 사용량 추가

        # 사용자 선호도에 따른 보상 조정
        user_reward = -abs(self.user_preference - self.temperature)  # 내부온도가 사용자의 선호온도에 근접할 수록 좋음 -> 절댓값이 작을 수록 좋음 -> - 붙여준 이유

        return self.temperature, user_reward - total_cost  # 온도와 비용(사용자가 실질적으로 얻는 보상) 반환

# DQN 에이전트 구현
class DQNAgent:
    def __init__(self, actions, learning_rate=0.001, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.99, memory_size=2000, batch_size=32):
        self.actions = actions  # 에이전트가 수행할 수 있는 행동의 목록을 저장
        self.learning_rate = learning_rate  # 모델이 학습할 때 가중치를 조정하는 속도를 설정
        self.discount_factor = discount_factor  # 미래 보상을 현재 보상에 얼마나 반영할지를 결정하는 값
        self.exploration_rate = exploration_rate  # 에이전트가 무작위 행동을 선택할 확률
        self.exploration_decay = exploration_decay  # 시간이 지남에 따라 탐험률을 줄이는 비율
        self.memory = []  # 에이전트가 경험을 저장할 메모리
        self.memory_size = memory_size  # 메모리에 저장할 수 있는 최대 경험 수를 설정
        self.batch_size = batch_size   # 메모리에서 샘플링할 경험의 수를 설정. 이는 모델을 학습할 때 사용할 경험의 수를 결정
        self.model = self.build_model()  # 모델을 생성하고, 그 결과를 self.model 속성에 저장하는 부분

    def build_model(self):
        model = models.Sequential()  # Keras의 Sequential 모델을 생성. 이 모델은 층을 순차적으로 쌓아 올리는 구조로, 주로 간단한 신경망에 사용
        model.add(layers.Dense(24, input_dim=1, activation='relu'))   # Dense(24): 24개의 노드를 가진 완전 연결층을 추가. 이 층은 입력을 받아 비선형 변환을 수행. / input_dim=1: 입력의 차원을 1 로 설정. / activation='relu': ReLU(Rectified Linear Unit) 활성화 함수를 사용하여 비선형성을 추가. 이는 신경망의 학습 성능을 향상시킴.
        model.add(layers.Dense(24, activation='relu'))  # 또 다른 24개의 노드를 가진 Dense 층을 추가. 첫 번째 층의 출력을 입력으로 받아 더 복잡한 패턴을 학습할 수 있게 함.
        model.add(layers.Dense(len(self.actions), activation='linear'))  # Dense(len(self.actions)): 에이전트가 선택할 수 있는 행동의 수만큼 노드를 가진 출력층을 추가. 이는 각 행동에 대한 Q-값을 출력. activation='linear': 선형 활성화 함수를 사용하여 Q-값을 직접적으로 출력
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))  # loss='mse': 손실 함수로 평균 제곱 오차(Mean Squared Error)를 사용. 이는 Q-값과 실제 보상 간의 차이를 최소화하는 데 사용 / optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate): Adam 최적화 알고리즘을 사용하여 모델을 학습
        return model

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(self.actions)
        else:  # 탐험이 아닌 활용(exploitation) 전략을 사용
            q_values = self.model.predict(np.array(state).reshape(1, -1), verbose=0)  # 현재 상태(state)를 NumPy 배열로 변환 / 상태가 [1, 2, 3]라는 1차원 배열이라면, reshape(1, -1)을 적용할 경우 결과는 [[1, 2, 3]]가 되어 1행 3열의 2차원 배열로 변환, reshape(1, -1)는 배열의 형태를 2차원으로 변경하여 모델에 입력할 수 있도록 준비하는 과정 / verbose=0은 예측 과정에서의 출력 로그를 표시하지 않도록 설정
            return np.argmax(q_values[0])

    def store_experience(self, state, action, reward, next_state):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)  # 만약 메모리의 크기가 self.memory_size를 초과하면, 가장 오래된 경험을 제거. pop(0)은 리스트의 첫 번째 요소를 삭제하는 메서드임.
        self.memory.append((state, action, reward, next_state))  # 새로운 경험을 메모리에 추가

    def replay(self):  # replay : DQN 에이전트가 경험 재플레이(Experience Replay)를 수행하는 기능을 담당
        if len(self.memory) < self.batch_size:  # 메모리에 저장된 경험의 수가 배치 크기(self.batch_size)보다 작은지를 확인. 배치 크기는 학습에 사용될 경험의 수를 정의
            return

        minibatch = random.sample(self.memory, self.batch_size)  # random.sample 함수는 self.memory에서 지정된 배치 크기(self.batch_size)만큼의 요소를 무작위로 선택하여 샘플링함.이 과정에서 메모리에서 선택된 경험의 리스트가 minibatch에 저장.
        states, actions, rewards, next_states = zip(*minibatch)  # 미니배치에서 상태, 행동, 보상, 다음 상태를 분리하여 별도의 리스트로 저장하는 과정

        states = np.array(states).reshape(self.batch_size, 1)  # states를 NumPy 배열로 변환, (batch_size, 1) 형태의 배열로 재구성
        next_states = np.array(next_states).reshape(self.batch_size, 1)  # next_states를 NumPy 배열로 변환하고, 차원을 (batch_size, 1)로 재구성

        targets = self.model.predict(states, verbose=0)  # self.model.predict(states)는 현재 상태에 대한 Q-값을 예측 / verbose를 0으로 설정하면, 예측 과정에서 출력되는 정보를 최소화함. 즉, 예측 결과만 반환
        next_q_values = self.model.predict(next_states, verbose=0)  # 다음 상태에 대한 Q-값을 예측

        for i in range(self.batch_size):
            target = rewards[i] + self.discount_factor * np.max(next_q_values[i])  # np.max(next_q_values[i])는 다음 상태에서 예측된 Q-값 중 가장 큰 값을 가져옴.
            targets[i][actions[i]] = target  # 계산된 target 값을 targets 배열에 업데이트

        self.model.fit(states, targets, epochs=1, verbose=0)   #  DQN 에이전트가 현재 상태(states)와 목표 Q-값(targets)을 사용하여 모델을 학습시키는 과정 / targets : 목표 Q-값의 배열 / epochs=1 : 한 번의 학습 루프만 수행 / verbose=0 :  학습하는 동안의 상세한 정보를 출력하지 않음.

    def decay_exploration(self):
        self.exploration_rate *= self.exploration_decay  # 탐험률 * 탐험감소율 로 탐험률 업데이트.

# 하이퍼파라미터 설정
actions = [0, 1, 2, 3, 4]  # 액션 정의
num_episodes = 1000   # 에피소드 수
rewards_per_episode = []  # 각 에피소드의 보상을 저장할 리스트


# TensorBoard 로그 설정
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(log_dir)

# 환경 및 에이전트 초기화
env = HVACEnvironment()
agent = DQNAgent(actions)

# 학습 과정
for episode in range(num_episodes):
    state = env.reset()   # state = env.reset() : 함수는 환경을 초기화하고, 초기 상태를 반환, 에피소드가 시작될 때마다 환경을 초기 상태로 설정
    done = False  # 에이전트가 목표를 달성하거나, 특정 조건이 충족되면 done이 True로 바뀌어 에피소드가 종료
    total_reward = 0  # 에피소드 동안의 총 보상 누적. 초기값은 0으로 설정.

    while not done:
        action = agent.choose_action(np.array([state]))  # 현재 상태(state)를 입력으로 받아서 에이전트가 취할 행동(action)을 선택
        next_state, reward = env.step(action)  # env.step(action) 메서드는 주어진 행동에 따라 환경을 업데이트하고, 그 결과로 새로운 상태와 보상을 반환
        agent.store_experience(state, action, reward, next_state)  # 현재 상태, 선택한 행동, 받은 보상, 그리고 다음 상태를 에이전트의 경험 저장소에 저장
        agent.replay()  # 경험 리플레이 메서드를 호출하여, 저장된 경험을 샘플링하고 모델을 학습시킴
        state = next_state  # 현재 상태를 다음 상태로 업데이트
        total_reward += reward  # 보상 누적

        if abs(state - env.target_temperature) < 1:  # state가 목표온도에 근접할 경우 종료
            done = True

    rewards_per_episode.append(total_reward)  # 에피소드당 보상 기록  # rewards_per_episode 리스트에 현재 에피소드에서 얻은 총 보상(total_reward)을 추가

    # TensorBoard에 총 보상 기록
    with summary_writer.as_default():  # 모든 로그는 이 writer를 통해 기록
        tf.summary.scalar('Total Reward', total_reward, step=episode)   #'Total Reward'라는 이름으로 총 보상을 기록

    agent.decay_exploration()  # 탐험 비율 감소. 에이전트는 초기에는 다양한 경험을 쌓고, 후에는 학습한 정보를 바탕으로 더 나은 결정을 내리도록 함.

# 학습 결과 확인
print("학습된 Q-네트워크 모델:")
print(agent.model.summary())

# Learning curve 시각화
plt.plot(rewards_per_episode)
plt.title('Learning Curve')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
