import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time
import os

from PIL import Image

#np.random.seed(69420)
#np.seterr(divide='ignore', invalid='ignore')
class RaceTrack:
    """
    Defines the Race Track.
    """
    def __init__(self, load_path="./race_track.png"):

        #Race Track is a 32x32 RGB image.
        #Any other square dimensional race track
        #image can be used.
        #Start must be denoted by red line.
        #Finish must be denoted by green line.
        image_race_track = Image.open(load_path).convert('RGB')
        race_track = np.asarray(image_race_track, dtype=np.float)
        
        self.race_track = np.average(race_track, axis=-1) / 255.0

        self.start = list()
        self.finish = list()
        self.f_positions = list()


        for x in range(self.race_track.shape[1]):
            for y in range(self.race_track.shape[0]):
                
                # Image indexing is X x Y (column x row)
                # Checks for red pixels. Denotes start line.
                if image_race_track.getpixel((x, y)) == (255, 0, 0):
                    # numpy array indexing is row x column
                    self.start.append((y, x))
                # Image indexing is X x Y (column x row)
                # Checks for green pixels. Denotes finish line
                elif image_race_track.getpixel((x, y)) == (0, 255, 0):
                    # numpy array indexing is row x column
                    self.finish.append((y, x))
    
    #Simulate a race episode. 
    def race(self, car, policy):
        start_loc = np.random.choice(np.arange(0, len(self.start)))
        start_loc = self.start[start_loc]
        self.pos = start_loc
        reward = 0.0
        car_trajectory = list()
        car.start_engine()
        
        while True:
            state = (self.pos[0], self.pos[1], car.v_y, car.v_x)
            action_idx, prob = car.get_velocity(self.pos, policy)
            #state = (self.pos[0], self.pos[1], v_y, v_x)
            #car_trajectory.append((state, action_idx))
            
            #Clips position to be within race track dimensions.
            #Otherwise may get index out-of bound error.
            self.pos = np.clip(np.add(self.pos, (car.v_y, car.v_x)), 0, self.race_track.shape[0] - 1)
            
            #Check colision.
            colision = self.check_colision(self.pos)
            if colision:
                #reward -= int(1e3)
                idx = np.random.choice(np.arange(0, len(self.start)))
                self.pos = self.start[idx]
                car.colision()
                #continue
            
            #Check if reached finish line.
            finish = self.check_finish(self.pos)
            if finish:
                reward += 0
                car_trajectory.append((state, (action_idx, prob), reward))
                self.f_positions.append(self.pos)
                break
            else:
                reward += -1
        
            car_trajectory.append((state, (action_idx, prob), reward))
        
        return car_trajectory
    
    def check_colision(self, pos):
        colision = False
        (pos_y, pos_x) = pos

        if self.race_track[pos_y, pos_x] == 0.0:
            colision = True

        return colision
    
    def check_finish(self, pos):
        finish = False
        
        if tuple(pos) in self.finish:
            finish = True

        return finish

class Car:
    """
    The car class used to create an
    instance of the race car.
    """
    def __init__(self, drop=False):
        self.v_x = 0
        self.v_y = 0
        self.drop = drop
        self.actions = [(-1, -1),
                        (-1, 0),
                        (-1, 1),
                        (0, -1),
                        (0, 0),
                        (0, 1),
                        (1, -1),
                        (1, 0),
                        (1, 1)]

    def get_velocity(self, pos, policy):
        #Corresponds to 0 update to the velocities.
        action_idx = 4
                
        if not (self.drop and np.random.rand() <= 0.1):
            action_idx, prob = policy(pos[0], pos[1], self.v_y, self.v_x)
            action = self.actions[action_idx]
            self.v_y = np.clip(self.v_y + action[0], -5, 0)
            self.v_x = np.clip(self.v_x + action[1], 0, 5)

        return action_idx, prob

    
    def start_engine(self):
        self.v_x = 0
        self.v_y = 0

    def colision(self):
        self.v_x = 0
        self.v_y = 0


def monte_carlo_control(race_track, car, episodes, gamma=0.7):
    state_action_values = np.zeros((race_track.race_track.shape[0], race_track.race_track.shape[1],
                                    6, 6, 9), dtype=np.float64)
    
    state_action_counts = np.zeros((race_track.race_track.shape[0], race_track.race_track.shape[1],
                                    6, 6, 9), dtype=np.float64)
    
    def car_policy(p_y, p_x, v_y, v_x):
        prob = 1.0/9.0
        return np.random.randint(0, 9), prob
        """
        #nonlocal initial
        #if initial:
        if np.random.rand() < 0.1:
            #initial = False
            return np.random.choice(np.arange(0, 9))

        max_value = np.max(state_action_values, axis=-1)[p_y, p_x, v_y, v_x]

        actions = [action for action in range(0, 9) if\
                   state_action_values[p_y, p_x, v_y, v_x, action] == max_value]

        return np.random.choice(actions)
    """
    
    pbar = tqdm.tqdm(range(episodes), desc="Bar Desc", leave=True)
    for eps in pbar:#tqdm.tqdm(range(episodes), desc="W: {}".format(W)):
        
        #initial = True
        #state_count_dict = dict()

        trajectory = race_track.race(car, car_policy)
        
        G = 0.0
        W = 1.0
        for state, action_, reward in reversed(trajectory):
            pbar.set_description("len(trajectory): {} .W: {}".format(len(trajectory), W))
            pbar.refresh()
            G = gamma*G + reward
            action = action_[0]
            a_prob = action_[1]
            #state_count_dict[(state, action)] = G
            state_action_counts[state[0], state[1], state[2], state[3], action] += W
            
            
            state_action_values[state[0], state[1], state[2], state[3], action] += \
                W * (G - state_action_values[state[0], state[1], state[2], state[3], action])/\
                state_action_counts[state[0], state[1], state[2], state[3], action]
            
            if action != np.argmax(state_action_values[state[0], state[1], state[2], state[3]], axis=-1):
                break

            W *= 1 / a_prob

            
            #state_action_counts[state[0], state[1], state[2], state[3], action] += 1

        """
        for (state, action) in state_count_dict.keys():
            state_action_values[state[0], state[1], state[2], state[3], action] +=\
                state_count_dict[(state, action)]
            
            state_action_counts[state[0], state[1], state[2], state[3], action] += 1
        """

    return state_action_values#/state_action_counts

#To plot race trajectories of test simulations.
def test_race(load_path, save_dir, num_races=5):
    race_track = RaceTrack()
    race_car = Car()

    state_action_values = np.load(load_path)
    policy = np.argmax(state_action_values, axis=-1)
    
    def greedy_policy(p_y, p_x, v_y, v_x):
        return policy[p_y, p_x, v_y, v_x], 1
        """
        max_value = np.max(state_action_values, axis=-1)
        
        actions = [action for action in range(0, 9) if \
                   state_action_values[p_y, p_x, v_y, v_x] == max_value]
        
        return np.random.choice(actions)
        """
    
    for race_num in range(num_races):
        trajectory = race_track.race(race_car, greedy_policy)

        track = race_track.race_track
        track = np.repeat(track[:, :, np.newaxis], 3, axis=2)
        track *= 255.0
        for (y, x) in race_track.start:
            track[y, x] = np.array([255.0, 0.0, 0.0])
        for (y, x) in race_track.finish:
            track[y, x] = np.array([0.0, 255.0, 0.0])

        for (p_y, p_x, v_y, v_x), _, _ in trajectory:
            track[p_y, p_x] = np.array([0.0, 0.0, 255.0])
        (p_y, p_x) = race_track.f_positions[-1]
        track[p_y, p_x] = np.array([0.0, 0.0, 255.0])
        
        cnt = len(trajectory)
        image = Image.fromarray(np.uint8(track))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image.save(os.path.join(save_dir, 
                                'race_{}_steps_taken_{}.png'.format(str(race_num+1), str(cnt))))

if __name__ == "__main__":
    
    race_track = RaceTrack()
    race_car = Car(drop=True)
    episodes = int(1e6)

    #Velocity updates becomes zero with a probability of 0.1.
    #print("Simulation 1. Velocity update may become ZERO with probability 0.1!\n")
    #sa = monte_carlo_control(race_track, race_car, episodes)
    #np.save("./c_race_drop_sa.npy", sa)

    #Normal race car.
    race_car.drop = False
    print("\nSimulation 2. Normal races!\n")
    sa = monte_carlo_control(race_track, race_car, episodes, gamma=0.9)
    np.save("./gamma_race_normal_sa.npy", sa)
   
    #Exhibit Sample Trajectories
    #test_race('./c_race_drop_sa.npy', './c_race_rdrop', 5)
    test_race('./gamma_race_normal_sa.npy', './gamma_race_normal', 5)

