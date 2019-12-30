import numpy as np

class GameGrid():
    def __init__(self):
        self.matrix = np.zeros((4,4))
        self.action_dict = {
            'w': 1,
            'a': 2,
            's': 3,
            'd': 4
        }
        
        self.add_two()
        self.add_two()

        
    def get_action_key(self):
        while True:
            k = input("Please input your action: ")
            action = self.action_dict.get(k, 0)
            if action > 0:
                break
            
        return action

    def get_action_random(self):
        action = np.random.randint(0,4)
        return action

    def update(self, action):
        
        # UP
        if action == 1:
            self.matrix = self.update_up(self.matrix)
            if self.is_full() == False:
                self.add_two()
            
        # LEFT
        elif action == 2:
            self.matrix = self.update_up(self.matrix.T)
            if self.is_full() == False:
                self.add_two()
            self.matrix = self.matrix.T
        
        # DOWN
        elif action == 3:
            self.matrix = self.update_up(np.flipud(self.matrix))
            if self.is_full() == False:
                self.add_two()
            self.matrix = np.flipud(self.matrix)

        # RIGHT
        elif action == 4:
            self.matrix = self.update_up(np.flipud(self.matrix.T))
            if self.is_full() == False:
                self.add_two()
            self.matrix = np.flipud(self.matrix).T
        
        done = self.is_terminal()
        reward = np.sum(self.matrix)
        return reward, done
    
    def update_up(self, matrix):
        N = matrix.shape[0]
        # Squeeze
        new = np.zeros((4,4))
        for j in range(N):
            count = 0
            for i in range(N):
                if matrix[i][j] != 0:
                    new[count][j] = matrix[i][j]
                    count += 1
        matrix = new


        # summation
        for i in range(N-1):
            for j in range(N):
                if matrix[i][j] == matrix[i+1][j]:
                    matrix[i][j] *= 2
                    matrix[i+1][j] = 0

        # squeeze
        new = np.zeros((4,4))
        for j in range(N):
            count = 0
            for i in range(N):
                if matrix[i][j] != 0:
                    new[count][j] = matrix[i][j]
                    count += 1
        matrix = new
        
        return matrix
        
    
    def add_two(self):
        count = 0
        while count < 20:
            a = np.random.randint(0,4)
            b = np.random.randint(0,4)
            if (self.matrix[a][b] == 0):
                self.matrix[a][b] = 2
                # print("add to position {}/{}".format(a,b))
                break
            count += 1
        
    def is_full(self):
        sum = np.sum(self.matrix == 0)
        return sum == 0

    def is_terminal(self):
        if self.is_full() == True:
            tmp = np.copy(self.matrix)
            if np.sum(np.abs(self.update_up(tmp)-tmp)) > 0:
                return False
            elif np.sum(np.abs(self.update_up(tmp.T).T-tmp)) > 0:
                return False
            elif np.sum(np.abs(np.flipud(self.update_up(np.flipud(tmp)))-tmp)) > 0:
                return False
            elif np.sum(np.abs(np.flipud(self.update_up(np.flipud(tmp).T).T)-tmp)) > 0:
                return False
            else:
                return True
        return False

    def print_grid(self):
        print(self.matrix)

    def get_state(self):
        return np.copy(self.matrix)



if __name__ == "__main__":
    game = GameGrid()

    done = False
    while True:
        action = game.get_action_random()
        reward, done = game.update(action)
        # game.print_grid()

        if done:
            game.print_grid()
            break
    print("game over")
    print("REWARD is {}".format(reward))
    
    

