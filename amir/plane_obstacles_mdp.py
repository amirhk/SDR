import numpy as np
from PIL import Image, ImageDraw


class PlaneObstaclesMDP(object):
    def __init__(self, H=40, W=40):
        super(PlaneObstaclesMDP, self).__init__()
        self.H, self.W = H, W
        self.agent_size = 4

        self.action_dim = 2
        self.arange_min = np.array([-3, -3])
        self.arange_max = np.array([3, 3])

        self.obstacles = np.array([[2, 1], [1, 2], [2, 3], [2, 1.5], [3, 2], [2, 2.5]])
        self.obstacles[:, 0] = (self.obstacles[:, 0] - 2.5) * 10 + 25
        self.obstacles[:, 1] = (self.obstacles[:, 1] - 2.5) * 15 + 27.5
        self.im = Image.new('L', (W, H))
        self.draw = ImageDraw.Draw(self.im)

    def reward_function(self, s, a):
        return -1, False

    def transition_function(self, s, a):
        
        
        true_state = s[0]
        next_state = true_state + a
        return (next_state, self.render(next_state))

    def sample_random_state(self, shape='rectangle'):
#        p = np.array([np.random.randint(self.agent_size, self.H/2 - self.agent_size + 1),
#                      np.random.randint(self.agent_size, self.W/2 - self.agent_size + 1)]) # added
        p = np.array([np.random.randint(self.agent_size, self.H - self.agent_size + 1),
                      np.random.randint(self.agent_size, self.W - self.agent_size + 1)])
        return (p, self.render(p, shape))

    def is_valid_action(self, s, a):
        ns = np.array(s[0]) + np.array(a)
        return self.agent_size <= ns[0] <= self.H - self.agent_size and \
               self.agent_size <= ns[1] <= self.W - self.agent_size

    def render(self, pos, shape='rectangle'):
        self.draw.rectangle((0, 0, self.W, self.H), fill=0)

        # draw obstacles
        '''
        if shape == 'rectangle':
            for obs in self.obstacles:
                x_start = obs[1] - 2
                x_end = obs[1] + 2
                y_start = obs[0] - 2
                y_end = obs[0] + 2
                self.draw.ellipse((x_start, y_start, x_end, y_end), fill=255)
        '''

        # draw agent
        if shape == 'rectangle':
            self.draw.rectangle((pos[1] - self.agent_size/2, pos[0] - self.agent_size/2, pos[1] + self.agent_size/2, pos[0] + self.agent_size/2), fill=255)
        elif shape == 'cross':
            #self.draw.pieslice((pos[1] - 3, pos[0] - 3, pos[1] + 3, pos[0] + 3), 135, 225, fill=255)
            #self.draw.ellipse((pos[1] - self.agent_size/2, pos[0] - self.agent_size/2, pos[1] + self.agent_size/2, pos[0] + self.agent_size/2), fill=255)
            self.draw.line((pos[1] , pos[0], pos[1] + self.agent_size/2, pos[0] + self.agent_size/2), fill=255)
            self.draw.line((pos[1] , pos[0], pos[1] - self.agent_size/2, pos[0] - self.agent_size/2), fill=255)
            self.draw.line((pos[1] , pos[0], pos[1] + self.agent_size/2, pos[0] - self.agent_size/2), fill=255)
            self.draw.line((pos[1] , pos[0], pos[1] - self.agent_size/2, pos[0] + self.agent_size/2), fill=255)
        else:
            print('Not Recognized Shape ' + shape)
            

        return np.asarray(self.im) / 255.0
