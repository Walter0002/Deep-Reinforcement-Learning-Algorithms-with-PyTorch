import random


class RectPiece:
    def __init__(self, left=0.0, bottom=0.0, width=0.0, height=0.0):
        self.left_ = left
        self.bottom_ = bottom
        self.width_ = width
        self.height_ = height
        self.color = (uniform(0.1, 0.9), uniform(0.1, 0.9), uniform(0.1, 0.9))

    def x(self):
        return self.left_

    def y(self):
        return self.bottom_

    def width(self):
        return self.width_

    def height(self):
        return self.height_

    def width_afterRotate(self):
        return self.height_

    def height_afterRotate(self):
        return self.width_

    def left(self):
        return self.left_

    def bottom(self):
        return self.bottom_

    def right(self):
        return self.left_ + self.width_

    def top(self):
        return self.bottom_ + self.height_

    def setWidth(self, w):
        self.width_ = w

    def setHeight(self, h):
        self.height_ = h

    def setXY(self, x, y):
        self.left_ = x
        self.bottom_ = y

    def setX(self, x):
        self.left_ = x

    def setY(self, y):
        self.bottom_ = y

    def rotate(self):
        self.width_, self.height_ = self.height_, self.width_

    def area(self):
        return self.width_ * self.height_

    def perimeter(self):
        return self.width_ + self.height_

    def long_edge(self):
        return max(self.width_, self.height_)

    def short_edge(self):
        return min(self.width_, self.height_)

import copy, os, csv, math
from collections import namedtuple
import gym
from gym import wrappers
from gym.envs.classic_control import rendering
import numpy as np
import matplotlib as mpl
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot
from random import randint, uniform

def get_data_fromCSV(file:str, format=""):
    pieces = []
    assert os.path.exists(file)
    with open(file, "r") as csvFile:
        reader = csv.reader(csvFile)
        for item in reader:
            # 忽略第一行
            if reader.line_num == 1:
                continue
            if len(item) < 2:
                break
            else:
                pieces.append(RectPiece(0, 0, float(item[0]), float(item[1])))
    return pieces

class Skyline:
    def __init__(self, x, y, w):
        self.x = x
        self.y = y
        self.w = w

class PlacementAction:
    ''' 描述矩形工件的放置方式
    参数依次为：左下角的x、y坐标，是否选择，是否贴近skyline左侧（对于渲染无效，仅仅是记录他用），矩形工件引用
    '''
    def __init__(self, l, b, r, atl, pi):
        self.left = l
        self.bottom = b
        self.rotate = r
        self.at_left = atl
        self.piece = pi

    def __copy__(self):
        return PlacementAction(self.left, self.bottom, self.rotate, self.piece)

Action = namedtuple('Action', ('skyline_index', 'piece_index', 'rotate', 'at_left'))

class RectPackingEnv(gym.Env):
    """Four rooms game environment as described in paper http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf"""
    environment_name = "RectPackingEnv"
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self, strip_width=None, pieces=None, data_dir=None, allow_rotate=False):
        self.data = []
        if data_dir and os.path.exists(data_dir):
            for path in os.listdir(data_dir):
                if not path.endswith('.csv'):
                    continue
                strip_width = float(path.split('_')[1].split('x')[0])
                data_path = os.path.join(data_dir, path)
                print(data_path, strip_width)
                self.data.append((strip_width, get_data_fromCSV(data_path)))
        elif strip_width and pieces:
            self.data = [(strip_width, pieces)]
        else:
            assert False
        assert self.data
        self.allow_rotate = allow_rotate
        self.reward_threshold = 100
        self.reward_for_achieving_goal = 100
        self.step_reward_for_invalid_action = -10
        self.step_reward_for_not_exist_action = -100
        self.embeding_size = 30   # t
        self.action_invalid_count = 0
        self.viewer = None
        self.reset()

        self.actions = set(range(self.embeding_size))
        # # Note that the indices of the grid are such that (0, 0) is the top left point
        self.action_to_effect_dict = {i:i for i in range(self.embeding_size)}
        self.state_only_dimension = 1
        # self.num_possible_states = self.grid_height * self.grid_width
        self.action_space = spaces.Discrete(self.embeding_size)

        self.seed()
        self.reward_threshold = 0.0
        self.trials = 100
        self.max_episode_steps = 1000
        # self.max_episode_steps = self.reward_for_achieving_goal
        self.id = "RectPackingEnv"

    def GetLeftSkyline(self, index):
        ''' 获取 index 左侧的 skyline
        Args:
            skylines:list 默认按照 x 递增排序
        '''
        s = Skyline(0, float('inf'), 0)
        if index - 1 > -1:
            s = self.skylines[index - 1]
        return s

    def GetRightSkyline(self, index):
        ''' 获取 index 右侧的 skyline
        Args:
            skylines:list 默认按照 x 递增排序
        '''
        s = Skyline(0, float('inf'), 0)
        if index + 1 < len(self.skylines):
            s = self.skylines[index + 1]
        return s
    def get_skyline_lh(self, index):
        return abs(self.GetLeftSkyline(index).y - self.skylines[index].y)
    def get_skyline_rh(self, index):
        return abs(self.GetRightSkyline(index).y - self.skylines[index].y)

    def UpdateSkylines(self, index, w=0, h=0, at_left=True):
        ''' Update Skylines
        根据 w 和 h 更新 下标为 index 的 Skyline
        Args:
            skylines:list 默认按照 x 递增排序
            w,h: 待排入矩形的宽高; w=0 表示没有矩形可以使用该 Skyline，需要合并到左或右较低的 Skyline
        Returns:
            ret: w=0 的情况下，是否删除了 index 之外的 skyline
        '''
        s = self.skylines[index]
        # print('---', s.x, s.y, s.w, w, h)
        # print('-----------', len(skylines))
        # for sl in skylines:
        #     print('-----------', int(sl.x), int(sl.y), int(sl.w))
        if w == 0:  # 合并到左或右较低的 skyline
            if len(self.skylines) < 2:
                return
            left_skyline = self.GetLeftSkyline(index)
            right_skyline = self.GetRightSkyline(index)
            self.skylines.pop(index)
            if left_skyline.y <= right_skyline.y:
                self.skylines[index - 1].w += s.w
                index -= 1
            else:
                self.skylines[index].w += s.w
                self.skylines[index].x -= s.w
        else:  # 拆分 skyline
            assert (s.w >= w and w > 0)
            if math.isclose(s.w, w):
                self.skylines[index] = Skyline(s.x, s.y + h, s.w)
            else:
                if at_left:
                    self.skylines[index] = Skyline(s.x, s.y + h, w)
                    self.skylines.insert(index + 1, Skyline(s.x + w, s.y, s.w - w))
                else:
                    self.skylines[index] = Skyline(s.x, s.y, s.w - w)
                    self.skylines.insert(index + 1, Skyline(s.x + s.w - w, s.y + h, w))

        # 如果左右的 y 相同，则需要合并
        s = self.skylines[index]
        left_skyline = self.GetLeftSkyline(index)
        if math.isclose(s.y, left_skyline.y):
            self.skylines[index - 1] = Skyline(left_skyline.x, left_skyline.y, left_skyline.w + s.w)
            self.skylines.pop(index)
            index -= 1
        s = self.skylines[index]
        right_skyline = self.GetRightSkyline(index)
        if math.isclose(s.y, right_skyline.y):
            self.skylines[index] = Skyline(s.x, s.y, s.w + right_skyline.w)
            self.skylines.pop(index + 1)

    def CompareSkylineByY(self, s0: Skyline, s1: Skyline):
        ''' 按 y 和 x 升序比较skyline，用于排序
        '''
        if s0.y > s1.y:
            return 1
        elif s0.y < s1.y:
            return -1
        if s0.x > s1.x:
            return 1
        elif s0.x < s1.x:
            return -1
        return 0


    def get_state_array(self):
        # 默认最大支持100个skyline，100个piece
        # sa = np.array([[sl.x, sl.y, sl.w, 0, 0] for sl in self.skylines])
        # pa = np.array([[pi.width(), pi.height()] for pi in self.remain_pieces])
        # sa = np.resize(sa, (100, 5))
        # pa = np.resize(pa, (100, 2))
        # return np.concatenate((sa, pa), axis=1)
        sa = np.array([[sl.x, sl.y, sl.w, self.get_skyline_lh(i), self.get_skyline_rh(i)] for i,sl in enumerate(self.skylines)])
        indexs = list(range(len(self.remain_pieces)))
        # random.shuffle(indexs)
        pa = np.array([[self.remain_pieces[i].width(), self.remain_pieces[i].height()] for i in indexs])
        sa = np.resize(sa, (self.embeding_size, 5))
        pa = np.resize(pa, (self.embeding_size, 2))
        state = np.concatenate((sa, pa), axis=1)
        # state /= state.max()
        return np.resize(state, (self.embeding_size * 7))

    def render(self, mode='human', close=False):
        if not self.viewer: self.viewer = rendering.Viewer(1000, 800)
        # render skyline
        for sl in self.skylines:
            line1 = rendering.Line((sl.x, sl.y), (sl.x+sl.w, sl.y))
            line1.set_color(1, 0, 0)
            self.viewer.draw_line(line1.start, line1.end, color=(1, 0, 0))
        # render placedPieces
        for place in self.placements:
            p = place.piece
            l, r, t, b = place.left, place.left+p.width(), place.bottom+p.height(), place.bottom
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(*p.color)
            self.viewer.draw_polygon(pole.v, filled=True, color=p.color)

        # 一般用的比较多的就是rgb_array，可以对图像进行修改。human会返回一个bool变量，主要是用来在屏幕上显示当前的游戏图像。ansi目前还不太了解相关具体的应用。
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment and returns the start state"""
        self.current_data_index = random.randint(0, len(self.data)-1)
        self.strip_width, self.pieces = self.data[self.current_data_index][0], copy.deepcopy(self.data[self.current_data_index][1])

        # data enhancement
        random.shuffle(self.pieces)
        # factor = random.randint(2, 20)
        factor = 10
        self.strip_width *= factor
        for p in self.pieces:
            p.setWidth(p.width() * factor)
            p.setHeight(p.height() * factor)

        self.achieving_area_goal = np.array([p.area() for p in self.pieces]).sum()

        self.remain_pieces = self.pieces
        self.skylines = [Skyline(0, 0, self.strip_width)]  # 保存天际线，需要保持从左到右的顺序；因为经常需要插入和删除
        self.placements = []
        self.use_radio = 0

        # self.desired_goal = [self.location_to_state(self.current_goal_location)]
        # self.achieved_goal = [self.location_to_state(self.current_user_location)]
        self.step_count = 0
        self.state = self.get_state_array()
        self.next_state = None
        self.reward = None
        self.done = False
        return self.state

    def array_to_action(self, action_array):
        piece_index = action_array
        rotate = False
        at_left = True
        return Action(0, piece_index, rotate, at_left)

    def step(self, desired_action):
        action = self.array_to_action(desired_action)

        self.step_count += 1
        self.reward = 1 + len(self.skylines)
        # state_next = copy.copy(state)
        self.done = False
        #  寻找最佳放置的 piece 的 skyline，也就是最低最左的那个
        skyline_index = 0
        for i in range(1, len(self.skylines)):
            if self.CompareSkylineByY(self.skylines[skyline_index], self.skylines[i]) > 0:
                skyline_index = i

        if len(self.remain_pieces) > action.piece_index:  # or len(state.skylines) > action.skyline_index:
            sl = self.skylines[skyline_index]
            piece = self.remain_pieces[action.piece_index]
            iw = piece.width_afterRotate() if action.rotate else piece.width()
            ih = piece.height_afterRotate() if action.rotate else piece.height()
            if (sl.w < iw + 0.001):
                # self.reward = self.step_reward_for_invalid_action
                # self.action_invalid_count += 1
                # if self.action_invalid_count > 2:
                #     self.UpdateSkylines(skyline_index)
                if len(self.skylines) > 1 :
                    self.UpdateSkylines(skyline_index)
                self.action_invalid_count += 1
                if self.action_invalid_count > 2:
                    self.reward = self.step_reward_for_invalid_action
                else:
                    self.reward = 0
            else:
                self.action_invalid_count = 0
                palce_x = sl.x if action.at_left else sl.x + sl.w - iw
                placement = PlacementAction(palce_x, sl.y, action.rotate, action.at_left, piece)

                self.placements.append(placement)
                self.remain_pieces.pop(action.piece_index)
                self.UpdateSkylines(skyline_index, iw, ih, action.at_left)
                self.reward -= len(self.skylines)
                self.reward /= 10
        else:
            self.reward = self.step_reward_for_not_exist_action
        if not self.remain_pieces:  # find treasure
            area = np.array([sl.y for sl in self.skylines]).max() * self.strip_width
            self.use_radio = self.achieving_area_goal/area
            # self.reward = -(1 - self.use_radio) * 10
            self.reward = self.use_radio * 100
            self.done = True
        # self.render()
        return self.get_state_array(), self.reward, self.done, {'use_radio': self.use_radio}

    # def compute_reward(self, achieved_goal, desired_goal, info):
    #     """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
    #     interface to fit with the open AI gym specifications"""
    #     if (achieved_goal == desired_goal).all():
    #         reward = self.reward_for_achieving_goal
    #     else:
    #         reward = self.step_reward_for_not_achieving_goal
    #     return reward

    # def calculate_desired_new_state(self, action):
    #     """Calculates the desired new state on basis of action we are going to do"""
    #     if action == 0:
    #         desired_new_state = (self.current_user_location[0] - 1, self.current_user_location[1])
    #     elif action == 1:
    #         desired_new_state = (self.current_user_location[0], self.current_user_location[1] + 1)
    #     elif action == 2:
    #         desired_new_state = (self.current_user_location[0] + 1, self.current_user_location[1])
    #     elif action == 3:
    #         desired_new_state = (self.current_user_location[0], self.current_user_location[1] - 1)
    #     else:
    #         raise ValueError("Action must be 0, 1, 2, or 3")
    #     return desired_new_state

    # def location_to_state(self, location):
    #     """Maps a (x, y) location to an integer that uniquely represents its position"""
    #     return location[0] + location[1] * self.grid_height
    #
    # def state_to_location(self, state):
    #     """Maps a state integer to the (x, y) grid point it represents"""
    #     col = int(state / self.grid_height)
    #     row = state - col*self.grid_height
    #     return (row, col)
    #


if __name__ == '__main__':
    env = RectPackingEnv(1000, [RectPiece(0,0, 300,40), RectPiece(0,0,30,300)])
    env.reset()
    while True:
        env.render()
