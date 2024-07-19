import cv2
import numpy as np
from Kalman_Filter import KalmanFilter

# 航迹头，自由点迹
# 航迹起始，连续2帧关联
# oo
# 航迹确认，连续3帧关联，升级成稳定航迹
# ooo
# 稳定航迹的匹配，5帧内最少有3帧关联
# oooxx  ooxox  oxoox  xooox
# ooxxo  oxoxo  xooxo
# oxxoo  xoxoo
# xxooo
# 航迹撤销，5帧内只有2帧关联，对于航迹头和起始航迹，只要下一帧没关联就删掉
# xxoox  xoxox  oxxox
# xooxx  oxoxx
# ooxxx

def calc_mse(a, b):
    # a[m, 2], b[n, 2]
    return np.sqrt(np.square(a[:, 0][:, None] - b[:, 0][None, :]) + 
                   np.square(a[:, 1][:, None] - b[:, 1][None, :]))

def get_confusion_matrix(states, tembeds, points, embeds, pixel_thresh, cos_thresh, coef):
    '''
    states: [m, 2]
    tembeds: [m, 128]
    points: [n, 2]
    embeds: [n, 128]
    pixel_thresh: pixel mse的最大容忍度
    cos_thresh: cos相似度的最低容忍度
    coef: mse 和 cos 的权重调节因子
    '''
    pos_mse = calc_mse(states, points) # [m, n]，越小越好
    pos_mse = pos_mse / pixel_thresh # 归一化到0-1之间
    pos_mse[pos_mse > 1] = 10000  # thresh截断

    neg_cos_sim = 1 - np.matmul(tembeds, embeds.T) # [m, n]，越小越好, 0-1之间
    neg_cos_sim = neg_cos_sim / (1 - cos_thresh) # 归一化到0-1之间
    neg_cos_sim[neg_cos_sim > 1] = 10000  # thresh截断

    cmatrix = (1 - coef) * pos_mse + coef * neg_cos_sim  # [m, n]
    return cmatrix, pos_mse, neg_cos_sim

def greedy_assign_pairs(matrix, pos_mse, neg_cos_sim, thresh):
    m, n = matrix.shape
    min_index = np.argsort(matrix.reshape(-1))
    rows = min_index // n
    cols = min_index % n

    set_r = set()
    set_c = set()
    pairs = []
    scores = []
    mse_scores = []
    cos_scores = []
    for i in range(min_index.shape[0]):
        r, c = rows[i], cols[i]
        score = matrix[r, c]
        mse_score = pos_mse[r, c]
        cos_score = neg_cos_sim[r, c]
        if score >= thresh:
            break
        if r in set_r or c in set_c:
            continue
        pairs.append([r, c])
        scores.append(score)
        mse_scores.append(mse_score)
        cos_scores.append(cos_score)
        set_r.add(r)
        set_c.add(c)

    return pairs, set_r, set_c, scores, mse_scores, cos_scores

class IdManager():
    def __init__(self, max_size=100000):
        self.id = 0
        self.max_size = max_size

    def get_new_id(self):
        id = self.id
        self.id = (self.id + 1) % self.max_size
        return id

class Track():
    def __init__(self, id, box_score, embedding, point, box, fps, mpp):
        self.id = id
        self.box_scores = [box_score]
        self.boxes = [box]
        self.measures = [point]  # a list of points [x, y]
        self.preds = []  # a list of points [x, vx, y, vy]
        self.embedding = embedding  # last frame's emb 128
        self.flags = [1]
        self.scores = [0]
        self.mse_scores = [0]
        self.cos_scores = [0]
        self.kalman_filter = KalmanFilter(fps, mpp)
        self.init_conv = np.array([ [1000,    0,    0,    0],
                                    [   0, 1000,    0,    0],
                                    [   0,    0, 1000,    0],
                                    [   0,    0,    0, 1000]], np.float32)

    def add_info(self, point, embedding, flag, score, mse_score, cos_score, pred_state, box_score, box):
        self.box_scores.append(box_score)
        self.boxes.append(box)
        self.measures.append(point)
        if embedding is not None:
            self.embedding = embedding
        self.flags.append(flag)
        self.scores.append(score)
        self.mse_scores.append(mse_score)
        self.cos_scores.append(cos_score)
        self.preds.append(np.copy(pred_state))

    def head_upgrade(self, point, embedding, box_score, box, score, mse_score, cos_score):
        init_state = np.array([point[0], point[0] - self.measures[0][0], 
                               point[1], point[1] - self.measures[0][1]], dtype=np.float32)
        self.preds = [init_state]
        pred_state = self.kalman_filter.predict(point, init_state=np.copy(init_state).reshape([4,1]), init_conv=self.init_conv)
        self.add_info(point, embedding, 1, score, mse_score, cos_score, pred_state, box_score, box)

    def init_upgrade(self, point, embedding, box_score, box, score, mse_score, cos_score):
        pred_state = self.kalman_filter.predict(point)
        self.add_info(point, embedding, 1, score, mse_score, cos_score, pred_state, box_score, box)

    def stable_update(self, point, embedding, box_score, box, score, mse_score, cos_score):
        pred_state = self.kalman_filter.predict(point)
        self.add_info(point, embedding, 1, score, mse_score, cos_score, pred_state, box_score, box)

    def need_delete(self):
        # 最近4帧只有两个关联
        return sum(self.flags[-4:]) <= 2
    
    def stable_guess(self):
        point = self.preds[-1][[0, 2]]
        pred_state = self.kalman_filter.predict(point)
        for i in range(1, 4):
            if self.flags[-i] > 0:
                break
        pred_box = np.copy(self.boxes[-i])
        pred_box[[0, 2]] += (self.preds[-1][0] - self.measures[-i][0]) # shift pred
        pred_box[[1, 3]] += (self.preds[-1][2] - self.measures[-i][1])
        self.add_info(None, None, 0, 10000, 10000, 10000, pred_state, 0, pred_box)

class TrackManager():
    def __init__(self, fps, mpp=0.01, measure_bias=10):
        '''
        mpp: meters per pixel
        measure_bias: 检测框预估和目标真实中心点的最大像素偏差
        '''
        # 位置和embedding相似度相结合，匀速运动模型，加速度作为噪声项
        # 人的步速正常在1-2m/s之间，跑步时2-4m/s之间，航迹头关联最大容忍度 v = 4 m/s
        # 假设fps=25, 最大帧速度 u = v / fps = 4 / 25 = 0.16 m/frame
        # 假设mpp=0.01，换算到像素距离 w = u / mpp = 0.16 / 0.01 = 16 pixel/frame
        # 所以航迹关联时能接受的最大标准差是16 + measure_bias
        self.fps = fps
        self.mpp = mpp
        self.coef_stable = 0.5  # 稳定航迹关联 mse 和 cos 的权重调节因子
        self.coef_init = 0.5  # 起始航迹关联 mse 和 cos 的权重调节因子
        self.coef_head = 0.9  # 航迹头关联 mse 和 cos 的权重调节因子
        self.pixel_thresh_stable = 10 / fps / mpp + measure_bias # 稳定航迹关联 pixel mse的最大容忍度
        self.pixel_thresh_init = 10 / fps / mpp + measure_bias # 起始航迹关联 pixel mse的最大容忍度
        self.pixel_thresh_head = 10 / fps / mpp + measure_bias # 航迹头关联 pixel mse的最大容忍度
        self.cos_thresh = 0.7 # cos相似度的最低容忍度
        self.track_heads = [] # 航迹头
        self.init_tracks = [] # 起始航迹
        self.stable_tracks = [] # 稳定航迹
        self.id_manager = IdManager()

    def update(self, scores, embeds, boxes):
        # embedding归一化
        embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True) # [n, 128]
        # 因为人头大小基本相同，所以把boxes转换成中心点
        point_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        point_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
        points = np.concatenate((point_x[:, None], point_y[:, None]),axis=1) # [n, 2]

        scores, embeds, points, boxes = self.stable_track_matching(scores, embeds, points, boxes)
        scores, embeds, points, boxes = self.initial_track_matching(scores, embeds, points, boxes)
        scores, embeds, points, boxes = self.track_head_matching(scores, embeds, points, boxes)
        self.free_dot_init(scores, embeds, points, boxes)

    def stable_track_matching(self, box_scores, embeds, points, boxes):
        # 稳定航迹 m 条，新检测点迹 n 个
        if len(self.stable_tracks) == 0 or points.shape[0] == 0:
            return box_scores, embeds, points, boxes
        
        states = np.array(list(map(lambda x: x.preds[-1], self.stable_tracks)))[:, (0, 2)] # [m, 2]
        tembeds = np.array(list(map(lambda x: x.embedding, self.stable_tracks))) # [m, 128]
        cmatrix, pos_mse, neg_cos_sim = get_confusion_matrix(states, tembeds, points, embeds, self.pixel_thresh_stable, self.cos_thresh, self.coef_stable)
        pairs, set_r, set_c, scores, mse_scores, cos_scores = greedy_assign_pairs(cmatrix, pos_mse, neg_cos_sim, thresh=1)

        # 稳定航迹维持
        new_tracks = []
        for (i, j), score, mse_score, cos_score in zip(pairs, scores, mse_scores, cos_scores):
            self.stable_tracks[i].stable_update(points[j], embeds[j], box_scores[j], boxes[j], score, mse_score, cos_score)
            new_tracks.append(self.stable_tracks[i])

        # 对于没有匹配到的航迹，根据规则判断是否删掉该航迹
        set_t = set(range(len(self.stable_tracks)))
        set_d = list(set_t - set_r)
        for i in set_d:
            if not self.stable_tracks[i].need_delete():
                self.stable_tracks[i].stable_guess()
                new_tracks.append(self.stable_tracks[i])
        self.stable_tracks = new_tracks

        # 返回新点迹中没匹配到的点迹
        set_p = set(range(points.shape[0]))
        set_d = list(set_p - set_c)
        return box_scores[set_d], embeds[set_d], points[set_d], boxes[set_d]

    def initial_track_matching(self, box_scores, embeds, points, boxes):
        # 初始航迹 m 个，新检测点迹 n 个
        if len(self.init_tracks) == 0 or points.shape[0] == 0:
            return box_scores, embeds, points, boxes
        
        states = np.array(list(map(lambda x: x.preds[-1], self.init_tracks)))[:, (0, 2)] # [m, 2]
        tembeds = np.array(list(map(lambda x: x.embedding, self.init_tracks))) # [m, 128]
        cmatrix, pos_mse, neg_cos_sim = get_confusion_matrix(states, tembeds, points, embeds, self.pixel_thresh_init, self.cos_thresh, self.coef_init)
        pairs, set_r, set_c, scores, mse_scores, cos_scores = greedy_assign_pairs(cmatrix, pos_mse, neg_cos_sim, thresh=1)

        # 初始航迹升级为稳定航迹
        for (i, j), score, mse_score, cos_score in zip(pairs, scores, mse_scores, cos_scores):
            self.init_tracks[i].init_upgrade(points[j], embeds[j], box_scores[j], boxes[j], score, mse_score, cos_score)
            self.stable_tracks.append(self.init_tracks[i])

        # 清空初始航迹，没匹配到的就扔掉
        self.init_tracks = []

        # 返回新点迹中没匹配到的点迹
        set_p = set(range(points.shape[0]))
        set_d = list(set_p - set_c)
        return box_scores[set_d], embeds[set_d], points[set_d], boxes[set_d]

    def track_head_matching(self, box_scores, embeds, points, boxes):
        # 航迹头 m 个，新检测点迹 n 个
        if len(self.track_heads) == 0 or points.shape[0] == 0:
            return box_scores, embeds, points, boxes
        
        states = np.array(list(map(lambda x: x.measures[0], self.track_heads))) # [m, 2]
        tembeds = np.array(list(map(lambda x: x.embedding, self.track_heads))) # [m, 128]
        cmatrix, pos_mse, neg_cos_sim = get_confusion_matrix(states, tembeds, points, embeds, self.pixel_thresh_head, self.cos_thresh, self.coef_head)
        pairs, set_r, set_c, scores, mse_scores, cos_scores = greedy_assign_pairs(cmatrix, pos_mse, neg_cos_sim, thresh=1)

        # 航迹头升级为初始航迹
        for (i, j), score, mse_score, cos_score in zip(pairs, scores, mse_scores, cos_scores):
            self.track_heads[i].head_upgrade(points[j], embeds[j], box_scores[j], boxes[j], score, mse_score, cos_score)
            self.init_tracks.append(self.track_heads[i])

        # 清空航迹头，没匹配到的就扔掉
        self.track_heads = []

        # 返回新点迹中没匹配到的点迹
        set_p = set(range(points.shape[0]))
        set_d = list(set_p - set_c)
        return box_scores[set_d], embeds[set_d], points[set_d], boxes[set_d]

    def free_dot_init(self, box_scores, embeds, points, boxes):
        if points.shape[0] == 0:
            return
        
        for i in range(points.shape[0]):
            id = self.id_manager.get_new_id()
            track = Track(id, box_scores[i], embeds[i], points[i], boxes[i], self.fps, self.mpp)
            self.track_heads.append(track)

if __name__ == "__main__":
    # a = np.ones((5, 2))
    # b = np.ones((4, 2))
    # a[:, 0] *= 2
    # a[:, 1] *= 3
    # b[:, 0] /= 2
    # b[:, 1] /= 3
    # x = calc_mse(a, b)

    # matrix = np.random.randn(3, 4)
    # pairs, set_r, set_c = greedy_assign_pairs(matrix, 1)

    idm = IdManager()
    id = idm.get_new_id()
    id = idm.get_new_id()
    a = 1