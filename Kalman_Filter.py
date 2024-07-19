import cv2
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, fps, mpp, sigma=10, beta=10):
        '''
        fps: frame per second
        mpp: meter per pixel, 一个像素点有多少米
        sigma: 随机加速度 高斯分布的标准差 m/s^2
        beta: 关联到的检测框与目标真实位置的标准差，不考虑误匹配，只看框的形状准不准
        '''
        sigma = sigma / mpp / fps / fps # pixel/frame^2

        kalman = cv2.KalmanFilter(4, 2)
        # 状态转移矩阵，单位是pixel, pixel/frame
        kalman.transitionMatrix = np.array([[1, 1, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 1],
                                            [0, 0, 0, 1]], np.float32)
        # 过程噪声转移矩阵
        Q = np.array([  [0.5,   0],
                        [  1,   0],
                        [  0, 0.5],
                        [  0,   1]], np.float32)
        # 随机加速度的协方差矩阵
        B = np.array([  [sigma**2,        0],
                        [       0, sigma**2]], np.float32)
        # 过程噪声的协方差矩阵
        kalman.processNoiseCov = np.linalg.multi_dot([Q, B, Q.T])
        # 观测矩阵
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                             [0, 0, 1, 0]], np.float32)
        # 量测噪声的协方差矩阵
        kalman.measurementNoiseCov = np.array([[beta**2,       0],
                                               [      0, beta**2]], np.float32)
        self.kalman = kalman

    def predict(self, points, init_state=None, init_conv=None):
        if init_state is not None:
            self.kalman.statePre= init_state
            self.kalman.errorCovPre = init_conv
        self.kalman.correct(points)
        pred_points = self.kalman.predict().reshape(-1)
        return pred_points

if  __name__ == '__main__':
    fps = 25
    mpp = 0.01
    kalman = KalmanFilter(fps, mpp)
    # 初始状态的估计值
    init_state = np.array([10, 2, 10, 2], dtype=np.float32).reshape([4,1])
    # 初始状态的不确定度(方差)
    init_conv = np.array([  [1000,  0,  0,  0],
                            [0,  1000,  0,  0],
                            [0,  0,  1000,  0],
                            [0,  0,  0,  1000]], np.float32)
    
    points = np.array([10.1, 10.2], dtype=np.float32)

    trues = np.zeros((10, 2))
    preds = np.zeros((10, 2))
    for i in range(10):
        print(points)
        if i == 0:
            points_pred=kalman.predict(points, init_state=init_state, init_conv=init_conv)
        else:
            points_pred=kalman.predict(points)
        print(points_pred)
        trues[i] = points
        preds[i] = points_pred[[0, 2]]
        points=points+1

    plt.plot(trues[:, 0], trues[:, 1], 'o')
    plt.plot(preds[:-1, 0], preds[:-1, 1], '*-.')
    plt.show()