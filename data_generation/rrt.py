"""
Path planning with Rapidly-Exploring Random Trees (RRT)
author: Aakash(@nimrobotics)
web: nimrobotics.github.io
"""

import cv2
import numpy as np
import math
import random
import argparse
import os


class Nodes:
    """Class to store the RRT graph"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent_x = []
        self.parent_y = []


class RRT():
    def __init__(self, img, start, end, step_size, max_search=1000):
        self.node_list = [0]
        self.img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.start = start
        self.end = end
        self.step_size = step_size
        self.max_search = max_search

    def solve(self):
        '''
        img: gray img
        img_gray: colored img
        '''
        img = np.copy(self.img)
        img_gray = self.img_gray
        node_list = self.node_list
        start = self.start
        end = self.end
        draw = False

        h, l = img_gray.shape  # dim of the loaded image
        # print(img.shape) # (384, 683)
        # print(h,l)

        # insert the starting point in the node class
        # node_list = [0] # list to store all the node points         
        node_list[0] = Nodes(start[0], start[1])
        node_list[0].parent_x.append(start[0])
        node_list[0].parent_y.append(start[1])

        i = 1;
        count = 0
        while count < self.max_search:
            nx, ny = self._rnd_point(h, l)
            # print("Random points:",nx,ny)

            nearest_ind = self._nearest_node(nx, ny)
            nearest_x = node_list[nearest_ind].x
            nearest_y = node_list[nearest_ind].y
            # print("Nearest node coordinates:",nearest_x,nearest_y)

            # check direct connection
            tx, ty, directCon, nodeCon = self._check_collision(nx, ny, nearest_x, nearest_y)
            if tx is None:  # out of map point
                continue

            # find path
            if directCon and nodeCon:
                print("Node can connect directly with end")
                node_list.append(i)
                node_list[i] = Nodes(tx, ty)
                node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
                node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
                node_list[i].parent_x.append(tx)
                node_list[i].parent_y.append(ty)

                path = []
                for j in range(len(node_list[i].parent_x) - 1):
                    path.append(np.asarray([node_list[i].parent_x[j], node_list[i].parent_y[j]]))
                path.append(np.asarray([end[0], end[1]]))
                return path

            # find connected node
            elif nodeCon:
                # print("Nodes connected")
                node_list.append(i)
                node_list[i] = Nodes(tx, ty)
                node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
                node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
                # print(i)
                # print(node_list[nearest_ind].parent_y)
                node_list[i].parent_x.append(tx)
                node_list[i].parent_y.append(ty)
                i = i + 1
                # display
                # cv2.circle(img, (int(tx),int(ty)), 2,(0,0,255),thickness=3, lineType=8)
                # cv2.line(img, (int(tx),int(ty)), (int(node_list[nearest_ind].x),int(node_list[nearest_ind].y)), (0,255,0), thickness=1, lineType=8)
                # cv2.imwrite("media/"+str(i)+".jpg",img)
                # cv2.imshow("sdc",img)
                # cv2.waitKey(1)
                count += 1
                continue

            # sampled point is not reachable to any node
            else:
                # print("No direct con. and no node con. :( Generating new rnd numbers")
                count += 1
                continue

        print("RRT can not solve path finding")
        return None

    # check collision
    def _collision(self, x1, y1, x2, y2):
        img = self.img_gray
        imshow_offset = 0.5
        x1 = x1 + imshow_offset
        x2 = x2 + imshow_offset
        y1 = y1 + imshow_offset
        y2 = y2 + imshow_offset
        if (int(x1) == int(x2)) or (int(y1) == int(y2)):
            if (img[int(y1), int(x1)] != 0) and (img[int(y2), int(x2)] != 0):
                # print("no collision")
                return False
            else:
                # print("collision")
                return True

        x = [(x1 * (100 - i) + i * x2) / 100 for i in range(101)]
        y = [(y1 * (100 - i) + i * y2) / 100 for i in range(101)]
        truncation_fix_x = [0.01, -0.01]
        truncation_fix_y = [0.01, -0.01]

        # print("collision",x,y)
        for i in range(len(x)):
            # print(int(x[i]),int(y[i]))
            try:
                for dx in truncation_fix_x:
                    for dy in truncation_fix_y:
                        if img[int(y[i] + dy), int(x[i] + dx)] == 0:
                            return True
            except:
                print("index error detected!")
                continue
        return False  # no-collision

        # color = []
        # x = [(x1 * (100 - i) + i * x2) / 100 for i in range(101)]
        # y = [(y1 * (100 - i) + i * y2) / 100 for i in range(101)]
        # # print("collision",x,y)
        # for i in range(len(x)):
        #     # print(int(x[i]),int(y[i]))
        #     try :
        #         color.append(img[int(y[i]), int(x[i])])
        #     except :
        #         print("index error detected!")
        #         continue
        #
        #
        # if (0 in color):
        #     return True  # collision
        # else:
        #     return False  # no-collision

    # check the  collision with obstacle and trim
    def _check_collision(self, x1, y1, x2, y2):  # x1,y1 is new sampled point
        img_gray = self.img_gray
        step_size = self.step_size

        _, theta = self._dist_and_angle(x2, y2, x1, y1)
        x = x2 + step_size * np.cos(theta)
        y = y2 + step_size * np.sin(theta)

        # TODO: trim the branch if its going out of image area
        # print("Image shape",img.shape)
        hy, hx = img_gray.shape
        if y < 0 or y > hy or x < 0 or x > hx:
            # print("Point out of image bound")
            # print((x,y))
            y = None
            x = None
            directCon = False
            nodeCon = False
        else:
            # check direct connection
            dist_from_end = np.sqrt((x - self.end[0]) ** 2 + (y - self.end[1]) ** 2)
            if self._collision(x, y, self.end[0], self.end[1]) or (dist_from_end > self.step_size):
                directCon = False
            else:
                directCon = True

            # check connection between two nodes
            if self._collision(x, y, x2, y2):
                nodeCon = False
            else:
                nodeCon = True

        return (x, y, directCon, nodeCon)

    # return dist and angle b/w new point and nearest node
    def _dist_and_angle(self, x1, y1, x2, y2):
        dist = math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
        angle = math.atan2(y2 - y1, x2 - x1)
        return (dist, angle)

    # return the neaerst node index
    def _nearest_node(self, x, y):
        node_list = self.node_list
        temp_dist = []
        for i in range(len(node_list)):
            dist, _ = self._dist_and_angle(x, y, node_list[i].x, node_list[i].y)
            temp_dist.append(dist)
        return temp_dist.index(min(temp_dist))

    # generate a random point in the image space
    def _rnd_point(self, h, l):
        new_y = random.randint(0, h)
        new_x = random.randint(0, l)
        return (new_x, new_y)

    def _draw_circle(self, event, x, y, flags, param):
        global coordinates
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(self.img_gray, (x, y), 5, (255, 0, 0), -1)
            coordinates.append(x)
            coordinates.append(y)
