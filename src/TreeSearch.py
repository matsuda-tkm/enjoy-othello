import sys
sys.path.append('../')
import torch
import numpy as np
from copy import copy
from model.networks import *

class Node:
    def __init__(self, board):
        self.data = [None, None, None, None]
        self.board = board
        self.children = []

    def add(self, child):
        self.children.append(child)

    def backward(self, model_v=None):
        if not self.children:
            self.data = self.evaluate(model_v)
        else:
            for child in self.children:
                child.backward(model_v)
            absmin,absmax,min_,max_ = np.inf,-np.inf,np.inf,-np.inf
            for child in self.children:
                absmin = min(absmin, child.get_absmin())
                absmax = max(absmax, child.get_absmax())
                min_ = min(min_, child.get_min())
                max_ = max(max_, child.get_max())
            self.data = [absmin, absmax, min_, max_]

    def evaluate(self, model_v=None):
        if model_v is None:
            assert self.board.is_game_over(), self.board.to_line()
            z = self.board.diff_num() if self.board.turn else -self.board.diff_num()
            return (abs(z), abs(z), z, z)
        else:
            with torch.no_grad():
                z = model_v(board_to_array_aug2(self.board,True)).mean().item()*64
            return (abs(z), abs(z), z, z)
    
    def get_absmin(self):
        return self.data[0]

    def get_absmax(self):
        return self.data[1]

    def get_min(self):
        return self.data[2]

    def get_max(self):
        return self.data[3]


def apply_move(board, move):
    """仮想盤面を生成"""
    board_ = copy(board)
    board_.move(move)
    return board_

def create_tree(node, depth):
    """根=node,深さ=depthの木を作成"""
    if depth == 0:
        return
    for move in list(node.board.legal_moves):
        new_board = apply_move(node.board, move)
        child = Node(new_board)
        node.add(child)
        if not new_board.is_game_over():
            create_tree(child, depth - 1)


def minimax_draw(node, turn, depth):
    '''ミニマックス法で引き分け最善手を探索する関数'''
    if depth == 0 or node.board.is_game_over():
        z = node.get_absmin()
        return z, list(node.board.legal_moves)[0]
    
    if node.board.turn == turn:
        zbest = np.inf
        mbest = None
        for i,child in enumerate(node.children):
            z,_ = minimax_draw(child, turn, depth-1)
            if zbest > z:
                zbest = z
                mbest = list(node.board.legal_moves)[i]
    else:
        zbest = -np.inf
        mbest = None
        for i,child in enumerate(node.children):
            z,_ = minimax_draw(child, turn, depth-1)
            if zbest < z:
                zbest = z
                mbest = list(node.board.legal_moves)[i]
    assert (abs(zbest)<100) and (mbest is not None), f'zbest={zbest}, mbest={mbest}'
    return zbest, mbest



def draw_ai(board, legal_moves, root, model_v, num=54, depth=3):
    if board.piece_sum() < num:
        root_tmp = Node(board)
        create_tree(root_tmp, depth)
        root_tmp.backward(model_v)
        _,move = minimax_draw(root_tmp, root_tmp.board.turn, depth)
        root = None
    else:
        if root is None:
            root = Node(board)
            create_tree(root, 100)
            root.backward()
        _,move = minimax_draw(root, root.board.turn, 100)
    return move, root