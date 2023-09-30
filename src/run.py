# input
## present board [string] : black(user)='X', white(AI)='O', empty='-'
## user's move [string] : 'a1', 'a2', ..., 'h8', 'pass'

# output
## AI's move [string] : 'a1', 'a2', ..., 'h8', 'pass', 'game over'
## board evaluation [int] : -64 ~ 64 (positive value means black is better)

import sys
sys.path.append('../')
from creversi import *
from TreeSearch import *
from model.networks import *


def main():
    line = input('present board :')
    assert len(line) == 64, 'length of "present board" should be 64.'
    board = Board()
    board.set_line(line,True)
    legal_moves = list(board.legal_moves)

    move_str = input("user's move :")
    assert move_str in ['pass'] + [chr(ord('a')+i)+str(j+1) for i in range(8) for j in range(8)], 'invalid character.'
    move = move_from_str(move_str)
    assert move in legal_moves, 'cannot move.'

    board.move(move)
    legal_moves = list(board.legal_moves)

    if board.is_game_over():
        print('game over')
        print(board.diff_num() if board.turn else -board.diff_num())
        return

    model_v = ValueNetwork()
    model_v.load_state_dict(torch.load('../model/value-network-v4.pth'))
    model_v.eval()

    if 64 in legal_moves:
        move = 64
    elif len(legal_moves) == 1:
        move = legal_moves[0]
    else:
        move, _ = draw_ai(board, legal_moves, None, model_v, 53, 4)
        # for child in _.children:
        #     print(child.data)

    board.move(move)
    print(move_to_str(move))

    with torch.no_grad():
        z = model_v(board_to_array_aug2(board,True)).mean().item()*64
    print(int(z))
    return
    

if __name__ == '__main__':
    main()
    

# test case
"""
present board :O-OXXXXOOXXOOXXOOXOOXXOOOOXXXOX-OOXXOOX-OOXOXXXXOXOOXXX-XXXXXXX-
user's move :b1
[2, 10, 2, 10]
[4, 14, 4, 14]
[2, 14, -2, 14]
[4, 14, 4, 14]
h4
3
"""