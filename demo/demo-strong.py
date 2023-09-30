try:
    import sys
    sys.path.append('../')
    import tkinter
    import torch
    from model.networks import *
    import numpy as np

    from GUI import *

    class Othello(OthelloMaster):
        def __init__(self, master):
            super().__init__(master)

        def com(self):
            '''COMに石を置かせる'''

            # 石が置けるマスを取得
            legal_moves = list(self.board_inner.legal_moves)
            
            # 最初のマスを次に石を置くマスとする
            with torch.no_grad():
                output = self.model_pol(board_to_array(self.board_inner,True).unsqueeze(0)).numpy()
            prob_legal = output[0][legal_moves]
            move = legal_moves[np.argmax(prob_legal)]
            x, y = move%8, move//8
            
            # 石を置く
            self.place(x, y, COM_COLOR)

    # スクリプト処理ここから
    app = tkinter.Tk()
    app.title('othello')
    othello = Othello(app)
    app.mainloop()

except Exception as e:
    print(e)
    import traceback
    traceback.print_exc()
    input()