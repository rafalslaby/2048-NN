from tkinter import *

from collections import defaultdict
import queue
import numpy as np


class GUI2048(Frame):
    SIZE = 500
    GRID_LEN = 4
    GRID_PADDING = 10

    BACKGROUND_COLOR_GAME = "#92877d"
    BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"
    BACKGROUND_COLOR_DICT = {2: "#eee4da", 4: "#ede0c8", 8: "#f2b179", 16: "#f59563",
                             32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72", 256: "#edcc61",
                             512: "#edc850", 1024: "#edc53f", 2048: "#edc22e"}
    BACKGROUND_COLOR_DICT = defaultdict(lambda: "#edc22e", BACKGROUND_COLOR_DICT)
    CELL_COLOR_DICT = {2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
                       32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
                       512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2"}

    CELL_COLOR_DICT = defaultdict(lambda: "#f9f6f2", CELL_COLOR_DICT)
    FONT = ("Verdana", 40, "bold")

    KEY_UP_ALT = "\'\\uf700\'"
    KEY_DOWN_ALT = "\'\\uf701\'"
    KEY_LEFT_ALT = "\'\\uf702\'"
    KEY_RIGHT_ALT = "\'\\uf703\'"

    KEY_UP = "'w'"
    KEY_DOWN = "'s'"
    KEY_LEFT = "'a'"
    KEY_RIGHT = "'d'"

    def __init__(self, state_queue, update_fps=100):
        """
        Freezes the thread in mainloop. Periodically each update_fps checks for new
        (state, wait_after_ms) tuples in state_queue and updates gui board with state
        """
        Frame.__init__(self)
        self.state_queue = state_queue
        self.update_fps = update_fps

        self.grid()
        self.master.title('2048')

        self.grid_cells = []
        self.matrix = np.zeros((4, 4))
        self.init_grid()
        self.update_grid_cells()
        self.update_state()
        self.mainloop()

    def init_grid(self):
        background = Frame(self, bg=GUI2048.BACKGROUND_COLOR_GAME, width=GUI2048.SIZE, height=GUI2048.SIZE)
        background.grid()
        for i in range(GUI2048.GRID_LEN):
            grid_row = []
            for j in range(GUI2048.GRID_LEN):
                cell = Frame(background, bg=GUI2048.BACKGROUND_COLOR_CELL_EMPTY, width=GUI2048.SIZE / GUI2048.GRID_LEN,
                             height=GUI2048.SIZE / GUI2048.GRID_LEN)
                cell.grid(row=i, column=j, padx=GUI2048.GRID_PADDING, pady=GUI2048.GRID_PADDING)
                t = Label(master=cell, text="", bg=GUI2048.BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER,
                          font=GUI2048.FONT, width=4, height=2)
                t.grid()
                grid_row.append(t)

            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        for i in range(GUI2048.GRID_LEN):
            for j in range(GUI2048.GRID_LEN):
                new_number = self.matrix[i][j]
                if new_number == 0:
                    self.grid_cells[i][j].configure(text="", bg=GUI2048.BACKGROUND_COLOR_CELL_EMPTY)
                else:
                    self.grid_cells[i][j].configure(text=str(new_number), bg=GUI2048.BACKGROUND_COLOR_DICT[new_number],
                                                    fg=GUI2048.CELL_COLOR_DICT[new_number])
        self.update_idletasks()

    def update_state(self):
        wait_after_ms = None
        try:
            state, wait_after_ms = self.state_queue.get_nowait()
            self.matrix = state
            self.update_grid_cells()
            self.state_queue.task_done()
        except queue.Empty:
            pass
        finally:
            self.after(wait_after_ms or self.update_fps, self.update_state)
