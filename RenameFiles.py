"""
This file is used to rename video images for model training

@Author: Yun Wu
@Email: yunwustc@gmail.com
"""

from src.utils import rename_files
import os

def main():

    MOVES = ['Stay', 'Run_Left', 'Run_Right', 'Jump_Left', 'Jump_Right', 'Jump']
    path = os.path.join('img')
    start_index = 710 
    end_index = 780 
    name = 'J'
    move = MOVES[0] 
    rename_files(path, start_index, end_index, name, move)


if __name__ == '__main__':
    main()
