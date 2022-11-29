import random

import numpy as np

EMPTY = 0
CROSSES = 1
CIRCLES = 2

ROWS = 3
COLUMNS = 3
DIAGONALS = 2


class Node:
    def __init__(self, board: np.array, moves: list) -> None:
        self.board = board
        self.moves = moves


class TicTacToe:
    def __init__(self) -> None:
        self.board = np.array(
            (EMPTY, EMPTY, EMPTY, EMPTY, CROSSES, EMPTY, EMPTY, EMPTY, EMPTY)
        )
        self.converter = lambda _: {EMPTY: " ", CROSSES: "X", CIRCLES: "O"}[_]

    def winning(self, board: np.array) -> tuple[bool, int]:
        """Returns the winner of a game

        Args:
            board (np.array): board state

        Returns:
            tuple[bool, int]: boolean if there is a winner, int corresponding to the winner symbol
        """

        # ROWS
        for i in range(ROWS):
            if board[i] == board[i + 1] == board[i + 2] and board[i] != EMPTY:
                return True, board[i]

        # COLUMNS
        for j in range(COLUMNS):
            if board[j] == board[j + 3] == board[j + 6] and board[j] != EMPTY:
                return True, board[j]

        # DIAGONALS (start top left to bottom right, then top right to bottom left)
        for k in range(DIAGONALS):
            if board[0 if k == 0 else 3] == board[4] == board[8 if k == 0 else 6]:
                return True, board[k]

        # No winner yet
        return False, None

    def winning_circle(self, board: np.array) -> int:
        """Create the environment along with the terminal conditions
            where the reward is 1 for the winner and 0 for the loser.

        Args:
            board (np.array): current state of board

        Returns:
            int: winning = 1, losing = 0
        """
        is_finished, winner = self.winning(board)

        if is_finished:
            return 1 if winner == CIRCLES else 0

    def print_board(self, board: np.array) -> None:
        """prints the current board in a 3x3 grid, instead of a 9 len array

        Args:
            board (np.array): board state you want to print
        """
        data = np.reshape(board, (ROWS, COLUMNS))

        for i, row in enumerate(data):
            print(
                f" {self.converter(row[0])} | {self.converter(row[1])} | {self.converter(row[2])}"
            )
            if i != len(data) - 1:
                print("---+---+---")

    def possible_moves(self, board: np.array) -> list:
        """returns a list with all indexes of possible moves

        Args:
            board (np.array): current board state

        Returns:
            list: list with locations of all possible moves
        """
        return [i for i in range(len(board)) if board[i] == EMPTY]

    def opponent_move(self, board: np.array) -> np.array:
        """Does an opponents move at an available slot

        Args:
            board (np.array): current state of board

        Raises:
            KeyError: if board is full

        Returns:
            np.array: new state of board
        """
        possible_moves = self.possible_moves(board)
        if len(possible_moves) != 0:
            board[random.choice(possible_moves)] = CROSSES
            return board
        else:
            raise KeyError("Board is full")

    def get_Q_values(self, node: Node = None, recursive: bool = True) -> list:
        """calculates the optimal strategy for play tic tac toe

        Args:
            node (Node, optional): node with data from previous state. Defaults to None.
            recursive (bool, optional): if this is main node. Defaults to True.

        Returns:
            list: IDK yet
        """
        # SLIDE 28 lecture 7
        termination, winner = self.winning(node.board)
        draw = len(self.possible_moves(node.board)) == 0
        if termination or draw:
            return 1 if winner == CIRCLES else 0

        new_board = self.opponent_move(node.board) if recursive else self.board
        moves = self.possible_moves(new_board)
        if len(moves) > 0:
            new_board[random.choice(moves)] = CIRCLES
        node_prime = Node(new_board, self.possible_moves(new_board))

        print(self.print_board(new_board))
        return self.get_Q_values(node_prime)


if __name__ == "__main__":
    ttt = TicTacToe()
    print(ttt.get_Q_values(node=Node(ttt.board, []), recursive=False))
