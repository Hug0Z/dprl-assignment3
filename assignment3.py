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
            _ = board[i] == board[i + 1] == board[i + 2]
            if _:
                return True, board[i]

        # COLUMNS
        for j in range(COLUMNS):
            _ = board[j] == board[j + 3] == board[j + 6]
            if _:
                return True, board[j]

        # DIAGONALS (start top left to bottom right, then top right to bottom left)
        for k in range(DIAGONALS):
            _ = board[0 if k == 0 else 3] == board[4] == board[8 if k == 0 else 6]
            if _:
                return True, board[k]

        # No winner yet
        return False, None

    def winning_circle(self, board: np.array) -> int:
        """Create the environment along with the terminal conditions
            where the reward is 1 for the winner and 0 for the loser.

        Returns:
            int: winning = 1, losing = 0
        """
        is_finished, winner = self.winning(board)

        if is_finished:
            return 1 if winner == CIRCLES else 0

    def print_board(self) -> None:
        """prints the current board in a 3x3 grid, instead of a 9 len array"""
        data = np.reshape(self.board, (ROWS, COLUMNS))

        for i, row in enumerate(data):
            print(
                f" {self.converter(row[0])} | {self.converter(row[1])} | {self.converter(row[2])}"
            )
            if i != len(data) - 1:
                print("---+---+---")

    def possible_moves(self, board: np.array) -> list:
        """returns a list with all indexes of possible moves

        Returns:
            list: list with locations of all possible moves
        """
        return [i for i in range(len(board)) if board[i] == EMPTY]

    def opponent_move(self) -> None:
        """Does an opponents move at an available slot"""
        possible_moves = self.possible_moves()
        if len(possible_moves) != 0:
            self.board[random.choice(possible_moves)] = CROSSES
        else:
            raise KeyError("Board is full")

    def get_Q_values(self, node: Node) -> list:
        # SLIDE 28 lecture 7
        termination, winner = self.winning(node)
        if termination:
            return 1 if winner == CIRCLES else 0

        actions = []
        value = None
        for a in actions:
            node_prime = Node(node.board, self.possible_moves(node.board))
            value = (sum(self.get_Q_values(node_prime))) / len(actions)
        self.board = node_prime.board
        return value


if __name__ == "__main__":
    ttt = TicTacToe()
    ttt.print_board()
