import random

import numpy as np

EMPTY = 0
CROSSES = 1
CIRCLES = 2

ROWS = 3
COLUMNS = 3
DIAGONALS = 2


class Node(object):
    def __init__(self, board: np.array, moves: list) -> None:
        self.board = board
        self.moves = moves


class Tree(object):
    def __init__(self, data: int = -1, count: int = 0, reward: int = 0):
        self.move = data
        self.count = count
        self.reward = reward
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

    # deze mag nog wel verbeterd worden zodat je de hele tree kan
    # printen ik heb nu beetje geprobeerd maar het lukte niet echt.
    def print_children(self):
        for c in self.children:
            print(
                "{} reward: {:>{width}}\t count: {:>{width}}".format(
                    c.move, c.reward, c.count, width=8
                )
            )
        print("printed children of: ", self.move)

    def find_child(self, move):
        for c in self.children:
            if c.move == move:
                return 1
        return 0

    def walk_tree_add_reward(self, moves, reward):
        for move in moves:
            for c in self.children:
                if c.move == move:
                    c.count += 1
                    c.reward += reward
                    self = c


class TicTacToe:
    converter = lambda _: {EMPTY: " ", CROSSES: "X", CIRCLES: "O"}[_]

    def __init__(self) -> None:
        self.board = np.array(
            (EMPTY, EMPTY, EMPTY, EMPTY, CROSSES, EMPTY, EMPTY, EMPTY, EMPTY)
        )
        self.player_made_moves = []

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
            move = random.choice(possible_moves)
            board[move] = CROSSES
            return board
        else:
            raise KeyError("Board is full")

    def get_Q_values(
        self, current_node, node: Node = None, recursive: bool = True
    ) -> list:
        """calculates the optimal strategy for play tic tac toe

        Args:
            node (Node, optional): node with data from previous state. Defaults to None.
            recursive (bool, optional): if this is main node. Defaults to True.

        Returns:
            list: IDK yet
        """
        # SLIDE 28 lecture 7
        termination, winner = self.winning(node.board)
        if termination:
            return 1 if winner == CIRCLES else 0
        elif len(self.possible_moves(node.board)) == 0:
            return 0.5

        new_board = self.opponent_move(node.board) if recursive else self.board.copy()

        moves = self.possible_moves(new_board)
        if len(moves) > 0:
            selected_node = random.choice(moves)
            # add newly found node. first check if it is already part of the children
            # check if the move is in the tree in the current state
            # if it is not in the tree add a new child node.
            res = current_node.find_child(selected_node)
            if res == 0:
                current_node.add_child(Tree(selected_node, 0))

            self.player_made_moves.append(selected_node)
            # laat de current_node wijzen naar de juiste node dus de child die net is toegevoegd.
            for c in current_node.children:
                if c.move == selected_node:
                    current_node = c
            new_board[selected_node] = CIRCLES
        node_prime = Node(new_board.copy(), self.possible_moves(new_board))

        # print(self.print_board(new_board))
        return self.get_Q_values(current_node, node_prime)

    def Q_convergence(self, epsilon=0.001) -> None:
        """Simulates the Game until convergence is achieved

        Args:
            epsilon (float, optional): max difference between iterations. Defaults to 0.001.
        """
        tree = Tree()
        current_node = tree

        for _ in range(5000):
            self.__init__()
            # met de reward kunnen we een counter toevoegen aan hoe vaak een bepaalde node in de tree gewonnen heeft.
            # hiervoor moeten we een tree walker maken zodat we de juiste stappen kunnen nemen om bij de juiste nodes de counter te incrementen.
            reward = self.get_Q_values(
                current_node, node=Node(ttt.board, []), recursive=False
            )
            # nu zetten we current_node weer terug naar de start node zodat de eerste stap die gemaakt wordt wordt toegevoegd aan de eerste node.
            current_node = tree
            # if reward == 1:
            #     print(reward)
            # print(player_made_moves)
            tree.walk_tree_add_reward(self.player_made_moves, reward)

        # print alle nodes in de tree nog niet heel mooi als je helemaal naar boven scrolled zie je de eerste node die heeft 8 children dit zijn de 8 eerste moves.
        # heb voor nu uitgezet dat alle nodes worden geprint alleen de eerste node met ze children word geprint en dat zijn alle eerste moves die gemaakt zijn.
        # als je het weer aan wilt zetten kan je dat doen in de functie print_children door de commented lines te uncommenten.
        current_node.print_children()


if __name__ == "__main__":
    ttt = TicTacToe()
    ttt.Q_convergence()
