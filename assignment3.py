import json
import random

import numpy as np

EMPTY = 0
CROSSES = 1
CIRCLES = 2

ROWS = 3
COLUMNS = 3
DIAGONALS = 2


class Tree(object):
    def __init__(
        self,
        id: int,
        board: np.array = None,
        mc_move: int = -1,
        opponent_move: int = -1,
        count: int = 0,
        reward: int = 0,
    ):
        self.id = id
        self.board = board
        self.move = mc_move
        self.opponent_move = opponent_move
        self.count = count
        self.reward = reward
        self.child_id = []
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

    def tree_q(self):
        q = 0
        for c in self.children:
            q += c.reward / c.count
        return q

    # deze mag nog wel verbeterd worden zodat je de hele tree kan
    # printen ik heb nu beetje geprobeerd maar het lukte niet echt.
    def print_children(self):
        for c in self.children:
            print(
                "{} reward: {:>{width}}\t count: {:>{width}}".format(
                    c.move, c.reward, c.count, width=8
                )
            )
            print(c.move, c.opponent_move)
        print("printed children of: ", self.move)
        i = 0
        for cc in c.children:
            print(
                "{} reward: {:>{width}}\t count: {:>{width}}".format(
                    cc.move, cc.reward, cc.count, width=8
                )
            )
            i += 1
            # print(i)
        print("printed children of: ", c.move)

    def find_child(self, move, opponent_move):
        for c in self.children:
            if c.move == move and c.opponent_move == opponent_move:
                return 1
        return 0

    def find_best_move(self, opponent_move, board):
        max = 0
        move = -1
        for c in self.children:
            if c.opponent_move == opponent_move:
                # print(c.move, "reward: ",c.reward," count: ",c.count)
                if c.reward > 0 and c.count > 0:
                    if c.reward / c.count > max and move not in board:
                        max = c.reward / c.count
                        move = c.move
        # print("best move is: ",move," with q val: ", max)
        return move

    def walk_tree_add_reward(self, moves, reward):
        j = 0
        for move in moves:
            j += 1
            if j < len(moves):
                # print("walking: ",moves[j],move)
                for c in self.children:
                    if c.move == moves[j] and c.opponent_move == move:
                        # print("found node")
                        c.count += 1
                        c.reward += reward
                        self = c


class TicTacToe:
    converter = lambda _: {EMPTY: " ", CROSSES: "X", CIRCLES: "O"}[_]
    tree = Tree(
        id=4,
        board=np.array(
            (EMPTY, EMPTY, EMPTY, EMPTY, CROSSES, EMPTY, EMPTY, EMPTY, EMPTY)
        ),
    )

    def __init__(self) -> None:
        self.board = np.array(
            (EMPTY, EMPTY, EMPTY, EMPTY, CROSSES, EMPTY, EMPTY, EMPTY, EMPTY)
        )
        self.player_made_moves = []
        self.opponent_made_moves = [4]
        self.game_order = [4]
        self.total_tree = {}

    def winning(self, board: np.array):
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
            self.opponent_made_moves.append(move)
            self.game_order.append(move)
            board[move] = CROSSES
            return board
        else:
            raise KeyError("Board is full")

    def save_tree(
        self, filename: str = "tree", top_node: Tree = tree, init: bool = True
    ) -> None:
        def node_data(node: Tree) -> None:
            self.total_tree[node.id] = {
                "Q_val": node.tree_q(),
                "board": node.board.tolist(),
                "cnt": node.count,
                "rwd": node.reward
                # "child": node.child_id,
            }

        if init:
            init = False
            node_data(top_node)
        for child in top_node.children:
            node_data(child)
            self.save_tree(top_node=child, init=False)

        with open(f"{filename}.json", "w") as fp:
            json.dump(self.total_tree, fp)

    def get_Q_values(self, current_node: Tree, recursive: bool = True) -> list:
        """calculates the optimal strategy for play tic tac toe

        Args:
            node (Node, optional): node with data from previous state. Defaults to None.
            recursive (bool, optional): if this is main node. Defaults to True.

        Returns:
            list: IDK yet
        """
        # SLIDE 28 lecture 7
        termination, winner = self.winning(current_node.board)
        if termination:
            return 1 if winner == CIRCLES else 0
        elif len(self.possible_moves(current_node.board)) == 0:
            return 0.5

        new_board = (
            self.opponent_move(current_node.board) if recursive else self.board.copy()
        )

        moves = self.possible_moves(new_board)
        if len(moves) > 0:
            selected_node = random.choice(moves)
            # selected_node = current_node.find_best_move(self.opponent_made_moves[0],self.game_order)
            # print(current_node.find_best_move(self.opponent_made_moves[0]))
            # add newly found node. first check if it is already part of the children
            # check if the move is in the tree in the current state
            # if it is not in the tree add a new child node.
            self.game_order.append(selected_node)
            self.player_made_moves.append(selected_node)

            res = current_node.find_child(selected_node, self.opponent_made_moves[0])
            if res == 0:
                current_node.add_child(
                    Tree(
                        id=int("".join([str(i) for i in self.game_order])),
                        mc_move=selected_node,
                        opponent_move=self.opponent_made_moves[0],
                        count=1,
                        reward=0.01,
                    )
                )
            self.opponent_made_moves.remove(self.opponent_made_moves[0])
            # current_node.add_child(Tree(selected_node, 0))
            # laat de current_node wijzen naar de juiste node dus de child die net is toegevoegd.
            for c in current_node.children:
                if c.move == selected_node:
                    current_node = c
            new_board[selected_node] = CIRCLES
            current_node.board = new_board.copy()

        # print(self.print_board(new_board))
        return self.get_Q_values(current_node)

    def Q_convergence(self, epsilon=0.000001) -> None:
        """Simulates the Game until convergence is achieved

        Args:
            epsilon (float, optional): max difference between iterations. Defaults to 0.00000001.
        """
        current_node = self.tree
        i = 0
        j = 0
        old_q = 0
        new_q = 1
        difference = 1
        while difference > epsilon:
            self.__init__()
            current_node = self.tree
            old_q = self.tree.tree_q()
            # met de reward kunnen we een counter toevoegen aan hoe vaak een bepaalde node in de tree gewonnen heeft.
            # hiervoor moeten we een tree walker maken zodat we de juiste stappen kunnen nemen om bij de juiste nodes de counter te incrementen.
            reward = self.get_Q_values(current_node, recursive=False)
            # nu zetten we current_node weer terug naar de start node zodat de eerste stap die gemaakt wordt wordt toegevoegd aan de eerste node.
            # print(self.game_order,reward)
            if reward > 0:
                i += 1
            self.tree.walk_tree_add_reward(self.game_order, reward)
            new_q = self.tree.tree_q()
            # print(old_q,new_q)
            difference = abs(old_q - new_q)
            if j % 10000 == 0:
                print(difference)
            j += 1
        # print alle nodes in de tree nog niet heel mooi als je helemaal naar boven scrolled zie je de eerste node die heeft 8 children dit zijn de 8 eerste moves.
        # heb voor nu uitgezet dat alle nodes worden geprint alleen de eerste node met ze children word geprint en dat zijn alle eerste moves die gemaakt zijn.
        # als je het weer aan wilt zetten kan je dat doen in de functie print_children door de commented lines te uncommenten.
        current_node.print_children()
        print("difference: ", difference)
        print("totalgames: ", j, " games won: ", i)

    def get_Q_valuess(self, current_node, recursive: bool = True) -> list:
        """calculates the optimal strategy for play tic tac toe

        Args:
            node (Node, optional): node with data from previous state. Defaults to None.
            recursive (bool, optional): if this is main node. Defaults to True.

        Returns:
            list: IDK yet
        """
        # SLIDE 28 lecture 7
        termination, winner = self.winning(current_node.board)
        if termination:
            return 1 if winner == CIRCLES else 0
        elif len(self.possible_moves(current_node.board)) == 0:
            return 0.5

        new_board = (
            self.opponent_move(current_node.board) if recursive else self.board.copy()
        )

        moves = self.possible_moves(new_board)
        if len(moves) > 0:
            # selected_node = random.choice(moves)
            selected_node = current_node.find_best_move(
                self.opponent_made_moves[0], self.game_order
            )
            if selected_node == -1:
                selected_node = random.choice(moves)
                print(selected_node)
            # print(current_node.find_best_move(self.opponent_made_moves[0]))
            # add newly found node. first check if it is already part of the children
            # check if the move is in the tree in the current state
            # if it is not in the tree add a new child node.
            res = current_node.find_child(selected_node, self.opponent_made_moves[0])
            if res == 0:
                # print("NO")
                current_node.add_child(
                    Tree(
                        id=int("".join([str(i) for i in self.game_order])),
                        mc_move=selected_node,
                        opponent_move=self.opponent_made_moves[0],
                        count=1,
                        reward=0.01,
                    )
                )
            self.opponent_made_moves.remove(self.opponent_made_moves[0])
            # current_node.add_child(Tree(selected_node, 0))

            self.player_made_moves.append(selected_node)
            self.game_order.append(selected_node)
            # laat de current_node wijzen naar de juiste node dus de child die net is toegevoegd.
            for c in current_node.children:
                if c.move == selected_node:
                    current_node = c
            new_board[selected_node] = CIRCLES
            current_node.board = new_board.copy()

        # print(self.print_board(new_board))
        return self.get_Q_valuess(current_node)

    def new_ai_game(self) -> None:
        i = 0
        j = 0
        for _ in range(10000):
            self.__init__()
            current_node = self.tree
            # met de reward kunnen we een counter toevoegen aan hoe vaak een bepaalde node in de tree gewonnen heeft.
            # hiervoor moeten we een tree walker maken zodat we de juiste stappen kunnen nemen om bij de juiste nodes de counter te incrementen.
            reward = self.get_Q_valuess(current_node, recursive=False)
            # nu zetten we current_node weer terug naar de start node zodat de eerste stap die gemaakt wordt wordt toegevoegd aan de eerste node.
            # print(self.game_order,reward)
            if reward == 1:
                i += 1
            if reward == 0.5:
                j += 1
            self.tree.walk_tree_add_reward(self.game_order, reward)

        # print alle nodes in de tree nog niet heel mooi als je helemaal naar boven scrolled zie je de eerste node die heeft 8 children dit zijn de 8 eerste moves.
        # heb voor nu uitgezet dat alle nodes worden geprint alleen de eerste node met ze children word geprint en dat zijn alle eerste moves die gemaakt zijn.
        # als je het weer aan wilt zetten kan je dat doen in de functie print_children door de commented lines te uncommenten.
        # current_node.print_children()
        print("totalgame: 10,000 games won: ", i, "draws: ", j)


if __name__ == "__main__":
    ttt = TicTacToe()
    ttt.Q_convergence()
    ttt.new_ai_game()
    ttt.save_tree()
