import json
import random

import numpy as np

EMPTY = 0
CROSSES = 1
CIRCLES = 2

ROWS = 3
COLUMNS = 3
DIAGONALS = 2

WIN = 1
DRAW = 0.5
LOSS = 0


class Tree(object):
    def __init__(
        self,
        id: int,
        parent: None,
        board: np.array = None,
        mc_move: int = -1,
        opponent_move: int = -1,
        count: int = 0,
        reward: int = 0,
    ):
        self.id = id
        self.board = board
        self.parent = parent
        self.move = mc_move
        self.opponent_move = opponent_move
        self.count = count
        self.reward = reward
        self.child_id = []
        self.children = []
        self.game_order = []

    def add_child(self, obj):
        self.children.append(obj)
        self.child_id.append(obj.id)

    def tree_q(self):
        try:
            q = 0
            for c in self.children:
                q += c.reward / c.count
            return q
        except:
            return 0

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

    def rollback(self, reward):
        while self.parent != None:
            #   print("rollback", self.move, self.opponent_move)
            self.count += 1
            self.reward += reward
            self = self.parent


class TicTacToe:
    converter = lambda _: {EMPTY: " ", CROSSES: "X", CIRCLES: "O"}[_]
    tree = Tree(
        id=4,
        parent=None,
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
        for i in range(0, ROWS * DIAGONALS, ROWS):
            if board[i] == board[i + 1] == board[i + 2] and board[i] != EMPTY:
                return True, board[i]

        # COLUMNS
        for j in range(COLUMNS):
            if board[j] == board[j + 3] == board[j + 6] and board[j] != EMPTY:
                return True, board[j]

        # DIAGONALS (start top left to bottom right, then top right to bottom left)
        for k in range(DIAGONALS):
            if board[0 if k == 0 else 2] == board[4] == board[8 if k == 0 else 6]:
                return True, board[k]

        # No winner yet
        return False, None

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
                "rwd": node.reward,
                "childs": node.child_id
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

    def selection(self, node: Tree) -> Tree:
        """select existing node

        Args:
            node (Tree): current tree (top node)

        Returns:
            Tree: returns selected node
        """
        if random.choice([True, False]):
            return node
        else:
            next_nodes = node.children
            if len(next_nodes) == 0:
                return node
            else:
                return self.selection(random.choice(next_nodes))

    def expansion(self, current_node: Tree, board: np.array) -> list:
        """finds all possible children and returns them in a list

        Args:
            current_node (Tree): selected node
            board (np.array): current board state

        Returns:
            list: list of children
        """
        children = []
        for move in self.possible_moves(board):
            # Copy data
            child_board = board.copy()
            # child_moves = game_moves.copy()

            # Adjust for child
            child_board[move] = CIRCLES
            for opponent in self.possible_moves(child_board):
                child = None
                for c in current_node.children:
                    if c.id == int(
                        str(current_node.id) + str(move) + str(opponent)
                    ):  # Exists
                        child = c
                        break
                if child is None:  # Add
                    new_board = child_board.copy()
                    new_board[opponent] = CROSSES
                    child = Tree(
                        id=int(str(current_node.id) + str(move) + str(opponent)),
                        parent=current_node,
                        board=new_board,
                        mc_move=move,
                        opponent_move=opponent
                        if current_node.parent is not None
                        else 4,
                    )
                    current_node.add_child(child)
                children.append(child)
        return children

    def terminating(self, board: np.array) -> float:
        """Returns result of terminating node

        Args:
            board (np.array): current board state

        Returns:
            float: reward (not terminating is None)
        """
        t, w = self.winning(board)
        if t:
            return WIN if w == CIRCLES else LOSS
        elif len(self.possible_moves(board)) == 0:
            return DRAW
        else:
            return None

    def rollout(self, node: Tree) -> float:
        """does random moves till node terminates

        Args:
            node (Tree): leaf node

        Returns:
            float: reward of game
        """
        rollout_board = node.board.copy()
        player = CROSSES
        while True:
            reward = self.terminating(rollout_board)
            if reward is not None:
                return reward

            pos_moves = self.possible_moves(rollout_board)
            move = random.choice(pos_moves)
            rollout_board[move] = player
            player = CIRCLES if player == CROSSES else CROSSES  # flip player

    def Q_learning(self, current_node: Tree, recursive: bool = True) -> float:
        reward = self.terminating(current_node.board)
        if reward is not None:
            current_node.rollback(reward)
            return reward
        else:
            new_board = (
                self.opponent_move(current_node.board)
                if recursive
                else self.board.copy()
            )

            moves = self.possible_moves(new_board)
            if len(moves) > 0:
                selected_node = random.choice(moves)

                new_board[selected_node] = CIRCLES
                print(self.opponent_made_moves)
                res = current_node.find_child(
                    selected_node, self.opponent_made_moves[0]
                )
                if res == 0 and current_node.move != -1:
                    current_node.add_child(
                        Tree(
                            current_node,
                            new_board,
                            selected_node,
                            self.opponent_made_moves[0],
                            1,
                            0.001,
                        )
                    )
                print(self.opponent_made_moves)
                self.opponent_made_moves.remove(self.opponent_made_moves[0])

                self.game_order.append(selected_node)
                for c in current_node.children:
                    if c.move == selected_node:
                        current_node = c

            return self.Q_learning(current_node)

    def Q_convergence(self, epsilon: float = 0.000000001) -> None:
        """gets Q convergence

        Args:
            epsilon (float, optional): max difference in q values. Defaults to 0.000000001.
        """
        old_q, new_q = 0, 1
        game_counter = 0
        difference = 1
        wins, draws, loss = 0, 0, 0
        while difference > epsilon:
            self.__init__()
            # Selection
            selected_node = self.selection(self.tree)
            # old_q = self.tree.tree_q()

            # Expansion
            for leaf in self.expansion(selected_node, selected_node.board.copy()):
                self.board = leaf.board.copy()
                self.game_order = leaf.game_order.copy()
                old_q = self.tree.tree_q()
                reward = self.Q_learning(leaf, recursive=False)
                # reward = self.rollout(child)
                if reward == WIN:
                    wins += 1
                elif reward == DRAW:
                    draws += 1
                else:
                    loss += 1
                # child.rollback(reward)

                # Check convergence
                new_q = self.tree.tree_q()
                difference = abs(old_q - new_q)
                if game_counter % 10000 == 0:
                    print(game_counter, difference)
                game_counter += 1
        print(game_counter)

        # self.tree.print_children()
        print(f"difference: {difference}")
        print(f"total games: {game_counter} win: {wins} draw: {draws}, loss: {loss}")

    def Q_sim(self, current_node: Tree, recursive: bool = True) -> float:
        """calculates the optimal strategy for play tic tac toe

        Args:
            node (Node, optional): node with data from previous state. Defaults to None.
            recursive (bool, optional): if this is main node. Defaults to True.

        Returns:
            float: reward
        """
        # SLIDE 28 lecture 7
        termination, winner = self.winning(current_node.board)
        if termination:
            if winner == CIRCLES:
                ret = 1
            else:
                ret = 0
            current_node.rollback(ret)
            return ret
        elif len(self.possible_moves(current_node.board)) == 0:
            current_node.rollback(0.5)
            return 0.5

        new_board = (
            self.opponent_move(current_node.board) if recursive else self.board.copy()
        )

        moves = self.possible_moves(new_board)
        if len(moves) > 0:
            selected_node = current_node.find_best_move(
                self.opponent_made_moves[0], self.game_order
            )
            # if we havent found the move before we generate a new one
            if selected_node == -1:
                selected_node = random.choice(moves)
            self.opponent_made_moves.remove(self.opponent_made_moves[0])

            self.game_order.append(selected_node)
            # laat de current_node wijzen naar de juiste node dus de child die net is toegevoegd.
            for c in current_node.children:
                if c.move == selected_node:
                    current_node = c
            new_board[selected_node] = CIRCLES
        return self.Q_sim(current_node)

    def new_ai_game(self) -> None:
        wins, draws, loss = 0, 0, 0
        for _ in range(10000):
            self.__init__()
            current_node = self.tree
            reward = self.Q_sim(current_node, recursive=False)
            if reward == WIN:
                wins += 1
            elif reward == DRAW:
                draws += 1
            else:
                loss += 1

        print(
            "total game: 10,000 games won: ", wins, "draws: ", draws, "losses: ", loss
        )


if __name__ == "__main__":
    ttt = TicTacToe()
    ttt.Q_convergence()
    ttt.new_ai_game()
    # ttt.save_tree()
