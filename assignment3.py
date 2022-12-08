import random

import numpy as np

EMPTY = 0
CROSSES = 1
CIRCLES = 2

ROWS = 3
COLUMNS = 3
DIAGONALS = 2
data_viz_output = ""

class Tree(object):
    def __init__(
        self,
        parent: None,
        board: np.array,
        mc_move: int = -1,
        opponent_move: int = -1,
        count: int = 1,
        reward: int = 0.01,
    ):
        self.move = mc_move
        self.opponent_move = opponent_move
        self.parent = parent
        self.board = board
        self.count = count
        self.reward = reward
        self.children = []
        self.game_order = []

    def add_child(self, obj):
        self.children.append(obj)

    def tree_q(self):
        q = 0
        for c in self.children:
            q += c.reward / c.count
        return q

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
            print(cc.game_order)
        print("printed children of: ", c.move)

    def find_child(self, move, opponent_move):
        for c in self.children:
            if c.move == move and c.opponent_move == opponent_move:
                return 1
        return 0

    def get_child(self, move, opponent_move):
        for c in self.children:
            if c.move == move and c.opponent_move == opponent_move:
                return c
        return 0

    def find_best_move(self, opponent_move, board, printing):
        global data_viz_output
        max = 0
        move = -1
        if printing:
            data_viz_output += str(self.board) + "\n"
        for c in self.children:
            if c.opponent_move == opponent_move and c.move != opponent_move:
                if c.reward > 0 and c.count > 0:
                    if printing:
                        data_viz_output += f"Q val: {c.reward / c.count} mv: {c.move} op move: {opponent_move}\n"
                    if c.reward / c.count > max and move not in board and move not in self.game_order:
                        max = c.reward / c.count
                        move = c.move
        if printing:
            data_viz_output += f"Q val: {max} mv: {move} op move: {opponent_move}\n\n" 
        return move

    def back_propegation(self, reward):
        while self.parent != None:
            self.count += 1
            self.reward += reward
            self = self.parent


class TicTacToe:
    tree = Tree(
        None,
        np.array((EMPTY, EMPTY, EMPTY, EMPTY, CROSSES, EMPTY, EMPTY, EMPTY, EMPTY)),
    )
    converter = lambda _: {EMPTY: " ", CROSSES: "X", CIRCLES: "O"}[_]

    def __init__(self) -> None:
        self.board = np.array(
            (EMPTY, EMPTY, EMPTY, EMPTY, CROSSES, EMPTY, EMPTY, EMPTY, EMPTY)
        )
        self.player_made_moves = []
        self.opponent_made_moves = [4]
        self.game_order = [4]

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

    def print_board(self, board: np.array):
        """prints the current board in a 3x3 grid, instead of a 9 len array

        Args:
            board (np.array): board state you want to print
        """
        mc_move = np.reshape(board, (ROWS, COLUMNS))

        for i, row in enumerate(mc_move):
            print(
                f" {self.converter(row[0])} | {self.converter(row[1])} | {self.converter(row[2])}"
            )
            if i != len(mc_move) - 1:
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

    def selection(self, node: Tree) -> Tree:
        """Selects random existing node

        Args:
            node (Tree): top node

        Returns:
            Tree: selected node
        """
        if len(node.children) == 0 and node.move == -1:
            print("beginning!")
            return node
        if random.choice([True, False]) and node.move != -1:
            return node
        else:
            next_nodes = node.children
            if len(next_nodes) == 0:
                return node
            else:
                return self.selection(random.choice(next_nodes))
                

    def expansion(self, current_node: Tree, board: np.array) -> list:
        """Finds all childs of selected node

        Args:
            current_node (Tree): selected node
            board (np.array): current state of the board

        Returns:
            list: list with all leafs of node
        """
        children = []
        for opponent in self.possible_moves(board):
            if opponent in current_node.game_order:
                continue
            if current_node.move == -1:
                opponent = 4
            # Copy mc_move
            child_board = board.copy()
            # child_moves = game_moves.copy()

            # Adjust for child
            child_board[opponent] = CROSSES

            for move in self.possible_moves(child_board):
                if current_node.find_child(move, opponent) == 0:
                    child_board2 = child_board.copy()
                    child_board2[move] = CIRCLES
                    self.opponent_made_moves.append(opponent)
                    child = Tree(current_node, child_board2, move, opponent, 1, 0.001)
                    for past_move in current_node.game_order:
                        child.game_order.append(past_move)
                    child.game_order.append(opponent)
                    child.game_order.append(move)
                    current_node.add_child(child)
                    children.append(child)
                else:
                  self.opponent_made_moves.append(opponent)
                  child = current_node.get_child(move,opponent)
                  children.append(child)
        return children

    def get_Q_values(
        self, current_node: Tree, board: np.array, recursive: bool = True
    ) -> float:
        """calculates the optimal strategy for play tic tac toe

        Args:
            node (Node, optional): node with mc_move from previous state. Defaults to None.
            recursive (bool, optional): if this is main node. Defaults to True.

        Returns:
            float: q reward
        """
        termination, winner = self.winning(board)
        if termination:
            if winner == CIRCLES:
                ret = 1
            else:
                ret = 0
            #current_node.rollback(ret)
            return ret
        elif len(self.possible_moves(board)) == 0:
            #current_node.rollback(0.5)
            return 0.5

        new_board = self.opponent_move(board) if recursive else self.board.copy()

        moves = self.possible_moves(new_board)
        if len(moves) > 0:
            selected_node = random.choice(moves)

            # add newly found node. first check if it is already part of the children
            # check if the move is in the tree in the current state
            # if it is not in the tree add a new child node.
            new_board[selected_node] = CIRCLES
            res = current_node.find_child(selected_node, self.opponent_made_moves[0])
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
            self.opponent_made_moves.remove(self.opponent_made_moves[0])

            self.game_order.append(selected_node)
            for c in current_node.children:
                if c.move == selected_node:
                    current_node = c
        return self.get_Q_values(current_node, new_board.copy())

    def Q_convergence(self, epsilon: float = 0.000001) -> None:
        """Simulates the Game until convergence is achieved

        Args:
            epsilon (float, optional): max difference between iterations. Defaults to 0.001.
        """
        i = 0
        j = 0
        old_q = 0
        new_q = 1
        difference = 1
        data = []
        while difference > epsilon:
            self.__init__()
            current_node = self.tree
            old_q = self.tree.tree_q()
            for _ in range(1000):
                current_node = self.tree
                current_node = self.selection(self.tree)
                for leaf in self.expansion(current_node, current_node.board):
                    prop_node = leaf
                    ttt.board = leaf.board.copy()
                    ttt.game_order = leaf.game_order.copy()
                    old_q = self.tree.tree_q()
                    if len(ttt.board) > 9:
                        print(len(ttt.board))

                    reward = self.get_Q_values(leaf, self.board.copy(), recursive=False)
                    prop_node.back_propegation(reward)
                    #print(self.game_order)
                    if reward > 0:
                        i += 1
                    new_q = self.tree.tree_q()
                    difference = abs(old_q - new_q)
                    if j % 9000 == 0:
                        print(difference, i)
                    j += 1
                    data.append([j, difference])
        np.savetxt("conv2.csv", data, delimiter=",")
        self.tree.print_children()
        print("difference: ", difference)
        print("total games: ", j, " games won: ", i)

    def q_sim(
        self, current_node: Tree, board: np.array, printing, recursive: bool = True,
    ) -> float:
        """simulates the q value function

        Args:
            node (Node, optional): node with mc_move from previous state. Defaults to None.
            recursive (bool, optional): if this is main node. Defaults to True.

        Returns:
            float: q reward
        """
        # SLIDE 28 lecture 7
        #print(self.game_order)
        termination, winner = self.winning(board)
        if termination:
            if winner == CIRCLES:
                ret = 1
            else:
                ret = 0
            #current_node.rollback(ret)
            return ret
        elif len(self.possible_moves(board)) == 0:
            #current_node.rollback(0.5)
            return 0.5

        new_board = self.opponent_move(board) if recursive else self.board.copy()

        moves = self.possible_moves(new_board)
        if len(moves) > 0:
            if printing:
                global data_viz_output
                data_viz_output += str(self.game_order) + "\n"
            selected_node = current_node.find_best_move(
                self.opponent_made_moves[0], self.game_order, printing=printing
            )
            # if we haven't found the move before we generate a new one
            if selected_node == -1:
                selected_node = random.choice(moves)
                while selected_node not in current_node.game_order and selected_node != -1:
                  selected_node = random.choice(moves)
            self.opponent_made_moves.remove(self.opponent_made_moves[0])

            self.game_order.append(selected_node)
            for c in current_node.children:
                if c.move == selected_node:
                    current_node = c
            new_board[selected_node] = CIRCLES
       # print(self.game_order)
        return self.q_sim(current_node, new_board.copy(), printing = printing)

    def new_ai_game(self) -> None:
        """simulates 10000"""
        i = 0
        j = 0
        for _ in range(10000):
            self.__init__()
            current_node = self.tree
            reward = self.q_sim(current_node,ttt.board, printing=False, recursive=False)
            # print(self.game_order, reward)
            if reward == 1:
                i += 1
            if reward == 0.5:
                j += 1

        print("total game: 10,000 games won: ", i, "draws: ", j)

    def new_ai_game(self) -> None:
        """simulates 10000"""
        i = 0
        j = 0
        for _ in range(10000):
            self.__init__()
            current_node = self.tree
            reward = self.q_sim(current_node,ttt.board, printing=False, recursive=False)
            # print(self.game_order, reward)
            if reward == 1:
                i += 1
            if reward == 0.5:
                j += 1

        print("total game: 10,000 games won: ", i, "draws: ", j)
        
    def viz_game(self) -> None:
        global data_viz_output
        for _ in range(5):
            print(_)
            self.__init__()
            current_node = self.tree
            reward = self.q_sim(current_node, ttt.board, recursive=False, printing=True)
        print(data_viz_output)
        print("start writing")
        with open("Output2.txt", "w") as f:
            f.write(data_viz_output)


if __name__ == "__main__":
    ttt = TicTacToe()
    ttt.Q_convergence()
    ttt.new_ai_game()
    ttt.viz_game()
