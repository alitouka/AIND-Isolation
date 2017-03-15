"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def legal_moves_from_location(game, location):
    """
    Finds all legal moves from a given location
    
    :param game: an instance of isolation.Board class
    :param location: a location on a board in a form of a tuple (row, column)
    :return: a number of legal moves, a list of legal moves 
    """
    result = 0
    row, col = location
    legal_moves = []

    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]

    for dr, dc in directions:
        r = row+dr
        c = col+dc

        if game.move_is_legal((r, c)):
            legal_moves.append((r, c))
            result += 1

    return result, legal_moves


def custom_score_1(game, player):
    """
    A heuristic function defined as

    N         - N               - N
     my_moves    opponent_moves    overlapping_moves

    :param game:
    :param player:
    :return:
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    own_moves_count = len(own_moves)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    opp_moves_count = len(opp_moves)

    own_moves_set = set(own_moves)
    opp_moves_set = set(opp_moves)
    overlapping_moves_set = own_moves_set & opp_moves_set

    return float(own_moves_count-opp_moves_count-len(overlapping_moves_set))


def custom_score_2(game, player):
    """
    A heuristic function defined as

    N                       - N
     my_unique_second_moves    opponent_unique_second_moves

    :param game:
    :param player:
    :return:
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    own_second_moves = set()
    opp_second_moves = set()

    for m in own_moves:
        _, second_moves = legal_moves_from_location(game, m)
        own_second_moves = own_second_moves | set(second_moves)

    for m in opp_moves:
        _, second_moves = legal_moves_from_location(game, m)
        opp_second_moves = opp_second_moves | set(second_moves)

    return len(own_second_moves) - len(opp_second_moves)


def custom_score_3(game, player):
    """
    A heuristic function defined as

    N                    - N
     all_my_second_moves    all_opponent_second_moves

    :param game:
    :param player:
    :return:
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    number_of_own_second_moves = 0
    for m in own_moves:
        s, _ = legal_moves_from_location(game, m)
        number_of_own_second_moves += s

    number_of_opp_second_moves = 0
    for m in opp_moves:
        s, _ = legal_moves_from_location(game, m)
        number_of_opp_second_moves += s

    return number_of_own_second_moves - number_of_opp_second_moves


def custom_score_4(game, player):
    """
    A heuristic function defined as

    N                    - 1.5*N                          - 0.5*N
     all_my_second_moves        all_opponent_second_moves        overlapping_second_moves

    :param game:
    :param player:
    :return:
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    own_second_moves = set()
    opp_second_moves = set()

    number_of_own_second_moves = 0
    for m in own_moves:
        s, second_moves = legal_moves_from_location(game, m)
        own_second_moves = own_second_moves | set(second_moves)
        number_of_own_second_moves += s

        number_of_opp_second_moves = 0
    for m in opp_moves:
        s, second_moves = legal_moves_from_location(game, m)
        opp_second_moves = opp_second_moves | set(second_moves)
        number_of_opp_second_moves += s

    overlapping_second_moves = own_second_moves & opp_second_moves
    return number_of_own_second_moves - 1.5*number_of_opp_second_moves - 0.5*len(overlapping_second_moves)


def custom_score(game, player):
    return custom_score_4(game, player)

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.transposition_table = {}
        self.use_transposition_table = True

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        self.transposition_table = {}

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        best_found_move = (-1, -1)
        if self.method == 'minimax':
            fn = self.minimax
        elif self.method == 'alphabeta':
            fn = self.alphabeta
        else:
            raise ValueError

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                depth = 1

                while self.time_left() >= self.TIMER_THRESHOLD:
                    _, best_found_move = fn(game, depth)
                    depth+=1
            else:
                _, best_found_move = fn(game, self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return best_found_move

    def transposition_table_key(self, game, mp):
        return game.to_string() + str(mp)

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            pass
            # raise Timeout()

        if depth == 0:
            return self.score(game, self), game.get_player_location(self)

        if self.use_transposition_table:
            key = self.transposition_table_key(game, maximizing_player)
            if key in self.transposition_table:
                cached_depth, cached_score, cached_move = self.transposition_table[key]
                if cached_depth >= depth:
                    return cached_score, cached_move

        legal_moves = game.get_legal_moves(game.active_player)
        best_found_score = self.__worst_score__(maximizing_player)
        best_found_move = (-1, -1)

        for m in legal_moves:
            updated_game = game.forecast_move(m)
            temp_score, temp_move = self.minimax(updated_game, depth-1, not maximizing_player)

            if (maximizing_player and temp_score > best_found_score)\
                    or (not maximizing_player and temp_score < best_found_score):
                best_found_score = temp_score
                best_found_move = m

        if self.use_transposition_table:
            if key in self.transposition_table:
                cached_depth, _, _ = self.transposition_table[key]
                if cached_depth < depth:
                    self.transposition_table[key] = (depth, best_found_score, best_found_move)
            else:
                self.transposition_table[key] = (depth, best_found_score, best_found_move)

        return best_found_score, best_found_move

    def __no_moves__(self, maximizing_player):
        return self.__worst_score__(maximizing_player), (-1, -1)

    def __worst_score__(self, maximizing_player):
        if maximizing_player:
            return float("-inf")
        else:
            return float("inf")


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
            return self.score(game, self), game.get_player_location(self)

        if self.use_transposition_table:
            key = self.transposition_table_key(game, maximizing_player)
            if key in self.transposition_table:
                cached_depth, cached_score, cached_move = self.transposition_table[key]
                if cached_depth >= depth:
                    return cached_score, cached_move

        legal_moves = game.get_legal_moves(game.active_player)
        best_found_score = self.__worst_score__(maximizing_player)
        best_found_move = (-1, -1)

        for m in legal_moves:
            updated_game = game.forecast_move(m)
            temp_score, temp_move = self.alphabeta(updated_game, depth-1, alpha, beta, not maximizing_player)

            if (maximizing_player and temp_score > best_found_score)\
                    or (not maximizing_player and temp_score < best_found_score):
                best_found_score = temp_score
                best_found_move = m

            if maximizing_player:
                if best_found_score >= beta:
                    break
                alpha = max(alpha, best_found_score)
            else:
                if best_found_score <= alpha:
                    break
                beta = min(beta, best_found_score)

        if self.use_transposition_table:
            if key in self.transposition_table:
                cached_depth, _, _ = self.transposition_table[key]
                if cached_depth < depth:
                    self.transposition_table[key] = (depth, best_found_score, best_found_move)
            else:
                self.transposition_table[key] = (depth, best_found_score, best_found_move)

        return best_found_score, best_found_move
