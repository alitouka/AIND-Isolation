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

def legal_second_moves(game, player):
    second_move_directions = [(-3, -3), (-3, -1), (-3, 1), (-3, 3),
                              (-2, 0),
                              (-1, -3), (-1, -1), (-1, 1), (-1, 3),
                              (0, -2), (0, 2),
                              (1, -3), (1, -1), (1, 1), (1, 3),
                              (2, 0),
                              (3, -3), (3, -1), (3, 1), (3, 3)]
    r, c = game.get_player_location(player)
    legal_moves = []

    for dr, dc in second_move_directions:
        move = (r+dr, c+dc)
        if game.move_is_legal(move):
            legal_moves.append(move)

    return legal_moves


def max_number_of_possible_moves(game, location):
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


def custom_score(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    own_second_moves = set()
    opp_second_moves = set()

    own_move_scores = {}
    total_own_move_scores = 0
    for m in own_moves:
        s, second_moves = max_number_of_possible_moves(game, m)
        own_move_scores[m] = s
        own_second_moves = own_second_moves | set(second_moves)
        total_own_move_scores += s


    opp_move_scores = {}
    total_opp_move_scores = 0
    for m in opp_moves:
        s, second_moves = max_number_of_possible_moves(game, m)
        opp_move_scores[m] = s
        opp_second_moves = opp_second_moves | set(second_moves)
        total_opp_move_scores += s

    overlapping_moves = set(own_moves) & set(opp_moves)
    overlapping_second_moves = own_second_moves & opp_second_moves
    board_size = game.width*game.height



    result = total_own_move_scores - 1.5*total_opp_move_scores - 0.5*len(overlapping_second_moves)

    # overlapping_moves = set(own_moves) & set(opp_moves)
    # total_overlapping_moves_score = 0
    #
    # for m in overlapping_moves:
    #     total_overlapping_moves_score += max(opp_move_scores[m], own_move_scores[m])
    #
    # result -= 2.0*total_overlapping_moves_score

    # if len(overlapping_moves) != 0 and total_own_move_scores != 0 and total_opp_move_scores != 0:
    #     avg_own_move_score = total_own_move_scores / float(len(own_moves))
    #     avg_opp_move_score = total_opp_move_scores / float(len(opp_moves))
    #     max_overlapping_move_score = 0
    #
    #     for m in overlapping_moves:
    #         temp_own = own_move_scores[m] / avg_own_move_score
    #         temp_opp = opp_move_scores[m] / avg_opp_move_score
    #         s = temp_own * temp_opp
    #
    #         if s > max_overlapping_move_score:
    #             max_overlapping_move_score = s
    #
    #     result += 1.5*max_overlapping_move_score

    return result




def custom_score_old(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    own_moves_count = len(own_moves)
    # own_moves_fraction = own_moves_count / (max_number_of_possible_moves(game, player)+1.0)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    opp_moves_count = len(opp_moves)
    # opp_moves_fraction = opp_moves_count / (max_number_of_possible_moves(game, game.get_opponent(player))+1.0)

    # Check if I can immediately block my opponent
    if opp_moves_count == 1 and opp_moves[0] in own_moves:
        return 1000.0

    # Check if my opponent can block me immediately
    if own_moves_count == 1 and own_moves[0] in opp_moves:
        return -1000;

    own_moves_set = set(own_moves)
    opp_moves_set = set(opp_moves)
    overlapping_moves_set = own_moves_set & opp_moves_set
    overlapping_moves_count = len(overlapping_moves_set)
    own_overlapping_moves_fraction = overlapping_moves_count / (own_moves_count+1.0)
    opp_overlapping_moves_fraction = overlapping_moves_count / (opp_moves_count + 1.0)

    # return float((1.0-own_overlapping_moves_fraction)*own_moves_count
    #              - (2.0+opp_overlapping_moves_fraction)*opp_moves_count)

    # own_second_moves_set = set(legal_second_moves(game, player))
    # own_second_moves_count = len(own_second_moves_set)
    # opp_second_moves_set = set(legal_second_moves(game, game.get_opponent(player)))
    # opp_second_moves_count = len(opp_second_moves_set)
    # overlapping_second_moves_set = own_second_moves_set & opp_second_moves_set

    # Check if an opponent can block me on their 2nd move
    # if len(own_second_moves_set) == 1 and (own_second_moves_set in opp_moves or own_second_moves_set in opp_second_moves_set):
    #     return -1000.0;

    return float(own_moves_count
                 - 2*opp_moves_count
                 - 1.5*len(overlapping_moves_set))
    #              # + 0.1*own_second_moves_count
    #              # - 0.2*opp_second_moves_count
    #              # - 0.05*len(overlapping_second_moves_set))

    # Falling back to "my moves - opponent moves" heuristic
    # return float(own_moves_count - opp_moves_count)


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
                unique_nodes_visited_previously = 0

                while self.time_left() >= self.TIMER_THRESHOLD:
                    _, best_found_move = fn(game, depth)

                    # _, unique_nodes_visited = game.counts

                    #if unique_nodes_visited == unique_nodes_visited_previously:
                    #    break; # We haven't found any new nodes in this iteration

                    # unique_nodes_visited_previously = unique_nodes_visited
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
