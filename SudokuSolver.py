""" Sudoku Solver

Possible future tasks:
    - Implement Naked triplets / Naked quads rules
    - Implement Hidden triplets / Hidden quads rules
    - Implement X-Wing / Y-Wing rules
    - Implement Fish rules (jellyfish, swordfish, etc)
    - Implement Unique Rectangle / Empty Rectangle rules

Additional references for advanced sudoku solving techniques
    # https://www.learn-sudoku.com/index.html
    # http://hodoku.sourceforge.net/en/techniques.php
"""

from copy import copy
import csv
import getopt
import math
import sys
import time

def pair_permutations(input: list) -> list:
    """Use list comprehension to create all pairs of values from the input list"""
    n = len(input)
    return [((input[i]), (input[(i + 1) % n])) for i in range(n)]

class Board:
    """The main storage class for a sudoku board

    Parameters
    ----------
    board_size : `int`, optional
        The number of rows and columns the board should have (default is 9)
    cells : `list[int]`, optional
        A grid to initialze the board to, rather than initializing a blank grid

    Raises
    ------
    ValueError
        If the supplied board size is not a valid Sudoku board size.
    """

    BOARD_SIZE: int
    BOX_SIZE: int
    DEFAULT_CANDIDATES: list[int]

    cells: list[list[int]]
    candidates: list[list[set[int]]]   # List of candidates for every cell

    def __init__(self, board_size: int = None, cells: list[int] = None):
        if (board_size == None):
            board_size = 9

        Board.validateBoardSize(board_size) # throws an exception if board size is not valid
        self.BOARD_SIZE=board_size

        # A Sudoku board is broken into boxes of sqrt(BOARD_SIZE) cells
        self.BOX_SIZE = math.floor(math.sqrt(self.BOARD_SIZE))

        # Any cell can have a value between 1 and BOARD_SIZE
        self.DEFAULT_CANDIDATES = [i + 1 for i in range(self.BOARD_SIZE)]

        # Set the board cells. Use the user supplied board if there was one.
        # Otherwise, default all cells to None
        if cells is not None:
            self.cells = cells
        else:
            self.cells = []
            for i in range(self.BOARD_SIZE):
                self.cells.append([None for _ in range(self.BOARD_SIZE)])

        # Initialze candidates list for each cell. If a cell has a value, it should
        # no longer have any candidates. Otherwise use the findPossibleValues
        # function to check all other cells the cell can "see" to get the initial
        # list of candidates
        self.candidates = []
        for i in range(self.BOARD_SIZE):
            self.candidates.append([])
            for j in range(self.BOARD_SIZE):
                if self.cells[i][j] == None:
                    self.candidates[i].append(self.findPossibleValues(i, j))
                else:
                    self.candidates[i].append(set())

    def __copy__(self):
        _copy = Board()
        _copy.BOARD_SIZE = self.BOARD_SIZE
        _copy.BOX_SIZE = self.BOX_SIZE
        _copy.DEFAULT_CANDIDATES = self.DEFAULT_CANDIDATES
        _copy.cells = []
        for row in self.cells:
            _copy.cells.append(row.copy())
        _copy.candidates = []
        for row in self.candidates:
            col = []
            for c in row:
                col.append(c.copy())
            _copy.candidates.append(col)
        return _copy

    def validateBoardSize(size:int) -> None:
        """ Validate a board size.

        Parameters
        ----------
        size : `int`
            The size to check for validity.

        Raises
        ------
        ValueError
            If the supplied board size is not a valid Sudoku board size.
        """
        if math.floor(math.sqrt(size)) != math.sqrt(size):
            raise ValueError("Board size must be a square number!")

    def getValuesInRow(self, row: int) -> list[int]:
        """ Get all populated values int he specified row

        Parameters
        ----------
        row : `int`
            The row to get all values of

        Returns
        ------
        values : `list[int]`
            A list of the populated values in the row
        """
        return self.cells[row]

    def getValuesInCol(self, col: int) -> list[int]:
        """ Get all populated values in the specified column

        Parameters
        ----------
        col : `int`
            The column to get all values of

        Returns
        ------
        values : `list[int]`
            A list of the populated values in the column
        """
        return [row[col] for row in self.cells]

    def getValuesInBox(self, row: int, col: int) -> list[int]:
        """ Get all populated values in the box that the cell [row, col] is part of

        Parameters
        ----------
        row : `int`
            The row of the cell to check the box of
        col : `int`
            The column of the cell to check the box of

        Returns
        ------
        values : `list[int]`
            A list of the populated values in the box
        """
        boxIndices = self.getBoxIndices(row, col)
        return [self.cells[row][col] for row, col in boxIndices]

    def findPossibleValues(self, row: int, col: int) -> set[int]:
        """ Get all possible values a specified cell can be given the values it can "see"

        Parameters
        ----------
        row : `int`
            The row of the cell of interest
        col : `int`
            The col of the cell of interest

        Returns
        -------
        values : `set[int]`
            A set of the possible values that the cell at [row, col] can be
        """
        all_options = set(self.DEFAULT_CANDIDATES.copy())
        row_used = set(self.getValuesInRow(row))
        col_used = set(self.getValuesInCol(col))
        box_used = set(self.getValuesInBox(row, col))

        # Get the union of the three sets, which are all values that can be
        # "seen" by the cell at [row, col]
        all_used = row_used | col_used | box_used

        # Possible candidates for cell at [row, col] is the set difference between
        # all possible values and the values that are currently "visible"
        return all_options.difference(all_used)

    def allCellsFilled(self) -> bool:
        """Check if all cells in the board have been filled

        Returns
        -------
        result : `bool`
            `True` if all cells in the board have a value, `False` otherwise
        """
        return all(all(cell is not None for cell in row) for row in self.cells)

    def isSolved(self) -> bool:
        """Check if the board has been successfully solved

        Returns
        -------
        state : `bool`
            `True` if the board has been successfully solved, `False` otherwise
        """

        # Check that all cells have a value
        if not self.allCellsFilled():
            return False

        # Set of all values 1-9
        candidates_set = set(self.DEFAULT_CANDIDATES.copy())

        # Check if each row has all values 1-9
        for row in range(self.BOARD_SIZE):
            row_vals = set(self.getValuesInRow(row))
            if not candidates_set == row_vals:
                return False

        # Check if each column has all values 1-9
        for col in range(self.BOARD_SIZE):
            col_vals = set(self.getValuesInCol(col))
            if not candidates_set == col_vals:
                return False

        # Check if each box has all values 1-9
        all_box_beginnings = self.getAllBoxBeginnings()
        for row, col in all_box_beginnings:
            if not candidates_set == set(self.getValuesInBox(row, col)):
                return False

        return True

    def getBoxIndices(self, row: int, col: int) -> list[tuple[int, int]]:
        """ Return indices of all cells in same box as cell at [row, col]

        Parameters
        ----------
        row : `int`
            The row of the cell of interest
        col : `int`
            The column of the cell of interest

        Returns
        -------
        indices : `list[tuple[int, int]]`
            The indices of all cells in the same box as the cell at [row, col]
        """
        indices = []
        r0 = (row // self.BOX_SIZE) * self.BOX_SIZE # First row of box
        c0 = (col // self.BOX_SIZE) * self.BOX_SIZE # First column of box
        for r in range(r0, r0 + self.BOX_SIZE):
            for c in range(c0, c0 + self.BOX_SIZE):
                indices.append((r, c))
        return indices

    def getAllBoxBeginnings(self) -> list[tuple[int, int]]:
        """ Returns the indices of the top-left cell in each box """
        indices = []
        row = 0
        while row < self.BOARD_SIZE:
            col = 0
            while col < self.BOARD_SIZE:
                indices.append([row, col])
                col += self.BOX_SIZE
            row += self.BOX_SIZE
        return indices

    def setCell(self, row:int, col:int, val:int) -> None:
        """ Set the value of a cell and remove that value from candidates of neighboring cells

        Parameters
        ----------
        row : `int`
            The row of the cell to set the value of
        col : `int`
            The column of the cell to set the value of
        val : `int`
            The new value to use for the cell at [row, col]
        """
        self.cells[row][col] = val
        self.candidates[row][col] = set()
        self.clearCandidates(row, col, val)

    def clearCandidates(self, row:int, col:int, val:int) -> None:
        """ Remove a value from candidates of neighbors of the cell at [row, col]

        Parameters
        ----------
        row : `int`
            The row of the cell of interest
        col : `int`
            The column of the cell of interest
        val : `int`
            The value to remove from the candidates of neighboring cells
        """
        self.removeCandidateFromRow(row, [col], val)
        self.removeCandidateFromCol(col, [row], val)
        self.removeCandidateFromBox(row, col, [row, col], val)

    def removeCandidateFromRow(self, row: int, exclude: list[int], val: int) -> None:
        """ Remove a candidate from all cells in a row except those in the exclude list

        Parameters
        ----------
        row : `int`
            The row in which to remove the candidate from all cells
        exclude: `list[int]`
            A list of columns to exclude from candidate removal
        val : `int`
            The candidate to remove
        """
        for col in range(self.BOARD_SIZE):
            if (col not in exclude) and (val in self.candidates[row][col]):
                self.candidates[row][col].remove(val)

    def removeCandidateFromCol(self, col: int, exclude: list[int], val: int) -> None:
        """ Remove a candidate from all cells in a column except those  in the exclude list

        Parameters
        ----------
        col : `int`
            The column in which to remove the candidate from all cells
        exclude : `list[int]`
            A list of rows to exclude from candidate removal
        val : `int`
            The candidate to remove
        """
        for row in range(self.BOARD_SIZE):
            if (row not in exclude) and (val in self.candidates[row][col]):
                self.candidates[row][col].remove(val)

    def removeCandidateFromBox(self, row: int, col: int, exclude: list[tuple[int, int]], val: int) -> None:
        """ Remove a candidate from all cells in a box except those in the exclude list

        Parameters
        ----------
        row, col : `int`
            The beginning cell of the box in which to remove the candidate from all cells
        exclude : `list[tuple[int, int]]`
            A list of cells to exclude from candidate removal
        val : `int`
            The candidate to remove
        """
        for r, c in self.getBoxIndices(row, col):
            if ([r, c] not in exclude) and (val in self.candidates[r][c]):
                self.candidates[r][c].remove(val)

    def createPair(self, row1:int, col1:int, row2:int, col2:int, val1:int, val2:int) -> None:
        """
        Remove a value pair from candidates in row/col/box except for specified
        pair at [row1, col1] and [row2, col2]

        Parameters
        ----------
        row1 : `int`
            The row of the first cell in the pair
        col1 : `int`
            The column of the first cell in the pair
        row2 : `int`
            The row of the second cell in the pair
        col2 : `int`
            The column of the second cell in the pair
        val1 : `int`
            The first value to remove from candidates that the pair can "see"
        val2 : `int`
            The second value to remove from candidates that the pair can "see"
        """
        if row1 == row2 and col1 == col2:
            print("Warning: both cells passed to createPair were the same cell")
            return

        # Cells within same row
        if row1 == row2:
            for col in range(self.BOARD_SIZE):
                if (col != col1) and (col != col2):
                    if val1 in self.candidates[row1][col]:
                        self.candidates[row1][col].remove(val1)
                    if val2 in self.candidates[row1][col]:
                        self.candidates[row1][col].remove(val2)

        # Cells within same column
        if col1 == col2:
            for row in range(self.BOARD_SIZE):
                if (row != row1) and (row != row2):
                    if val1 in self.candidates[row][col1]:
                        self.candidates[row][col1].remove(val1)
                    if val2 in self.candidates[row][col1]:
                        self.candidates[row][col1].remove(val2)

        # Cells within same box
        if [row2, col2] in self.getBoxIndices(row1, col1):
            box_indices = self.getBoxIndices(row1, col1)
            for row, col in box_indices:
                if (row != row1 and col != col1) and (row != row2 and col != col2):
                    if val1 in self.candidates[row][col]:
                        self.candidates[row][col].remove(val1)
                    if val2 in self.candidates[row][col]:
                        self.candidates[row][col].remove(val2)

    def fromCsv(csvPath: str, board_size: int = None, delim: str = None) -> "Board":
        """ Create a new Sudoku board from a supplied CSV file

        Parameters
        ----------
        csvPath : `str`
            The file path of the CSV file
        board_size : `int`, optional
            The size of board to generate
        delim : `str`, optional
            The delimiter to use in the csv parsing

        Returns
        -------
        board : `Board`
            The new Sudoku board read in from the CSV file
        """

        if delim is None:
            delim = ';'

        # Parse each row from the CSV file using the user-supplied delimiter
        # For each value, set the value in the new board to the specified value,
        # or None if it should be blank. Finally, return a new Board object
        # using the parsed data.
        cells = []
        try:
            with open(csvPath) as csvfile:
                reader = csv.reader(csvfile, delimiter=delim)
                for row in reader:
                    tmp_row = []
                    for col in row:
                        val = (int(col) if col else None)
                        tmp_row.append(val)
                    cells.append(tmp_row)
                csvfile.close()
        except FileNotFoundError as e:
            print("Error reading csv file: {}".format(e), file=sys.stderr)
            sys.exit(3)
        b = Board(board_size, cells)
        return b

    def __str__(self):
        """ Override the string conversion operator for the class to aid in
        printing the Board to screen

        I started from this link and made the printing generic for
        any board size: https://stackoverflow.com/a/37953563
        """
        row_separator = "+" + ("---+" * self.BOARD_SIZE)
        cell_markers = "+" + "   +" * self.BOARD_SIZE
        result = row_separator
        for row in range(self.BOARD_SIZE):
            result += '\n'
            num_boxes_per_row = self.BOARD_SIZE // self.BOX_SIZE
            segment = [" {} "] * self.BOX_SIZE
            rowstr = "|" + (' '.join(segment) + "|") * num_boxes_per_row
            rowstr = rowstr.format(*[self.cells[row][i] for i in range(self.BOARD_SIZE)])
            result += rowstr
            if row % self.BOX_SIZE == self.BOX_SIZE - 1:
                result += "\n" + row_separator
            else:
                result += "\n" + cell_markers
        return result.replace("None", ' ')

class CellPair:
    """Simple helper class for a pair of cells"""

    cell1 : tuple[int, int]
    cell2 : tuple[int, int]
    values: set[int]

    def __init__(self, cell1: tuple[int, int], cell2: tuple[int, int], values: set[int]):
        self.cell1 = cell1
        self.cell2 = cell2
        self.values = values

    def __eq__(self, other: "CellPair"):
        return ((self.cell1 == other.cell1 and self.cell2 == other.cell2) or \
                (self.cell1 == other.cell2 and self.cell2 == other.cell1)) and \
                (self.values == other.values)

class Solver:
    """A class with various methods to solve a sudoku puzzle

    Parameters
    ----------
    board : `Board`
        The sudoku board this solver will be in charge of solving
    verbose : `bool`
        Whether or not to print solve steps to the console (True by default)
    """

    board : Board
    verbose : bool
    rules : list["function"]  # List of all possible placement rules, in the order they should be tried
    pairRules : list["function"] # List of rules to create candidate pairs, in the order they should be tried
    knownPairs : list[CellPair] # Keep track of known pairs to avoid "finding" a pair that was already found
    knownPointingRow : list[tuple[int, int, int]] # Keep track of known pointing values in rows to avoid infinite loops
    knownPointingCol : list[tuple[int, int, int]] # Keep track of known pointing values in columns to avoid infinite loops
    knownClaimedRow : list[tuple[int, int, int]] # Keep track of known claimed values in rows to avoid infinite loops
    knownClaimedCol : list[tuple[int, int, int]] # Keep track of known claimed values in columns to avoid infinite loops

    def __init__(self, board: Board, verbose: bool = True):
        self.board = board
        self.verbose = verbose
        self.rules = [
            self.LoneSingle,
            self.OpenSingle,
            self.HiddenSingle,
        ]
        self.pairRules = [
            self.NakedPair,
            self.HiddenPair
        ]
        self.limitRules = [
            self.LockedCandidatesPointing,
            self.LockedCandidatesClaimed
        ]
        self.knownPairs = []
        self.knownPointingRow = []
        self.knownPointingCol = []
        self.knownClaimedRow = []
        self.knownClaimedCol = []

    def nextMove(self) -> bool:
        """Attempts to find the next easiest move to set a value in the board by
        sequentially trying to apply the list of rules

        Returns
        -------
        moveWasMade : `bool`
            `True` if a rule was able to be applied
            `False` if there was no rule that could be used to set another cell's value
        """
        # Don't bother trying to solve the puzzle if it's already solved
        if self.board.allCellsFilled():
            print("Board is already solved")
            return True

        # Sequentially try each placement rule in the rule list, and stop if one was applied
        for rule in self.rules:
            if self.applyRule(rule):
                return True

        # Sequentially try each pair rule in the rule list, and stop if one was applied
        for rule in self.pairRules:
            if self.applyPairRule(rule):
                return True

        # Sequentially try each candidate limitation rule, and stop if one was applied
        for rule in self.limitRules:
            if rule():
                return True

        # If the function gets here, either the puzzle isn't solveable or the
        # solver needs more complicated rules. Print a message to the user so
        # they know to stop trying with this puzzle
        print("No move found. There are two reasons this may happen:")
        print("\t- The board may be in an unsolveable state")
        print("\t- This solver may need more advanced solving rules")
        return False

    def solve(self) -> None:
        """ Solve the board using repeated calls to `nextMove()`

            Prints the number of moves made and the amount of time it took to make them.
            Prints if the board was solved or not.
            Prints the final state of the board.
        """
        startTime = time.perf_counter_ns()
        movesMade = 0
        while not self.board.allCellsFilled():
            movesMade += 1
            if not self.nextMove():
                break
        elapsed = time.perf_counter_ns() - startTime
        elapsed /= 1000000

        print("{} moves made in {} milliseconds".format(movesMade, elapsed))
        if not self.board.isSolved():
            print("Board was not able to be solved :(")
        else:
            print("Board solved!")
        print(self.board)

    def applyRule(self, rule: "function") -> bool:
        """ Try to apply a rule to the board

        Returns
        -------
        success: `bool`
            `True` if the rule was able to fill in a cell,
            `False` otherwise

        Notes
        -----
        This function will also print the rule applied to which cell if verbose
        mode isn't disabled
        """
        row, col, val = rule()
        if row is None or col is None or val is None:
            return False
        if self.verbose:
            print("{} technique used to set cell at row {} and column {} to {}".format(rule.__name__, row + 1, col + 1, val))
        return True

    def applyPairRule(self, rule: "function") -> bool:
        """ Try to apply a rule to the board

        Returns
        -------
        success: `bool`
            `True` if the rule was able to fill in a cell,
            `False` otherwise

        Notes
        -----
        This function will also print the rule applied to create which candidate
        pair if verbose mode isn't disabled
        """
        results = rule()
        if None in results:
            return False
        if self.verbose:
            row1, col1, row2, col2, val1, val2 = results
            print("{} technique used to set ({}, {}) and ({}, {}) as a candidate pair for values {} and {}".format(
                rule.__name__, row1 + 1, col1 + 1, row2 + 1, col2 + 1, val1, val2))
            self.knownPairs.append(CellPair([row1, col1], [row2, col2], set([val1, val2])))
        return True

    def OpenSingle(self) -> tuple[int, int, int]:
        """ The open single rule applies when there is a row, column, or box that is
        only missing one possible value

        Returns
        -------
        row, col : `int`
            The cell affected by the rule
        val : `int`
            The value the cell was set to

        Notes
        -----
        Returns will be (None, None, None) if the rule could not be applied
        """
        # Try to find a row with only one missing value
        for row in range(self.board.BOARD_SIZE):
            row_values = self.board.getValuesInRow(row)
            missing_values = list(set(self.board.DEFAULT_CANDIDATES).difference(set(row_values)))
            if len(missing_values) == 1:
                col = row_values.index(None) # find column that is not set
                self.board.setCell(row, col, missing_values[0])
                return row, col, missing_values[0]

        # Try to find a column with only one missing value
        for col in range(self.board.BOARD_SIZE):
            col_values = self.board.getValuesInCol(col)
            missing_values = list(set(self.board.DEFAULT_CANDIDATES).difference(set(col_values)))
            if len(missing_values) == 1:
                row = col_values.index(None) # find row that is not set
                self.board.setCell(row, col, missing_values[0])
                return row, col, missing_values[0]

        # Try to find a box with only one missing value
        for box_row, box_col in self.board.getAllBoxBeginnings():
            box_values = self.board.getValuesInBox(box_row, box_col)
            box_indices = self.board.getBoxIndices(box_row, box_col)
            missing_values = list(set(self.board.DEFAULT_CANDIDATES).difference(set(box_values)))
            if len(missing_values) == 1:
                index = box_values.index(None) # find cell in box that is not set
                row, col = box_indices[index]
                self.board.setCell(row, col, missing_values[0])
                return row, col, missing_values[0]

        return None, None, None

    def LoneSingle(self) -> tuple[int, int, int]:
        """ The lone single rule applies when a cell has only one candidate left

        Returns
        -------
        row, col : `int`
            The cell affected by the rule
        val : `int`
            The value the cell was set to

        Notes
        -----
        Returns will be (None, None, None) if the rule could not be applied
        """
        for row in range(self.board.BOARD_SIZE):
            for col in range(self.board.BOARD_SIZE):
                if len(self.board.candidates[row][col]) == 1:
                    val = list(self.board.candidates[row][col])[0]
                    self.board.setCell(row, col, val)
                    return row, col, val
        return None, None, None

    def HiddenSingle(self) -> tuple[int, int, int]:
        """ The hidden single rule applies when a candidate has only one possible
        location within a row, column, or box

        Returns
        -------
        row, col : `int`
            The cell affected by the rule
        val : `int`
            The value the cell was set to

        Notes
        -----
        Returns will be (None, None, None) if the rule could not be applied
        """
        for candidate in self.board.DEFAULT_CANDIDATES:
            # Check for hidden single across rows
            for row in range(self.board.BOARD_SIZE):
                count = 0 # Count of instances of candidate
                index = self.board.BOARD_SIZE # If there is only one column with the candidate, this will be that column
                for col in range(self.board.BOARD_SIZE):
                    if candidate in self.board.candidates[row][col]:
                        count += 1
                        index = col
                    if count > 1:
                        break
                if count == 1:
                    self.board.setCell(row, index, candidate)
                    return row, index, candidate

            # Check for hidden single across cols
            for col in range(self.board.BOARD_SIZE):
                count = 0 # Count of instances of candidate
                index = self.board.BOARD_SIZE # If there is only one row with the candidate, this will be that row
                for row in range(self.board.BOARD_SIZE):
                    if candidate in self.board.candidates[row][col]:
                        count += 1
                        index = row
                    if count > 1:
                        break
                if count == 1:
                    self.board.setCell(index, col, candidate)
                    return index, col, candidate

            # Check for hidden single within boxes
            for box_row, box_col in self.board.getAllBoxBeginnings():
                count = 0 # Count of instances of candidate
                row_index = self.board.BOARD_SIZE # If there is only one cell with the candidate, this will be the row of that cell
                col_index = self.board.BOARD_SIZE # If there is only one cell with the candidate, this will be the col of that cell
                box_indices = self.board.getBoxIndices(box_row, box_col)
                for row, col in box_indices:
                    if candidate in self.board.candidates[row][col]:
                        count += 1
                        row_index = row
                        col_index = col
                    if count > 1:
                        break
                if count == 1:
                    self.board.setCell(row_index, col_index, candidate)
                    return row_index, col_index, candidate

        return None, None, None

    def NakedPair(self) -> tuple[int, int, int, int, int, int]:
        """ Try to apply the "Naked Pairs" rule to the board

        The naked pairs rule applies when there are two and only two cells within
        a row, column, or box that are limited to the same two candidates. These
        two candidates can then be removed as candidates from any cell that both
        cells of the pair can "see"

        Returns
        -------
        row1, col1 : `int`
            The first cell in the pair created
        row2, col2 : `int`
            The second cell in the pair created
        val1, val2 : `int`
            The values the cells are a candidate pair for

        Notes
        -----
        Returns will be (None, None, None, None, None, None) if the rule could not be applied
        """

        # Check for row pairs
        for row in range(self.board.BOARD_SIZE):
            # Count the number of candidates in each cell
            row_candidate_count = [len(self.board.candidates[row][col]) for col in range(self.board.BOARD_SIZE)]

            # Limit to only cells with exactly two candidates
            two_candidates = [col for col in range(self.board.BOARD_SIZE) if row_candidate_count[col] == 2]

            # Need at least two cells with exactly two candidates to check for naked pairs
            if len(two_candidates) < 2:
                continue

            # Check all possible pairs. If any two cells are limited to the same two candidates, and this is not
            # a known pair, create a new pair and return
            possible_pairs = pair_permutations(two_candidates)
            for pair in possible_pairs:
                cand1 = self.board.candidates[row][pair[0]]
                cand2 = self.board.candidates[row][pair[1]]
                if cand1 == cand2:
                    c1 = pair[0]
                    c2 = pair[1]
                    val1 = list(cand1)[0]
                    val2 = list(cand1)[1]
                    newPair = CellPair([row, c1], [row, c2], set([val1, val2]))
                    if newPair not in self.knownPairs:
                        self.board.createPair(row, c1, row, c2, val1, val2)
                        return row, c1, row, c2, val1, val2

        # Check for column pairs
        for col in range(self.board.BOARD_SIZE):
            # Count the number of candidates in each cell
            col_candidate_count = [len(self.board.candidates[row][col]) for row in range(self.board.BOARD_SIZE)]

            # Limit to only cells with exactly two candidates
            two_candidates = [row for row in range(self.board.BOARD_SIZE) if col_candidate_count[row] == 2]

            # Need at least two cells with exactly two candidates to check for naked pairs
            if len(two_candidates) < 2:
                continue

            # Check all possible pairs. If any two cells are limited to the same two candidates, and this is not
            # a known pair, create a new pair and return
            possible_pairs = pair_permutations(two_candidates)
            for pair in possible_pairs:
                cand1 = self.board.candidates[pair[0]][col]
                cand2 = self.board.candidates[pair[1]][col]
                if cand1 == cand2:
                    r1 = pair[0]
                    r2 = pair[1]
                    val1 = list(cand1)[0]
                    val2 = list(cand1)[1]
                    newPair = CellPair([r1, col], [r2, col], set([val1, val2]))
                    if newPair not in self.knownPairs:
                        self.board.createPair(r1, col, r2, col, val1, val2)
                        return r1, col, r2, col, val1, val2

        # Check for box pairs
        boxes = [self.board.getBoxIndices(row, col) for row, col in self.board.getAllBoxBeginnings()]
        for box in boxes:
            # Count the number of candidates in each cell
            cell_candidate_count = [len(self.board.candidates[row][col]) for row, col in box]

            # Limit to only cells with exactly two candidates
            two_candidates = [box[cell] for cell in range(len(box)) if cell_candidate_count[cell] == 2]

            # Need at least two cells with exactly two candidates to check for naked pairs
            if len(two_candidates) < 2:
                continue

            # Check all possible pairs. If any two cells are limited to the same two candidates, and this is not
            # a known pair, create a new pair and return
            possible_pairs = pair_permutations(two_candidates)
            for pair in possible_pairs:
                r1 = pair[0][0]
                c1 = pair[0][1]
                r2 = pair[1][0]
                c2 = pair[1][1]
                cand1 = self.board.candidates[r1][c1]
                cand2 = self.board.candidates[r2][c2]
                if cand1 == cand2:
                    val1 = list(cand1)[0]
                    val2 = list(cand1)[1]
                    newPair = CellPair([r1, c1], [r2, c2], set([val1, val2]))
                    if newPair not in self.knownPairs:
                        self.board.createPair(r1, c1, r2, c2, val1, val2)
                        return r1, c1, r2, c2, val1, val2

        return None, None, None, None, None, None

    def HiddenPair(self) -> tuple[int, int, int, int, int, int]:
        """ Try to apply the "Hidden Pairs" rule to the board

        The hidden pairs rule applies when two candidates appear in exactly two
        cells in the same row, column, or box. These two candidates can then be
        removed as candidates from any cell that both cells of the pair can "see"

        Returns
        -------
        row1, col1 : `int`
            The first cell in the pair created
        row2, col2 : `int`
            The second cell in the pair created
        val1, val2 : `int`
            The values the cells are a candidate pair for

        Notes
        -----
        Returns will be (None, None, None, None, None, None) if the rule could not be applied
        """
        for row in range(self.board.BOARD_SIZE): # Check for row pairs
            row_candidate_count = {candidate: 0 for candidate in self.board.DEFAULT_CANDIDATES}
            row_candidate_cols = {candidate: [] for candidate in self.board.DEFAULT_CANDIDATES}

            for col in range(self.board.BOARD_SIZE):
                for candidate in self.board.candidates[row][col]:
                    row_candidate_count[candidate] += 1
                    row_candidate_cols[candidate].append(col)

            # Find candidates with two possible cells
            two_spots = [candidate for candidate,count in row_candidate_count.items() if count == 2]

            # Only one candidate with two possible cells. A pair requires two candidates in the same two cells
            if len(two_spots) < 2:
                continue

            # Use list comprehension to create all possible pairs of candidates from the two_spots list
            possible_pairs = pair_permutations(two_spots)

            for candidate_pair in possible_pairs:
                if row_candidate_cols[candidate_pair[0]] == row_candidate_cols[candidate_pair[1]]:
                    c1 = row_candidate_cols[candidate_pair[0]][0]
                    c2 = row_candidate_cols[candidate_pair[0]][1]
                    val1 = candidate_pair[0]
                    val2 = candidate_pair[1]
                    newPair = CellPair([row, c1], [row, c2], set([val1, val2]))
                    if newPair not in self.knownPairs:
                        self.board.createPair(row, c1, row, c2, val1, val2)
                        return row, c1, row, c2, val1, val2

        for col in range(self.board.BOARD_SIZE): # Check for col pairs
            col_candidate_count = {candidate: 0 for candidate in self.board.DEFAULT_CANDIDATES}
            col_candidate_cols = {candidate: [] for candidate in self.board.DEFAULT_CANDIDATES}

            for row in range(self.board.BOARD_SIZE):
                for candidate in self.board.candidates[row][col]:
                    col_candidate_count[candidate] += 1
                    col_candidate_cols[candidate].append(row)

            # Find candidates with two possible cells
            two_spots = [candidate for candidate,count in col_candidate_count.items() if count == 2]

            # Only one candidate with two possible cells. A pair requires two candidates in the same two cells
            if len(two_spots) < 2:
                continue

            # Use list comprehension to create all possible pairs of candidates from the two_spots list
            possible_pairs = pair_permutations(two_spots)

            for candidate_pair in possible_pairs:
                if col_candidate_cols[candidate_pair[0]] == col_candidate_cols[candidate_pair[1]]:
                    r1 = col_candidate_cols[candidate_pair[0]][0]
                    r2 = col_candidate_cols[candidate_pair[0]][1]
                    val1 = candidate_pair[0]
                    val2 = candidate_pair[1]
                    newPair = CellPair([r1, col], [r2, col], set([val1, val2]))
                    if newPair not in self.knownPairs:
                        self.board.createPair(r1, col, r2, col, val1, val2)
                        return r1, col, r2, col, val1, val2

        boxes = [self.board.getBoxIndices(row, col) for row, col in self.board.getAllBoxBeginnings()]
        for box in boxes: # Check for box pairs
            box_candidate_count = {candidate: 0 for candidate in self.board.DEFAULT_CANDIDATES}
            box_candidate_rows = {candidate: [] for candidate in self.board.DEFAULT_CANDIDATES}
            box_candidate_cols = {candidate: [] for candidate in self.board.DEFAULT_CANDIDATES}
            for row, col in box:
                for candidate in self.board.candidates[row][col]:
                    box_candidate_count[candidate] += 1
                    box_candidate_rows[candidate].append(row)
                    box_candidate_cols[candidate].append(col)

            # Find candidates with two possible cells
            two_spots = [candidate for candidate,count in box_candidate_count.items() if count == 2]

            # Only one candidate with two possible cells. A pair requires two candidates in the same two cells
            if len(two_spots) < 2:
                continue

            # Use list comprehension to create all possible pairs of candidates from the two_spots list
            possible_pairs = pair_permutations(two_spots)

            for candidate_pair in possible_pairs:
                val1 = candidate_pair[0]
                val2 = candidate_pair[1]
                if box_candidate_rows[val1] == box_candidate_rows[val2] and box_candidate_cols[val1] == box_candidate_cols[val2]:
                    r1 = box_candidate_rows[val1][0]
                    r2 = box_candidate_rows[val1][1]
                    c1 = box_candidate_cols[val1][0]
                    c2 = box_candidate_cols[val1][1]
                    newPair = CellPair([r1, c1], [r2, c2], set([val1, val2]))
                    if newPair not in self.knownPairs:
                        self.board.createPair(r1, c1, r2, c2, val1, val2)
                        return r1, c1, r2, c2, val1, val2

        return None, None, None, None, None, None

    def LockedCandidatesPointing(self) -> bool:
        """ Try to apply the "Locked Candidates - Pointing" rule to the board

        This rule applies when a candidate is locked to only one row or column
        within a box. That candidate can be removed from the rest of that row or column.

        Returns
        -------
        applied : `bool`
            Whether or not this rule was able to limit candidates
        """
        boxes = [self.board.getBoxIndices(row, col) for row, col in self.board.getAllBoxBeginnings()]
        for b in range(len(boxes)):
            box = boxes[b]
            candidate_counts = {candidate: 0 for candidate in self.board.DEFAULT_CANDIDATES}
            candidate_rows = {candidate: set() for candidate in self.board.DEFAULT_CANDIDATES}
            candidate_cols = {candidate: set() for candidate in self.board.DEFAULT_CANDIDATES}
            for row, col in box:
                for candidate in self.board.candidates[row][col]:
                    candidate_counts[candidate] += 1
                    candidate_rows[candidate].add(row)
                    candidate_cols[candidate].add(col)

            for candidate, count in candidate_counts.items():
                if count > 0:
                    if len(candidate_rows[candidate]) == 1:
                        row = list(candidate_rows[candidate])[0]
                        newCandidate = [b, row, candidate]
                        if newCandidate not in self.knownPointingRow:
                            self.board.removeCandidateFromRow(row, list(candidate_cols[candidate]), candidate)
                            if self.verbose:
                                print("LockedCandidatesPointing technique used to limit candidates for row {} to a single box for candidate {}".format(row + 1, candidate))
                            self.knownPointingRow.append(newCandidate)
                            return True
                    if len(candidate_cols[candidate]) == 1:
                        col = list(candidate_cols[candidate])[0]
                        newCandidate = [b, col, candidate]
                        if newCandidate not in self.knownPointingCol:
                            self.board.removeCandidateFromCol(col, list(candidate_rows[candidate]), candidate)
                            if self.verbose:
                                print("LockedCandidatesPointing technique used to limit candidates for column {} to a single box for candidate {}".format(col + 1, candidate))
                            self.knownPointingCol.append(newCandidate)
                            return True

        return False

    def LockedCandidatesClaimed(self) -> bool:
        """ Try to apply the "Locked Candidates - Claimed" rule to the board

        This rule applies when a candidate is locked to only one box within a row
        or column. That candidate can be removed from the rest of that box.

        Returns
        -------
        applied : `bool`
            Whether or not this rule was able to limit candidates
        """
        boxes = [self.board.getBoxIndices(row, col) for row, col in self.board.getAllBoxBeginnings()]

        # Check across rows
        for row in range(self.board.BOARD_SIZE):
            row_candidate_counts = {candidate: 0 for candidate in self.board.DEFAULT_CANDIDATES}
            row_candidate_cols = {candidate: set() for candidate in self.board.DEFAULT_CANDIDATES}

            for col in range(self.board.BOARD_SIZE):
                for candidate in self.board.candidates[row][col]:
                    row_candidate_counts[candidate] += 1
                    row_candidate_cols[candidate].add(col)

            # Filter to non-zero counts
            row_candidate_counts = {candidate: count for candidate, count in row_candidate_counts.items() if count > 1}

            for candidate in row_candidate_counts.keys():
                cols = row_candidate_cols[candidate]
                if len(cols) <= 1:
                    continue
                box = None
                same = True

                # Check if all columns for the candidate in this row are in the same box
                for col in row_candidate_cols[candidate]:
                    for b in range(len(boxes)):
                        if [row, col] in boxes[b]:
                            if box == None:
                                box = b
                            elif b != box:
                                same = False
                                break
                    if not same:
                        break

                # Limit candidates and log to console if rule can be applied
                if same and box is not None:
                    exclude = [[row, col] for col in cols]
                    self.board.removeCandidateFromBox(boxes[box][0][0], boxes[box][0][1], exclude, candidate)
                    if self.verbose:
                        print("LockedCandidatesClaimed technique used to limit candidates for box {} to a row {} for candidate {}".format(box + 1, row + 1, candidate))
                    self.knownPointingRow.append([box, row, candidate])
                    return True

        # Check down columns
        for col in range(self.board.BOARD_SIZE):
            col_candidate_counts = {candidate: 0 for candidate in self.board.DEFAULT_CANDIDATES}
            col_candidate_rows = {candidate: set() for candidate in self.board.DEFAULT_CANDIDATES}

            for row in range(self.board.BOARD_SIZE):
                for candidate in self.board.candidates[row][col]:
                    col_candidate_counts[candidate] += 1
                    col_candidate_rows[candidate].add(row)

            # Filter to non-zero counts
            col_candidate_counts = {candidate: count for candidate, count in col_candidate_counts.items() if count > 1}

            for candidate in col_candidate_counts.keys():
                rows = col_candidate_counts[candidate]
                if len(cols) <= 1:
                    continue
                box = None
                same = True

                # Check if all rows for the candidate in this column are in the same box
                for row in col_candidate_rows[candidate]:
                    for b in range(len(boxes)):
                        if [row, col] in boxes[b]:
                            if box == None:
                                box = b
                            elif b != box:
                                same = False
                                break
                    if not same:
                        break

                # Limit candidates and log to console if rule can be applied
                if same and box is not None:
                    exclude = [[row, col] for row in rows]
                    self.board.removeCandidateFromBox(boxes[box][0][0], boxes[box][0][1], exclude, candidate)
                    if self.verbose:
                        print("LockedCandidatesClaimed technique used to limit candidates for box {} to a column {} for candidate {}".format(box + 1, col + 1, candidate))
                    self.knownPointingCol.append([box, col, candidate])
                    return True

        return False

class SudokuGame:
    """A class to handle the interactive game functionality of this sudoku program

    Parameters
    ----------
    board : `Board`
        The sudoku board this solver will be in charge of solving
    """
    board : Board

    def __init__(self, board : Board):
        self.board = board

    def help(self) -> None:
        """ Print help to the player """
        print("Commands: ")
        print("\thelp                   - Displays this help")
        print("\tset <row> <col> <val>  - Set the cell at [row, col] to val")
        print("\tclue                   - Use the solver to print the next move")
        print("\tcheck                  - Check the current board for any invalid entries or if the board is solved")
        print("\tsolve                  - Solve the board from the current state")
        print("\tcandidates <row> <col> - Get the possible candidates for the cell at [row, col]")
        print("\tclear                  - Clear the board and start a new one")
        print("\tprint                  - Print the board to the screen")
        print("\tquit                   - Exit the program")
        print("Valid row values: {}".format([i + 1 for i in range(self.board.BOARD_SIZE)]))
        print("Valid col values: {}".format([i + 1 for i in range(self.board.BOARD_SIZE)]))
        print("Valid candidate values: {}".format(self.board.DEFAULT_CANDIDATES))

    def getCommand() -> list[str]:
        """Get the next command from the user

        Returns
        -------
        command : `list[str]`
            The command that the user entered and the args that the user entered
        """
        validCommands = ["help", "set", "clue", "check", "solve", "candidates", "clear", "print", "quit"]
        while True:
            commandStr = input("Enter a command. Type `help` for syntax: ")
            command = commandStr.split(" ")
            if command[0] in validCommands:
                return command
            else:
                print("Invalid command {}".format(command[0]))

    def checkBoard(self) -> None:
        """ Check if the board is solved or if it has any mistakes """

        mistakes = False

        # Check for row mistakes:
        for row in range(self.board.BOARD_SIZE):
            vals = {val: 0 for val in self.board.DEFAULT_CANDIDATES}
            for val in self.board.getValuesInRow(row):
                if val is not None:
                    vals[val] += 1
            counts_above_one = [val for val, count in vals.items() if count > 1]
            if len(counts_above_one) > 0:
                print("The following values were specified more than once in row {}: {}".format(row + 1, counts_above_one))
                mistakes = True

        # Check for col mistakes:
        for col in range(self.board.BOARD_SIZE):
            vals = {val : 0 for val in self.board.DEFAULT_CANDIDATES}
            for val in self.board.getValuesInCol(col):
                if val is not None:
                    vals[val] += 1
            counts_above_one = [val for val, count in vals.items() if count > 1]
            if len(counts_above_one) > 0:
                print("The following values were specified more than once in col {}: {}".format(col + 1, counts_above_one))
                mistakes = True

        # Check for box mistakes:
        boxes = self.board.getAllBoxBeginnings()
        for b in range(len(boxes)):
            box = boxes[b]
            vals = {val: 0 for val in self.board.DEFAULT_CANDIDATES}
            for val in self.board.getValuesInBox(box[0], box[1]):
                if val is not None:
                    vals[val] += 1
            counts_above_one = [val for val, count in vals.items() if count > 1]
            if len(counts_above_one) > 0:
                print("The following values were specified more than once in box {}: {}".format(b + 1, counts_above_one))
                mistakes = True

        if not mistakes and self.board.isSolved():
            print("The board seems to be solved correctly!")
        elif not mistakes:
            print("Looks good so far!")

    def play(self) -> None:
        # Continue playing until the user wants to stop
        quit = False
        while not quit:
            command = SudokuGame.getCommand()
            if command[0] == "help":
                self.help()
            elif command[0] == "quit":
                quit = True
            elif command[0] == "set":
                if len(command) != 4:
                    print("Invalid syntax for `set` command")
                else:
                    row = int(command[1]) - 1
                    col = int(command[2]) - 1
                    val = int(command[3])
                    if row not in range(self.board.BOARD_SIZE):
                        print("Specified row out of range.")
                    elif col not in range(self.board.BOARD_SIZE):
                        print("Specified col out of range.")
                    elif val not in self.board.DEFAULT_CANDIDATES:
                        print("Specified value out of range.")
                    else:
                        self.board.setCell(row, col, val)
                        print("Cell at row {} and col {} set to {}.".format(row, col, val))
                        print(self.board)
            elif command[0] == "clue":
                # Make a copy of the board so the solver doesn't change the actual board the player is using
                solver = Solver(copy(self.board))
                solver.nextMove()
            elif command[0] == "check":
                self.checkBoard()
            elif command[0] == "solve":
                solver = Solver(self.board)
                solver.solve()
            elif command[0] == "candidates":
                if len(command) != 3:
                    print("Invalid syntax for `candidates` command")
                else:
                    row = int(command[1]) - 1
                    col = int(command[2]) - 1
                    if row not in range(self.board.BOARD_SIZE):
                        print("Specified row out of range.")
                    elif col not in range(self.board.BOARD_SIZE):
                        print("Specified col out of range.")
                    else:
                        print("Possible values for cell [{}, {}] are: {}".format(row, col, self.board.candidates[row][col]))
            elif command[0] == "clear":
                self.board = Board()
                print(self.board)
            elif command[0] == "print":
                print(self.board)
            else:
                print("Invalid command {}.".format(command[0]))

        print("Quitting.")

def printUsage():
    print("SudokuSolver.py [-h] [-b <size>] [-c <path>] [-d <char>] [-s]")
    print("\t-h, --help                 - Print program usage")
    print("\t-b <n>, --board-size <n>   - Set the board size of the sudoku board. The board will have n rows and n columns (n x n)")
    print("\t-c <path>, --csv <path>    - The path to the CSV file to read to populate the initial board state")
    print("\t-d <char>, --delim <char>  - The delimiter to use when parsing the CSV file. Default is ';'")
    print("\t-s, --solve                - Automatically solve the board instead of using the board interactively")

def main(argv:list[str]):
    board_size: int = None
    csvfile: str = None
    delim: str = None
    solve: bool = False

    # Parse the command line arguments
    try:
        opts, args = getopt.getopt(argv, "hb:c:d:s", ["help", "board-size=", "csv=", "delim=", "solve"])
    except getopt.GetoptError as e:
        print("Error parsing command line arguments: {}".format(e), file=sys.stderr)
        printUsage()
        sys.exit(2)

    # Process the parsed command line arguments
    for opt, arg in opts:
        if opt in ('-h', "--help"):
            printUsage()
            sys.exit(0)
        elif opt in ("-b", "--board-size"):
            board_size = int(arg)
        elif opt in ("-c", "--csv"):
            csvfile = arg
        elif opt in ("-d", "--delim"):
            delim = arg
        elif opt in ("-s", "--solve"):
            solve = True

    # Print the interpreted command line options to the user
    print("Board Size: {}".format(board_size))
    print("CSVFile: {}".format(csvfile))
    print("Delimiter: {}".format(delim))
    print("Solve: {}".format(solve))

    if csvfile is not None:
        b = Board.fromCsv(csvfile, board_size, delim)
    else:
        b = Board(board_size)
    print(b)
    solver = Solver(b)
    if solve:
        solver.solve()
    else:
        print("Running in interactive mode")
        game = SudokuGame(b)
        game.play()

if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except ValueError as e:
        print(e, file=sys.stderr)
        print("Exiting with error code 1.")
        exit(1)
