from pprint import pprint
from find_sudoku_img import extract_sudoku
from solve_sudoku import SudokuSolver


def main(image_filename=None, sudoku_matrix=None):
    if image_filename:
        sudoku_matrix = extract_sudoku(image_filename)
    su = SudokuSolver(sudoku_matrix)
    result = su.solve_sudoku()  # list of lists
    su.print_possible_result()  # With possible cell values
    # su.print_simple_result()


if __name__ == "__main__":
    main('test_img3.jpg')
