import numpy as np
import numpy.typing as npt


class SudokuCell:
    def __init__(self, row_ind, col_ind, value) -> None:
        self.value = value
        self.possible_values = set()
        self.brute_value = None
        self.row_ind = row_ind
        self.col_ind = col_ind

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f'{str(self.value)}, POS: {self.possible_values}, CORDS: {self.row_ind}:{self.col_ind}'


class SudokuSolver:
    def __init__(self, sudoku, box_size=3):
        self.size = len(sudoku)
        self.box_size = box_size
        self.allowed_values = {str(value) for value in range(1, self.size+1)}
        for row_ind in range(self.size):
            sudoku[row_ind] = [SudokuCell(row_ind, col_ind, cell) for col_ind, cell in enumerate(sudoku[row_ind])]

        self.sudoku: npt.ArrayLike[SudokuCell] = np.array(sudoku)
        self.sudoku_list: list[SudokuCell] = np.hstack(self.sudoku).tolist()

    def get_related_row(self, cell: SudokuCell, include_origin=False):
        result = [row_cell for row_cell in self.sudoku[cell.row_ind]]
        if not include_origin:
            result.remove(cell)
        return result

    def get_related_col(self, cell: SudokuCell, include_origin=False):
        result = [col_cell[cell.col_ind] for col_cell in self.sudoku]
        if not include_origin:
            result.remove(cell)
        return result

    def get_related_box(self, cell: SudokuCell, include_origin=False):
        row_box_ind = cell.row_ind // self.box_size
        col_box_ind = cell.col_ind // self.box_size
        box = self.sudoku[row_box_ind * self.box_size:row_box_ind * self.box_size + self.box_size,
                          col_box_ind * self.box_size:col_box_ind * self.box_size + self.box_size]
        result = np.hstack(box).tolist()
        if not include_origin:
            result.remove(cell)
        return result

    def get_related_cells(self, cell, cells_type='all', include_origin=False):
        if cells_type == 'row':
            return set(self.get_related_row(cell, include_origin))
        elif cells_type == 'col':
            return set(self.get_related_col(cell, include_origin))
        elif cells_type == 'box':
            return set(self.get_related_box(cell, include_origin))
        elif cells_type == 'all':
            return set(self.get_related_row(cell, include_origin) +
                       self.get_related_col(cell, include_origin) +
                       self.get_related_box(cell, include_origin))
        else:
            raise TypeError(f"Unsupported cells type: {cells_type}. Supported types: row, col, box, all.")

    def get_related_values(self,
                           cell: SudokuCell,
                           cells_type='all'):
        return {rel_cell.value for rel_cell in self.get_related_cells(cell, cells_type=cells_type) if rel_cell.value}

    def get_related_pos_values(self,
                               cell: SudokuCell,
                               cells_type='all'):
        result = set()
        for rel_cell in self.get_related_cells(cell, cells_type=cells_type):
            result.update(rel_cell.possible_values)
        return result

    def get_brute_values(self,
                         cell: SudokuCell,
                         cells_type='all',
                         get_values=False):
        result = set()
        for rel_cell in self.get_related_cells(cell, cells_type=cells_type):
            if not rel_cell.value:
                result.add(rel_cell.brute_value)
            elif get_values:
                result.add(rel_cell.value)
        result.discard(None)
        return result

    def is_pos_val_single_line(self,
                               cell: SudokuCell,
                               value: str):
        box_cells = self.get_related_cells(cell, cells_type='box', include_origin=True)
        box_cells = list(filter(lambda box_cell: value in box_cell.possible_values, box_cells))

        if box_cells:
            first_cell = box_cells[0]
            first_x, first_y = first_cell.row_ind, first_cell.col_ind

            is_single_row = all(box_cell.row_ind == first_x for box_cell in box_cells)
            is_single_col = all(box_cell.col_ind == first_y for box_cell in box_cells)
            if is_single_row and is_single_col:
                return 'all'
            elif is_single_row:
                return 'row'
            elif is_single_col:
                return 'col'

    def remove_possible_value(self,
                              cell: SudokuCell,
                              cells_type='all',
                              values=None,
                              exclude_box=False,
                              exclude_list=[]):
        if not values:
            values = [cell.value]
        elif isinstance(values, str):
            values = list(values)

        box_cells = set()
        if exclude_box:
            box_cells = self.get_related_cells(cell, cells_type='box')

        related_cells: set[SudokuCell] = self.get_related_cells(cell, cells_type=cells_type)
        for rel_cell in related_cells:
            if rel_cell not in box_cells and rel_cell not in exclude_list:
                for value in values:
                    rel_cell.possible_values.discard(value)

    def bruteforce_sudoku(self):
        cell_num = 0
        backwards = False

        for cell in self.sudoku_list:
            cell.possible_values = set()

        while cell_num >= 0 and cell_num <= len(self.sudoku_list) - 1:
            curr_cell = self.sudoku_list[cell_num]

            # Если есть значение - идем дальше
            if curr_cell.value:
                if backwards:
                    cell_num -= 1
                else:
                    cell_num += 1
                continue

            # Выставляем возможные значения, если мы не откатываемся назад
            if not curr_cell.possible_values and not backwards:
                curr_cell.possible_values = self.allowed_values - self.get_brute_values(curr_cell, get_values=True)

            # Если еще есть возможные значения - пробуем их
            if curr_cell.possible_values:
                curr_cell.brute_value = min(curr_cell.possible_values)
                curr_cell.possible_values.discard(curr_cell.brute_value)
                backwards = False
                cell_num += 1
            # Если возможные значения закончились - откатываемся назад
            else:
                curr_cell.brute_value = None
                backwards = True
                cell_num -= 1
                continue

        else:
            for cell in self.sudoku_list:
                if not cell.value:
                    cell.value = cell.brute_value

    def solve_sudoku(self):
        tries = 0
        changed = True

        while changed and tries <= 10:
            changed = False
            sub_changed = True

            # Выставляем возможные значения
            for cell in self.sudoku_list:
                if not cell.value:
                    possible_values = self.allowed_values - self.get_related_values(cell)
                    print(f'{self.allowed_values} - {self.get_related_values(cell)} = {possible_values}')
                    if cell.possible_values != possible_values:
                        print(f'changed possible value from {cell.possible_values} to {possible_values}')
                        cell.possible_values = possible_values

            # Проверяем блок, если возможное значение только в этом ряду/столбце
            for cell in self.sudoku_list:
                if not cell.value:
                    for pos_value in cell.possible_values:
                        single_attr = self.is_pos_val_single_line(cell, pos_value)
                        if single_attr:
                            self.remove_possible_value(cell, cells_type=single_attr,
                                                       values=[pos_value], exclude_box=True)

            # Naked pair
            for cell in self.sudoku_list:
                if len(cell.possible_values) == 2:
                    related_cells = self.get_related_cells(cell)
                    related_cells = list(filter(lambda rel_cell: rel_cell.possible_values ==
                                                cell.possible_values, related_cells))
                    # Если находим две ячейки с одинаковыми двумя возможными значениями
                    if len(related_cells) == 1:
                        sec_cell = related_cells[0]
                        # Удаляем возможные значения с совпадающей строчке или строке
                        if cell.row_ind == sec_cell.row_ind:
                            self.remove_possible_value(cell, cells_type='row',
                                                       values=cell.possible_values, exclude_list=[sec_cell])
                        elif cell.col_ind == sec_cell.col_ind:
                            self.remove_possible_value(cell, cells_type='col',
                                                       values=cell.possible_values, exclude_list=[sec_cell])

                        # Если они еще и в одном блоке, то удаляем значения в этом блоке тоже
                        if sec_cell in self.get_related_cells(cell, cells_type='box'):
                            self.remove_possible_value(cell, cells_type='box',
                                                       values=cell.possible_values, exclude_list=[sec_cell])

            while sub_changed:
                print('values')
                sub_changed = False
                # Выставляем значения ячеек на основе возможных значений
                for cell in self.sudoku_list:
                    if cell.value:
                        continue
                    # Если только одно возможное значение в ячейке
                    elif len(cell.possible_values) == 1:
                        cell.value = cell.possible_values.pop()
                        self.remove_possible_value(cell)
                        changed = True
                    elif cell.possible_values:
                        for possible_value in cell.possible_values:
                            # Если возможное значение встречается только здесь в строке/столбце/блоке
                            if possible_value not in self.get_related_pos_values(cell, cells_type='row') or \
                                    possible_value not in self.get_related_pos_values(cell, cells_type='col') or \
                                    possible_value not in self.get_related_pos_values(cell, cells_type='box'):
                                cell.value = possible_value
                                cell.possible_values.clear()
                                self.remove_possible_value(cell)

                                sub_changed = True
                                changed = True
                                break

            print(tries+1)
            # self.print_possible_result()
            # self.print_simple_result()
            tries += 1

        if not changed or tries == 10:
            # if logic didnt help try to bruteforce
            print("Can't solve by logic. Trying to bruteforce.")
            self.bruteforce_sudoku()

        return self.sudoku.tolist()

    def print_simple_result(self, brute_values=False):
        result_str = ''
        x, y = 1, 1
        horizontal_row = '-'*(self.size*3 + self.size // self.box_size + 1)
        result_str += f"\n{horizontal_row}\n"
        for row in self.sudoku:
            result_str += '|'
            for cell in row:
                if cell.value:
                    result_str += f' {cell.value} '
                elif brute_values and cell.brute_value:
                    result_str += f' {cell.brute_value} '
                else:
                    result_str += ' . '

                if x % self.box_size == 0:
                    result_str += '|'
                x += 1
            if y % 3 == 0:
                result_str += f"\n{horizontal_row}"
            result_str += '\n'
            y += 1
        print(result_str)

    def print_possible_result(self):
        print(
            '',
            '--------------',
            '| 1  2  .    |',
            '| 4  .  6    | --> Possible values',
            '| .  .  .    |',
            '|          . | --> Value',
            '--------------',
            sep='\n'
        )
        result_str = ''
        row_strings = self.size // self.box_size
        horizontal_row = '-'*(self.size * 3 * 4 + self.size + 2 + self.size // self.box_size)
        result_str += f"\n{horizontal_row}\n{horizontal_row}\n"
        for row_ind in range(len(self.sudoku)):
            # result_str += '|'
            for str_row_index in range(row_strings+1):
                str_row = '||'
                for col_ind, cell in enumerate(self.sudoku[row_ind]):
                    if str_row_index == row_strings:
                        if cell.value:
                            str_row += f"{'   '*self.box_size} {cell.value} |"
                        else:
                            str_row += f"{'   '*self.box_size} . |"
                    else:
                        for allowed in sorted(self.allowed_values)[str_row_index*self.box_size:(str_row_index+1)*self.box_size]:
                            if allowed in cell.possible_values:
                                str_row += f' {allowed} '
                            else:
                                str_row += ' . '
                        str_row += '   |'
                    if (col_ind + 1) % self.box_size == 0:
                        str_row += '|'
                str_row += '\n'
                result_str += str_row

            result_str += f'{horizontal_row}\n'
            if (row_ind + 1) % self.box_size == 0:
                result_str += f'{horizontal_row}\n'

        print(result_str)


if __name__ == "__main__":
    from pprint import pprint
    test_sudoku = [['8', '', '', '', '', '', '', '', ''],
                   ['', '', '3', '6', '', '', '', '', ''],
                   ['', '7', '', '', '9', '', '2', '', ''],
                   ['', '5', '', '', '', '7', '', '', ''],
                   ['', '', '', '', '4', '5', '7', '', ''],
                   ['', '', '', '1', '', '', '', '3', ''],
                   ['', '', '1', '', '', '', '', '6', '8'],
                   ['', '', '8', '5', '', '', '', '1', ''],
                   ['', '9', '', '', '', '', '4', '', '']]

    su = SudokuSolver(test_sudoku)
    result = su.solve_sudoku()
    su.print_simple_result()
