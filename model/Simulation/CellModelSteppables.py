from cc3d.core.PySteppables import *
import numpy as np
import random
import time
import ode_coupled


def generate_random_array(size, min_val, max_val):
    return [random.uniform(min_val, max_val) for _ in range(size)]


class CellModelSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.prev_time = 0  # time track for each step

        self.cell_count = 9
        self.boundary_matrix = np.zeros((self.cell_count, self.cell_count))

        self.calciumX = generate_random_array(self.cell_count, 0.1, 1)
        self.calciumY = generate_random_array(self.cell_count, 0.5, 1.3)
        print(f"\n\nStarting calcium X = {self.calciumX}")
        print(f"Starting calcium Y = {self.calciumY}\n\n")

    def start(self):
        self.create_scalar_field_cell_level_py("vis")
        self.prev_time = time.time() * 1000

    def step(self, mcs):
        current_time = time.time() * 1000
        elapsed_time = current_time - self.prev_time

        self.prev_time = current_time

        results = ode_coupled.solve_from(
            self.calciumX, self.calciumY, elapsed_time / 1000, self.cell_count, self.boundary_matrix)

        self.calciumX = results[0]
        self.calciumY = results[1]

        try:
            field = self.field.vis
            field.clear()
        except KeyError:
            self.create_scalar_field_py("vis")
            field = self.field.vis

        matrix = np.zeros((self.cell_count, self.cell_count))

        for cell in self.cell_list:
            # No need need to check nucleus boundaries
            if cell.type == self.NUCLEUS:
                field[cell] = self.calciumY[(cell.id - 2) % self.cell_count]
                continue

            boundary_pixels = self.get_cell_boundary_pixel_list(cell)

            for data in boundary_pixels:
                for (x, y, z) in ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)):
                    # Gets the cell at this relative pixel
                    nb_cell = self.cell_field[data.pixel.x +
                                              x, data.pixel.y + y, data.pixel.z + z]

                    # If this is a neighbour cell and its not a nucleus, update the count
                    if nb_cell and self.are_cells_different(cell, nb_cell) and nb_cell.type == self.CELL:
                        row_i = cell.clusterId - 1
                        col_i = nb_cell.clusterId - 1

                        matrix[col_i][row_i] = matrix[col_i][row_i] + 1

            field[cell] = self.calciumX[(cell.id - 1) % self.cell_count]

        m_min = np.min(matrix)
        m_max = np.max(matrix)

        if m_max - m_min != 0:
            norm_matrix = (matrix - m_min) / (m_max - m_min)
            self.boundary_matrix = norm_matrix

    def finish(self):
        pass

    def on_stop(self):
        pass
