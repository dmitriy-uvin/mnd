from math import sqrt
from random import randint

import pandas as pd
from numpy.linalg import det


class RomanovskyCriterion:
    m = 7
    average_y = []
    dispersion_y = []
    f_uv = []
    sigma_uv = []
    r_uv = []
    deviation = 0
    romanovsky_coef_value = 0
    criterion_table = {
        (2, 3, 4): 1.72,
        (5, 6, 7): 2.13,
        (8, 9): 2.37,
        (10, 11): 2.54,
        (12, 13): 2.66,
        (14, 15, 16, 17): 2.8,
        (18, 19, 20): 2.96
    }
    x1 = [-1, -1, 1]
    x2 = [-1, 1, -1]
    use_max = 1

    def __init__(self, var, x1_min, x1_max, x2_min, x2_max):
        self.x2_max = x2_max
        self.x2_min = x2_min
        self.x1_max = x1_max
        self.x1_min = x1_min
        self.var = var

        self.y_min = (20 - var) * 10
        self.y_max = (30 - var) * 10

        self.nx1 = [x1_min if self.x1[i] == -1 else x1_max for i in range(3)]
        self.nx2 = [x2_min if self.x2[i] == -1 else x2_max for i in range(3)]

        self.y_1 = [randint(self.y_min, self.y_max) for _ in range(self.m)]
        self.y_2 = [randint(self.y_min, self.y_max) for _ in range(self.m)]
        self.y_3 = [randint(self.y_min, self.y_max) for _ in range(self.m)]
        self.y_lists = [self.y_1, self.y_2, self.y_3]

    @staticmethod
    def __create_table(f_names, rows):
        n_df = {i: [] for i in f_names}
        for row in rows:
            for i, val in enumerate(n_df):
                n_df[val].append(row[i])

        df = pd.DataFrame(data=n_df)
        print(df)

    def __dispersion_calc(self, y_list, y_avg):
        return sum([(i - y_avg) ** 2 for i in y_list]) / self.m

    def __get_average_y(self):
        return [
            sum(self.y_1) / self.m,
            sum(self.y_2) / self.m,
            sum(self.y_3) / self.m
        ]

    def __get_dispersion_y(self):
        return [round(self.__dispersion_calc(self.y_lists[i], self.__get_average_y()[i]), 4) for i in range(3)]

    def __get_deviation(self):
        return sqrt((2 * (2 * self.m - 2)) / self.m * (self.m - 4))

    def __get_f_uv(self):
        uv = [
            [self.dispersion_y[0], self.dispersion_y[1]],
            [self.dispersion_y[1], self.dispersion_y[2]],
            [self.dispersion_y[2], self.dispersion_y[0]]
        ]
        return [round(max(uv[i]) / min(uv[i]), 4) for i in range(3)]

    def __get_sigma_coef(self):
        return [round(((self.m - 2) / self.m * f), 4) for f in self.f_uv]

    def __get_r_uv(self):
        return [round((abs(sigma - 1) / self.deviation), 4) for sigma in self.sigma_uv]

    def __is_romanovsky_criterion_exists(self) -> bool:
        self.average_y = self.__get_average_y()
        self.dispersion_y = self.__get_dispersion_y()
        self.deviation = self.__get_deviation()
        self.f_uv = self.__get_f_uv()
        self.sigma_uv = self.__get_sigma_coef()
        self.r_uv = self.__get_r_uv()

        for key in self.criterion_table.keys():
            if self.m in key:
                self.romanovsky_coef_value = self.criterion_table[key]
                break
            elif self.m >= 21 and self.use_max:
                print('M too big, we will available maximum')
                self.m = 20
            elif self.m >= 21 and not self.use_max:
                print('M too big. Exit!')
                exit()
        return max(self.r_uv) <= self.romanovsky_coef_value

    def execute(self):
        while not self.__is_romanovsky_criterion_exists():
            for i in self.y_lists:
                i.append((randint(self.y_min, self.y_max)))
            self.m += 1

        mx1, mx2, my = sum(self.x1) / 3, sum(self.x2) / 3, sum(self.average_y) / 3
        a1 = sum([i ** 2 for i in self.x1]) / 3
        a2 = sum([self.x1[i] * self.x2[i] for i in range(3)]) / 3
        a3 = sum([i ** 2 for i in self.x2]) / 3

        a11 = sum([self.x1[i] * self.average_y[i] for i in range(3)]) / 3
        a22 = sum([self.x2[i] * self.average_y[i] for i in range(3)]) / 3

        determinant = det([
            [1, mx1, mx2],
            [mx1, a1, a2],
            [mx2, a2, a3]
        ])
        b0 = det([
            [my, mx1, mx2],
            [a11, a1, a2],
            [a22, a2, a3]
        ]) / determinant
        b1 = det([
            [1, my, mx2],
            [mx1, a11, a2],
            [mx2, a22, a3]
        ]) / determinant
        b2 = det([
            [1, mx1, my],
            [mx1, a1, a11],
            [mx2, a2, a22]
        ]) / determinant

        delta_x1 = abs(self.x1_max - self.x1_min) / 2
        delta_x2 = abs(self.x2_max - self.x2_min) / 2
        x_10 = (self.x1_max + self.x1_min) / 2
        x_20 = (self.x2_max + self.x2_min) / 2

        nb0 = b0 - b1 * (x_10 / delta_x1) - b2 * (x_20 / delta_x2)
        nb1 = b1 / delta_x1
        nb2 = b2 / delta_x2

        f_names = ['X1', 'X2', *[f"Y{i}" for i in range(1, self.m + 1)]]
        rows = [[self.x1[i], self.x2[i], *self.y_lists[i]] for i in range(len(self.y_lists))]
        self.__create_table(f_names, rows)

        print('\n')

        f_names = ['AVG Y', 'Dispersion Y', 'F_uv', 'σ_uv', 'R_uv']
        rows = [[self.average_y[i], self.dispersion_y[i], self.f_uv[i], self.sigma_uv[i], self.r_uv[i]] for i in
                range(len(self.y_lists))]
        self.__create_table(f_names, rows)

        print('\n')

        f_names = ['NX1', 'NX2', 'AVG Y', 'Experimental']
        rows = [
            [
                self.nx1[i],
                self.nx2[i],
                self.average_y[i],
                round(nb0 + a1 * self.nx1[i] + a2 * self.nx2[i], 4)
            ] for i in range(len(self.y_lists))
        ]
        self.__create_table(f_names, rows)

        print('\n')

        print('Equation')
        print(f"y = {round(b0, 4)} + {round(b1, 4)}*x1 + {round(b2, 4)}*x2")

        print('Normalized')
        print(f"y = {round(nb0, 3)} + {round(nb1, 3)}*nx1 + {round(nb2, 3)}*nx2")

        print(f"Deviation: {self.deviation}")
        print(f"Romanovsky Criterion: {self.romanovsky_coef_value}")


variant = 223
x1_min = -30
x2_min = -15

x1_max = 0
x2_max = 35

romanovskyCriterion = RomanovskyCriterion(variant, x1_min, x2_min, x1_max, x2_max)
romanovskyCriterion.execute()

