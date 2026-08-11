import numpy as np

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

def crossbar_distributed(R, V_in, Rrow_seg, Rcol_seg, Rrow_in=1e-6, Rcol_out=1e-6):
    """
    Расчёт напряжений и токов в кроссбаре с распределёнными сопротивлениями линий.

    Параметры:
        R : np.ndarray (N, M) - сопротивления элементов матрицы (Ом)
        V_in : np.ndarray (N,) - входные напряжения на строках (В)
        Rrow_seg : float - сопротивление сегмента строки между соседними столбцами
        Rcol_seg : float - сопротивление сегмента столбца между соседними строками
        Rrow_in : float - сопротивление на входе каждой строки (последовательно с источником)
        Rcol_out : float - сопротивление на выходе каждого столбца (на землю)

    Возвращает:
        V_row_nodes : np.ndarray (N, M) - напряжения на узлах строк S_{i,j}
        V_col_nodes : np.ndarray (N, M) - напряжения на узлах столбцов C_{i,j}
        I_elem : np.ndarray (N, M) - токи через резисторы матрицы
        I_col : np.ndarray (M,) - токи столбцов (через Rcol_out)
    """
    N, M = R.shape
    # Общее число узлов: для каждого пересечения два узла (строка и столбец)
    num_nodes = 2 * N * M
    # Удобные индексы:
    # узел строки S_{i,j} -> idx = i*M + j
    # узел столбца C_{i,j} -> idx = N*M + i*M + j
    def S_idx(i, j): return i * M + j
    def C_idx(i, j): return N * M + i * M + j

    # Матрица проводимости (разреженная)
    A = lil_matrix((num_nodes, num_nodes))
    b = np.zeros(num_nodes)

    # Проводимости (g = 1/R)
    g_row_seg = 1.0 / Rrow_seg if Rrow_seg != 0 else 0.0
    g_col_seg = 1.0 / Rcol_seg if Rcol_seg != 0 else 0.0
    g_row_in = 1.0 / Rrow_in if Rrow_in != 0 else 0.0
    g_col_out = 1.0 / Rcol_out if Rcol_out != 0 else 0.0

    # -------- Уравнения для узлов строк --------
    for i in range(N):
        for j in range(M):
            idx = S_idx(i, j)
            # Ток через резистор матрицы: (V_S - V_C) / R_ij
            g_ij = 1.0 / R[i, j] if R[i, j] != 0 else 0.0
            A[idx, idx] += g_ij
            A[idx, C_idx(i, j)] -= g_ij

            # Горизонтальные связи (соседние столбцы)
            if j > 0:
                A[idx, S_idx(i, j-1)] -= g_row_seg
                A[idx, idx] += g_row_seg
            if j < M-1:
                A[idx, S_idx(i, j+1)] -= g_row_seg
                A[idx, idx] += g_row_seg

            # Вертикальные связи отсутствуют для узлов строк (они только вдоль строки)
            # Но строки могут быть соединены через столбцы, это учтено через резисторы матрицы

    # -------- Уравнения для узлов столбцов --------
    for i in range(N):
        for j in range(M):
            idx = C_idx(i, j)
            g_ij = 1.0 / R[i, j] if R[i, j] != 0 else 0.0
            A[idx, idx] += g_ij
            A[idx, S_idx(i, j)] -= g_ij

            # Вертикальные связи (соседние строки)
            if i > 0:
                A[idx, C_idx(i-1, j)] -= g_col_seg
                A[idx, idx] += g_col_seg
            if i < N-1:
                A[idx, C_idx(i+1, j)] -= g_col_seg
                A[idx, idx] += g_col_seg

            # Горизонтальные связи отсутствуют для узлов столбцов

    # -------- Граничные условия --------
    # 1) На левом краю каждой строки (перед первым столбцом) подключаем источник через Rrow_in
    #    Для простоты подключим источник к узлу S_{i,0} через Rrow_in.
    #    Уравнение: (V_in[i] - V_S) / Rrow_in = ток, втекающий в узел S_{i,0}
    #    Это даёт добавку в матрицу: g_row_in * V_S - ... = V_in[i] * g_row_in
    for i in range(N):
        idx = S_idx(i, 0)
        A[idx, idx] += g_row_in
        b[idx] += V_in[i] * g_row_in

    # 2) На нижнем краю каждого столбца (после последней строки) подключаем Rcol_out на землю
    #    Уравнение: V_C / Rcol_out = ток, стекающий с узла C_{N-1,j} на землю
    #    Добавляем проводимость к земле: g_col_out * V_C = 0 (правая часть 0)
    for j in range(M):
        idx = C_idx(N-1, j)
        A[idx, idx] += g_col_out
        # b[idx] остаётся 0

    # Решение системы
    A_csr = csr_matrix(A)
    x = spsolve(A_csr, b)

    # Извлечение результатов
    V_S = np.zeros((N, M))
    V_C = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            V_S[i, j] = x[S_idx(i, j)]
            V_C[i, j] = x[C_idx(i, j)]

    # Токи через резисторы матрицы
    I_elem = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            if R[i, j] != 0:
                I_elem[i, j] = (V_S[i, j] - V_C[i, j]) / R[i, j]

    # Токи столбцов (через Rcol_out) – равны V_C[N-1, j] / Rcol_out
    I_col = V_C[N-1, :] / Rcol_out if Rcol_out != 0 else np.zeros(M)

    return V_S, V_C, I_elem, I_col


# ============ ПРИМЕР ИСПОЛЬЗОВАНИЯ ============
if __name__ == "__main__":
    # Размеры
    N, M = 3, 4
    # Однородная матрица сопротивлений
    R = np.full((N, M), 1000.0)   # все элементы 1 кОм
    # Одинаковые входные напряжения
    V_in = np.array([5.0, 5.0, 5.0])

    # Параметры линий (сегменты)
    Rrow_seg = 10.0    # сопротивление участка строки между столбцами
    Rcol_seg = 10.0    # сопротивление участка столбца между строками
    Rrow_in = 50.0     # сопротивление на входе строки (от источника к первому столбцу)
    Rcol_out = 100.0   # сопротивление на выходе столбца (на землю)

    V_S, V_C, I_elem, I_col = crossbar_distributed(R, V_in, Rrow_seg, Rcol_seg,
                                                   Rrow_in, Rcol_out)

    # print("Напряжения на узлах строк (S):\n", V_S)
    # print("\nНапряжения на узлах столбцов (C):\n", V_C)
    # print("\nТоки через элементы матрицы:\n", I_elem)
    print("\nТоки столбцов (через Rcol_out):", I_col)

    # Проверка: токи столбцов должны быть разными, несмотря на одинаковые R и V_in
    print("\nРазброс токов столбцов:", np.std(I_col))