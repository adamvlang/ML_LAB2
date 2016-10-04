from cvxopt.solvers import qp
from cvxopt.base import matrix
import cvxopt

import numpy, pylab, random, math

def kernel(x, y):
    return numpy.dot(numpy.transpose(x), y)+1

def p_matix(in_set):
    print('Creating P matrix...')
    P = []

    for i in range(len(in_set)):
        row_P = []
        x_i = []
        t_i = in_set[i][2]
        x_i.append(in_set[i][0])
        x_i.append(in_set[i][1])
        for j in range(len(in_set)):
            x_j = []
            t_j = in_set[j][2]
            x_j.append(in_set[j][0])
            x_j.append(in_set[j][1])
            row_P.append(t_i*t_j*kernel(x_i, x_j))
        P.append(row_P)
    return matrix(P)

def build_qGh(n):
    print('Building q, G, h...')
    q = matrix(-1., (n, 1))
    h = matrix(0., (n, 1))
    G = matrix(0., (n, n))
    G[::(n+1)] = -1

    return q, h, G

def gen_rand_data():
    classA = [(random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1),1.0) for i in range(5)] + \
    [(random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0) for i in range(5)]
    classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(10)]
    data = classA + classB
    random.shuffle(data)
    print(data)
    return data

def main():
    data = gen_rand_data()
    in_set = [[1., 2., 1.], [7., 5., -1.], [2., 4., 1.]]
    P = p_matix(data)
    q, h, G = build_qGh(len(data))
    print(q)
    r = qp(P, q, G, h)
    alpha = list(r['x'])





main()
