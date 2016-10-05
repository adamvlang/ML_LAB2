from cvxopt.solvers import qp
from cvxopt.base import matrix
import cvxopt

import numpy, pylab, random, math

def kernel(x, y):
    return numpy.dot(numpy.transpose(x), y)+1
    #return math.pow(numpy.dot(numpy.transpose(x),y)+1,2)
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
    classA = [[random.normalvariate(-1.5, 1), random.normalvariate(0.5, 1),1.0] for i in range(5)] + \
    [[random.normalvariate(1.5, 1), random.normalvariate(0.5, 1), 1.0] for i in range(5)]
    classB = [[random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0] for i in range(10)]
    data = classA + classB
    random.shuffle(data)
    return data

def indicator(new, data):
    ind = []
    data = [s for s in data if s[3] > math.pow(10.,-5)]
    for i in range(len(data)):
        ind.append(data[i][3]*data[i][2]*kernel(new, [data[i][0], data[i][1]]))
    return sum(ind)

def cluster_plot(A, B, data):
    pylab.hold(True)
    pylab.plot([p[0] for p in A], [p[1] for p in A], 'bo')
    pylab.plot([p[0] for p in B], [p[1] for p in B], 'ro')

    x_range = numpy.arange(-4, 4, 0.05)
    y_range = numpy.arange(-4, 4, 0.05)
    grid = matrix([[indicator([x,y], data) for y in y_range] for x in x_range])
    pylab.contour(x_range,y_range,grid,(-1.0,0.0,1.0),colors=('red','black','blue'),linewidths=(1,3,1))
    pylab.show()

def main():
    data = gen_rand_data()
    #data = [[1., 2., 1.], [7., 5., -1.], [2., 4., 1.], [3., 3., 1.], [-4., 4., -1.]]
    P = p_matix(data)
    q, h, G = build_qGh(len(data))
    r = qp(P, q, G, h)
    alpha = list(r['x'])
    for s in range(len(alpha)):
        data[s].append(alpha[s])
    
    pos = [s for s in data if s[2] > 0]
    neg = [s for s in data if s[2] < 0]

    
    cluster_plot(pos, neg, data)
main()
