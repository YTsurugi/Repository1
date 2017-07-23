
# coding: utf-8

# In[1]:

from scipy.optimize import fmin
import math

banana = lambda X, a: 100*(X[1] - X[0]**2)**2 + (a - X[0])**2
a = math.sqrt(2)
arg = (a, )
fmin(banana, [-1, 1.2], args=arg)


# In[2]:

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

seed = 0
np.random.seed(seed)

def y(x, a):
    return (x-3.*a) * (2.*x-a) * (3.*x+a) * (x+2.*a) / 100.

a_orig = 2.
xs = np.linspace(-5, 7, 1000)
ys = y(xs,a_orig)

num_data = 30
data_x = np.random.uniform(-5, 5, num_data)
data_y = y(data_x, a_orig) + np.random.normal(0, 0.5, num_data)

plt.plot(xs, ys, label='true a = %.2f'%(a_orig))
plt.plot(data_x, data_y, 'o', label='data')
plt.legend()


# In[4]:

plt.show()


# In[11]:

from scipy.optimize import least_squares

def calc_residuals(params, data_x, data_y):
    model_y = y(data_x, params[0])
    return model_y - data_y

a_init = -2
res = least_squares(calc_residuals, np.array([a_init]), args=(data_x, data_y))

a_fit = res.x[0]
ys_fit = y(xs,a_fit)

plt.plot(xs, ys, label='true a = %.2f'%(a_orig))
plt.plot(xs, ys_fit, label='fit a = %.2f'%(a_fit))
plt.plot(data_x, data_y, 'o')
plt.legend()


# In[12]:

plt.show()


# In[13]:

from scipy.optimize import basinhopping
a_init = -3.0
minimizer_kwargs = {"args":(data_x, data_y)}
res = basinhopping(calc_cost, np.array([a_init]),stepsize=2.,minimizer_kwargs=minimizer_kwargs)
print(res.x)

a_fit = res.x[0]
ys_fit = y(xs,a_fit)

plt.plot(xs, ys, label='true a = %.2f'%(a_orig))
plt.plot(xs, ys_fit, label='fit by basin-hopping a = %.2f'%(a_fit))
plt.plot(data_x, data_y, 'o')
plt.legend()


# In[3]:

import numpy as np, scipy.optimize as so
p = [31, 86, 29, 73, 46, 39, 58] # 利益 / 円
v = [10, 60, 25, 50, 35, 30, 40] # 分散 / 円
t = 50 # 目標利益 / 円
so.fmin_slsqp(lambda x: sum(v*x*x), np.zeros(len(p)), 
    eqcons=[lambda x: sum(x) - 1], ieqcons=[lambda x: sum(p*x) - t])
###
#Optimization terminated successfully.    (Exit mode 0)
#            Current function value: 4.50899167487
#            Iterations: 14
#            Function evaluations: 136
#            Gradient evaluations: 14
#array([ 0.26829785,  0.13279566,  0.09965076,  0.1343941 ,  0.11783349,
#        0.11506705,  0.13196109])


# In[5]:

import pulp

problem = pulp.LpProblem('sample', pulp.LpMinimize) # 最小化する場合
#problem = pulp.LpProblem('Problem Name', pulp.LpMaximize) # 最大化する場合

a = pulp.LpVariable('a', 0, 1) #(variable_name, min, max, variable_type)
b = pulp.LpVariable('b', 0, 1)

problem += a + b

problem += a >= 0
problem += b >= 0.1
problem += a + b == 0.5

status = problem.solve()
print ("Status", pulp.LpStatus[status])

print (problem)

print ("Result")
print ("a", a.value())
print ("b", b.value())


# In[6]:

import numpy as np
from pytz import timezone
import scipy

trading_freq = 20

def initialize(context):
    
    context.stocks = [ sid(19662),  # XLY Consumer Discrectionary SPDR Fund
                       sid(19656),  # XLF Financial SPDR Fund
                       sid(19658),  # XLK Technology SPDR Fund
                       sid(19655),  # XLE Energy SPDR Fund
                       sid(19661),  # XLV Health Care SPRD Fund
                       sid(19657),  # XLI Industrial SPDR Fund
                       sid(19659),  # XLP Consumer Staples SPDR Fund
                       sid(19654),  # XLB Materials SPDR Fund
                       sid(19660),  # XLU Utilities SPRD Fund
                       sid(33652)]  # BND Vanguard Total Bond Market ETF
    
    context.x0 = 1.0*np.ones_like(context.stocks)/len(context.stocks)

    set_commission(commission.PerShare(cost=0.013, min_trade_cost=1.3))
    
    context.day_count = -1

def handle_data(context, data):
     
    # Trade only once per day
    loc_dt = get_datetime().astimezone(timezone('US/Eastern'))
    if loc_dt.hour == 16 and loc_dt.minute == 0:
        context.day_count += 1
        pass
    else:
        return
    
    # Limit trading frequency
    if context.day_count % trading_freq != 0.0:
        return
    
    prices = history(21,'1d','price').as_matrix(context.stocks)
    ret = np.diff(prices,axis=0) # daily returns
    
    bnds = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1))
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1.0})
    
    res= scipy.optimize.minimize(variance, context.x0, args=ret,method='SLSQP',constraints=cons,bounds=bnds)
    
    allocation = res.x
    allocation[allocation<0]=0
    denom = np.sum(allocation)
    if denom != 0:
        allocation = allocation/denom
        
    context.x0 = allocation
        
    record(stocks=np.sum(allocation[0:-1]))
    record(bonds=allocation[-1])
    
    for i,stock in enumerate(context.stocks):
        order_target_percent(stock,allocation[i])
        
def variance(x,*args):
    
    p = np.asarray(args)
    Acov = np.cov(p.T)
    
    return np.dot(x,np.dot(Acov,x))


# In[ ]:

#http://d.hatena.ne.jp/saitodevel01/20110212/1297487137

###分散共分散行列を計算する
import cvxopt
correl_mat = cvxopt.matrix([[1.0, 0.16, -0.06, -0.05, -0.29],
                            [0.16, 1.0, -0.25, 0.27, 0.73],
                            [-0.06, -0.25, 1.0, 0.56, 0.11],
                            [-0.05, 0.27, 0.56, 1.0, 0.91],
                            [-0.29, 0.73, 0.11, 0.91, 1.0]])
stddev_mat = cvxopt.spdiag([5.40, 22.15, 13.25, 19.59, 26.25])*0.01
sigma_mat = stddev_mat*correl_mat*stddev_mat

###株価や投資信託の基準価額の時系列データから分散共分散行列を計算する
import numpy
data1 = numpy.array([投資対象1の時系列データ], dtype=float)
data2 = numpy.array([投資対象2の時系列データ], dtype=float)

# リターンの計算
return_mat1 = data1[0:-1]/data1[1:]
return_mat2 = data2[0:-1]/data2[1:]

# 分散共分散行列の計算
sigma_mat = numpy.cov(numpy.vstack(return_mat1, return_mat2)) # 単位が年率でないので注意

#### 効率的フロンティア上のポートフォリオを計算する
import math
import cvxopt
import cvxopt.solvers
def calc_portfolio(sigma_mat, r, r0):
  """
  sigma_mat : 分散共分散行列 (cvxopt.base.matrix)
  r  : 各投資対象の期待リターンのベクトル (cvxopt.base.matrix または list)
  r0 : 目標リターン (float)
  """
  n = sigma_mat.size[0]
  minus_r = [-x for x in r]

  P = sigma_mat
  q = cvxopt.matrix(0.0, (n, 1))
  
  G = cvxopt.matrix([cvxopt.matrix([[1.0]*n, minus_r]).T, cvxopt.spdiag([-1.0]*n)])
  h = cvxopt.matrix([1.0, -r0] + [0 for i in xrange(n)])
  
  x = cvxopt.solvers.coneqp(P, q, G, h)['x']

  # (配分割合のリスト、ポートフォリオのリターン, ポートフォリオのリスク(標準偏差))
  return list(x), (r.T*x)[0], math.sqrt((x.T*P*x)[0])

from scipy.stats import norm
def calc_shortfall_risk(r_min, r, sigma):
  return norm.cdf(r_min, r, sigma)

def calc_value_at_risk(probability, r, sigma):
  return norm.ppf(1.0-probability, r, sigma)

def calc_sharp_ratio(r_norisk, r, sigma):
  return (r - r_norisk)/sigma
cvxopt.solvers.options['show_progress'] = False

