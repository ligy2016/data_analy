#coding=utf-8
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from numpy.random import  randn
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter


def test():
    fig = plt.figure()
    # ax1 = fig.add_subplot(2,2,1)
    # ax2 = fig.add_subplot(2,2,2)
    # ax3 = fig.add_subplot(2,2,3)

    cftc_aud = [24218, 16748, 12056, 10294, 4845, -3848, -3256, -1510, 3941, 13473, 20871, 20957, 30705, 41518, 41108,
                40867, 31923, 29980, 26104, 23938, 15008, 6848, 36467, 38959, 42566, 42757, 41113, 34884, 31363, 31510,
                33431, 16216, 4903,  -1952, -7043, -6778,-15808,-4758,124,24893,38158,52395,59540,44106,35122,26845,23466,
                18030,12782,29195,16861,9575,2807,-5626,-26168,-32798,-36267,-23043,-13761,-17545,-20854,-10452,-33579,-46648,-57145,-66464,-52830,-38625,-36352,-38412,-33705,-40839]


    cftc_eur =  [    -46764,    - 44951,    - 45713,    - 52348,    - 66500,    - 65823,    - 70056,
                - 69408,    - 78045,    - 87513,    - 114556,        - 119240,    - 119348,    - 119182,    - 129314,   - 137385,    - 123856,    - 109268,    - 93472,        - 82059,    - 76030,    - 85025,    - 81475,
                - 92630,    - 81925,    - 76658,    - 92508,        - 98399,    - 104103,    - 112600,    - 99891,
                - 87660,    - 75327,    - 61934,    - 61346,    - 56489,  -67112,      - 37654,    - 37895,     - 22587,  - 21872,   - 23619,    - 39667,- 46917,     - 52051,- 53487,
                 - 63811,     - 66053,      - 77555,   - 71907,   - 68541,     - 46857,      - 48205,    - 63314,     - 87073,  - 127215,      - 137015,    - 146451,     - 160643,
                 - 160550,      - 161047,    - 159961,     - 172331,   - 182845,      - 175484,    - 164177,     - 142939, - 134334,  - 105934,      - 62566,    - 80576,     - 88810        ]

    cftc_usd=[45861,46733,46572,48422,49122,53128,54271,52644,52367,56712,54602,53403,53189,48428,52296,
                54330,53990,45422,33549,12100,12205,14400,17074,16961,14966,15968,16471,15769,
                17717,22479,14526,13984,12279,12179,9372,4692,              9799,              9906,              11178,              10446,              12117,              9174,              11330,              13449,              13849,
              17315,              17750,              17674,              17511,              26378,              26785,              29425,              31342,              35013,              44872,              44225,
              42775,              42719,              40709,              33615,              33407,              36902,              51028,              47337,              46146,              46527,              43784,              39926,              34691,              28203,              38875,              45991
              ]

    date = pd.date_range('2015.10.6',periods=72,freq='W-TUE')
    # print (index)
    cftc_eur.reverse()
    cftc_aud.reverse()
    cftc_usd.reverse()

    ts_eur = pd.Series(cftc_eur,index=date)
    ts_aud = pd.Series(cftc_aud, index=date)
    ts_usd = pd.Series(cftc_usd, index=date)
    plt.plot(ts_eur,'g')
    plt.plot(ts_aud,'r')
    plt.plot(ts_usd, 'y')


    # ax1.hist(randn(100),bins = 20,color='k',alpha=0.3)
    plt.title(' cftc')
    plt.fmt_xdata = DateFormatter('%Y-%m-%d')
    plt.show()

def show_cftc():
    obj = pd.Series[cftc_aud]
def multipleFigure():
    x1= [1,2,3,4,5]
    y1= [1,4,9,16,25]
    x2=[1,2,4,6,8]
    y2=[2,4,8,12,16]
    plt.plot(x1,y1,'r')
    plt.plot(x2, y2, 'g')
    plt.title('plot of y vs x')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.xlim(0.0,9.0)
    plt.ylim(0.0, 30.0)
    plt.show()

def test1():
    import matplotlib.pyplot as plt
    import numpy as np

    import matplotlib.cbook as cbook

    fname = cbook.get_sample_data('msft.csv', asfileobj=False)
    fname2 = cbook.get_sample_data('data_x_x2_x3.csv', asfileobj=False)

    # test 1; use ints
    plt.plotfile(fname, (0, 5, 6))

    # test 2; use names
    plt.plotfile(fname, ('date', 'volume', 'adj_close'))

    # test 3; use semilogy for volume
    plt.plotfile(fname, ('date', 'volume', 'adj_close'),
                 plotfuncs={'volume': 'semilogy'})

    # test 4; use semilogy for volume
    plt.plotfile(fname, (0, 5, 6), plotfuncs={5: 'semilogy'})

    # test 5; single subplot
    plt.plotfile(fname, ('date', 'open', 'high', 'low', 'close'), subplots=False)

    # test 6; labeling, if no names in csv-file
    plt.plotfile(fname2, cols=(0, 1, 2), delimiter=' ',
                 names=['$x$', '$f(x)=x^2$', '$f(x)=x^3$'])

    # test 7; more than one file per figure--illustrated here with a single file
    plt.plotfile(fname2, cols=(0, 1), delimiter=' ')
    plt.plotfile(fname2, cols=(0, 2), newfig=False,
                 delimiter=' ')  # use current figure
    plt.xlabel(r'$x$')
    plt.ylabel(r'$f(x) = x^2, x^3$')

    # test 8; use bar for volume
    plt.plotfile(fname, (0, 5, 6), plotfuncs={5: 'bar'})

    plt.show()
def pandas_draw():
    s= pd.Series(np.random.randn(10).cumsum(),index=np.arange(0,100,10))
    s.plot()
    plt.show()
def drawExchange():
    csv_reader = csv.reader(open('E:\EURUSD1440.csv', encoding='utf-8'))
    # df = pd.DataFrame({'date':0,'time':1,'open':2,'high':3,'low':4,'close':5,'volume':6},columns=['date', 'time', 'open', 'high', 'low', 'close', 'volume'],index=[0])
    df = pd.DataFrame(columns=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
    i = 0
    for row in csv_reader:
        df2= pd.DataFrame([row],columns=df.columns)
        # print("df2:\n",df2)
        df=df.append(df2, ignore_index=True)
        # df.append({'date':row[0],'time':row[1],'open':row[2],'high':row[3],'low':row[4],'close':row[5],'volume':row[6]},ignore_index=True)
        # print (df)
        i+=1
        if i>5:
            break
    print (df)
    # df.plot()


    # s=pd.Series([1,2,3,4,5,6,7], index=['date', 'time', 'open', 'high', 'low', 'close', 'volume'])
    # print(s)
def test_append():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
    df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
    df3=df.append(df2, ignore_index=True)
    print(df3)

if __name__ == '__main__':
    drawExchange()