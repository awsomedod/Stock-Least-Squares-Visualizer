import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

stock = input('What stock would you like to predict? \n')
days_in_past = input('How much in the past would you like to look at?\n (Example: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)\n')
days_in_future = int(input("How many days into the future would you like to predict?\n"))
degree_polynomial = int(input("With what degree polynomial would you like to predict with?\n"))


# get stock data
stk = yf.Ticker(stock)

# get historical market data   gets closing price
hist = stk.history(period=days_in_past)["Close"]

#reverse order of stock data
lines = []
for price in hist:
    lines.append(price)
lines = lines[::-1]

#arranges stock data into a matrix of prices and index
nmb_of_data_pts = len(lines)
stk_data = np.zeros([nmb_of_data_pts, 2])
for i in range(nmb_of_data_pts):
    stk_data[i,1] = float(lines[nmb_of_data_pts -1 -i])
    stk_data [i, 0] = i + 1

# creates a numbered list of the total amount of prediction points
d = stk_data[:,0]
predict_vec = [i for i in range(nmb_of_data_pts + 1, nmb_of_data_pts + days_in_future)]
d = [int(i) for i in d]
total_prediction = d + predict_vec

#creates the least squares algorithm
def least_squares(A,b):
    return np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)

def predict_with_degree(deg):
    a1 = (stk_data[:,0]).reshape([nmb_of_data_pts, 1])
    a2 = np.ones([nmb_of_data_pts, 1])
    tup = [a1, a2]


    on = a1
    for i in range(deg-1):
        on = a1 * on
        tup.insert(0, on)

    A = np.hstack(tuple(tup))
    b = (stk_data[:,1]).reshape([nmb_of_data_pts, 1])
    x = least_squares(A, b)

    have = x[0]
    for i in range(1, deg+1):
        have = have * total_prediction
        have = have + x[i]

    return have

while True:
    #creates new prediction equation
    y_new = predict_with_degree(degree_polynomial)


    #plots the matrix
    plt.plot(stk_data[:,0], stk_data[:,1], "g^")
    plt.plot(total_prediction, y_new, "r", linewidth=2.0)
    plt.show()

    degree_polynomial = int(input("With what degree polynomial would you like to predict with?\n"))
