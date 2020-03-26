# kevylees
# Kevin Lee
# CSE163 AF
# This program will use Kobe's basketball statistics
# to gauge his effectiveness on the Los Angeles Lakers.

import os

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error

path = 'kobe-games/'
sns.set()


def combine_datas(entries):
    """
    Helps combine all the csv files.
    """
    df = pd.DataFrame()
    all_dfs = list()
    for entry in entries:
        all_dfs.append(pd.read_csv(path + entry))
    df = pd.concat(all_dfs, sort=False)
    return df


def fit_and_predict_kobe(data):
    """
    Returns the mean squared error value to
    help compare Kobe Bryant's p
    """
    X = data.loc[:, data.columns != 'W/L']
    y = data['W/L'].str[0]
    y = pd.get_dummies(y)
    model = DecisionTreeRegressor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_test_pred)


def plot_kobe_pts(data):
    """
    Plots a scattered-bar plot based on Kobe's point per game
    and the teams wins/loses.
    """
    sns.catplot(x='W/L', y='PTS', data=data)
    plt.title('Kobe\'s Points')
    plt.savefig('points.png', bbox_inches='tight')


def plot_kobe_ast(data):
    """
    Plots a scattered-bar plot based on Kobe's assists per game
    and the teams wins/loses.
    """
    sns.catplot(x='W/L', y='AST', data=data)
    plt.title('Kobe\'s Assists')
    plt.savefig('assists.png', bbox_inches='tight')


def plot_kobe_fg(data):
    """
    Plots a scattered-bar plot based on Kobe's field
    goal percentages per game and the teams wins/loses.
    """
    sns.catplot(x='W/L', y='FG%', data=data)
    plt.title('Kobe\'s Field Goal Percentages')
    plt.savefig('fieldgoal.png', bbox_inches='tight')


def plot_kobe_attendance(data):
    """
    Plots a bar plot based on Kobe's attendance, and
    compares the number of wins and loses.
    """
    winners = data[data['W/L'] == 'W']
    losers = data[data['W/L'] == 'L']
    w_total = len(winners)
    l_total = len(losers)
    df = pd.DataFrame(columns=['W/L', 'Total'])
    df = df.append({'W/L': 'W', 'Total': w_total}, ignore_index=True)
    df = df.append({'W/L': 'L', 'Total': l_total}, ignore_index=True)
    sns.catplot(x='W/L', y='Total', kind='bar', data=df)
    if len(data) < 200:
        plt.title('Kobe\'s Absence')
        plt.savefig('absence.png', bbox_inches='tight')
    else:
        plt.title('Kobe\'s Presence')
        plt.savefig('presence.png', bbox_inches='tight')
    

def main():
    all_files = os.listdir(path)
    df = combine_datas(all_files)
    df['W/L'] = df['W/L'].str[0]
    filtered = df.loc[:, ['PTS', 'AST', 'FG%', 'W/L']]
    filtered = filtered.dropna()
    fit_and_predict_kobe(filtered)
    plot_kobe_pts(filtered)
    plot_kobe_ast(filtered)
    plot_kobe_fg(filtered)
    absent_data = df[df['PTS'].isnull()]
    present_data = df[df['PTS'].notnull()]
    plot_kobe_attendance(absent_data)
    plot_kobe_attendance(present_data)


if __name__ == "__main__":
    main()