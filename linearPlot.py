plt.figure(figsize=(4, 3))
ax = plt.axes()
ax.scatter(X, Y)
ax.plot(X_test_year, y_pred)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axis('tight')
plt.show()


# Preparing the required data
# independent = X_train['Current Version Release Year', 'Original Release Year'].values.reshape(-1, 4)
# dependent = Y_train['Average User Rating']

# Creating a variable for every dimension
a = X_train['Current Version Release Year']
b = X_train['Original Release Year']
c = Y_train

a_range = np.linspace(a.min(), a.max(), 100)
b_range = np.linspace(b.min(), b.max(), 100)
a1_range = np.linspace(c.min(), c.max(), 100)
a_range, b_range, a1_range = np.meshgrid(a_range, b_range, a1_range)

# # Ploting the model for visualization
# plt.style.use('fivethirtyeight')

X_test_plot = np.array([a_range.flatten(), b_range.flatten(), a1_range.flatten()]).T
y_pred = regr.predict(X_test_plot)

# Initializing a matplotlib figure
fig = plt.figure(figsize=(15, 6))

axis1 = fig.add_subplot(131, projection='3d')
axis2 = fig.add_subplot(132, projection='3d')
axis3 = fig.add_subplot(133, projection='3d')

axis = [axis1, axis2, axis3]
# y_pred = np.array([a_range.flatten(), b_range.flatten(), a1_range.flatten()]).T


for ax in axis:
    ax.plot(a, b, c, color='k', zorder=10, linestyle='none', marker='o', alpha=0.1)
    ax.scatter(a_range.flatten(), b_range.flatten(), y_pred, facecolor=(0, 0, 0, 0), s=20, edgecolor='#70b3f0')
    ax.set_xlabel('Area', fontsize=10, labelpad=10)
    ax.set_ylabel('Bedrooms', fontsize=10, labelpad=10)
    ax.set_zlabel('Prices', fontsize=10, labelpad=10)
    ax.locator_params(nbins=3, axis='x')
    ax.locator_params(nbins=3, axis='x')

axis1.view_init(elev=25, azim=-60)
axis2.view_init(elev=15, azim=15)
axis3.view_init(elev=25, azim=60)

print('-Multiple Regression:')
# print('Mean Square Error', metrics.mean_squared_error(np.asarray(Y_test), y_pred))
# print('Accuracy', "%.4f" % (metrics.r2_score(Y_test, y_pred)), '\n')

fig.suptitle(f'Multi-Linear Regression Model Visualization (R2 = {(metrics.r2_score(a, b))}, ("fontsize"))')
