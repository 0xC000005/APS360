# Test the Matplotlib library

import matplotlib.pyplot as plt

# Create a list of numbers from 0 to 10
x = list(range(11))

# Create a list of squares of the numbers
y = [i**2 for i in x]

# Plot the data
plt.plot(x, y)

# Add labels and title
plt.xlabel('x')

plt.ylabel('y')

plt.title('y = x^2')

# Show the plot
plt.show()

if __name__ == '__main__':
    pass
