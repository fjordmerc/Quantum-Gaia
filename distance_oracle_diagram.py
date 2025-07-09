import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

# Block positions
x0, y0 = 0.5, 2.5  # xi, yi
x1 = x0 + 1.2       # dx, dy
x2 = x1 + 2.0       # square
x3 = x2 + 2.0       # sum
x4 = x3 + 1.7       # d^2
x5 = x4 + 1.7       # Comp
x6 = x5 + 2.0       # output

y_x = 2.5
y_y = 1.0

# Draw input labels
ax.text(x0-0.3, y_x, r'$x_i$', va='center', ha='right', fontsize=16)
ax.text(x0-0.3, y_y, r'$y_i$', va='center', ha='right', fontsize=16)
ax.text(x0+0.3, y_x-0.5, r'$x_j$', va='center', ha='left', fontsize=16, color='gray')
ax.text(x0+0.3, y_y-0.5, r'$y_j$', va='center', ha='left', fontsize=16, color='gray')

# Draw dx, dy blocks
ax.add_patch(Rectangle((x1-0.5, y_x-0.4), 1.0, 0.8, fill=False, lw=2))
ax.text(x1, y_x, 'dx', ha='center', va='center', fontsize=14)
ax.add_patch(Rectangle((x1-0.5, y_y-0.4), 1.0, 0.8, fill=False, lw=2))
ax.text(x1, y_y, 'dy', ha='center', va='center', fontsize=14)

# Draw square blocks
ax.add_patch(Rectangle((x2-0.7, y_x-0.4), 1.4, 0.8, fill=False, lw=2))
ax.text(x2, y_x, 'square', ha='center', va='center', fontsize=14)
ax.add_patch(Rectangle((x2-0.7, y_y-0.4), 1.4, 0.8, fill=False, lw=2))
ax.text(x2, y_y, 'square', ha='center', va='center', fontsize=14)

# Draw sum block
ax.add_patch(Rectangle((x3-0.7, 1.0-0.6), 1.4, 2.1, fill=False, lw=2))
ax.text(x3, 1.75, 'sum', ha='center', va='center', fontsize=14)

# Draw d^2 block
ax.add_patch(Rectangle((x4-0.6, 1.75-0.5), 1.2, 1.0, fill=False, lw=2))
ax.text(x4, 1.75, r'$d^2$', ha='center', va='center', fontsize=16)

# Draw Comp block
ax.add_patch(Rectangle((x5-0.7, 1.75-0.5), 1.4, 1.0, fill=False, lw=2))
ax.text(x5, 1.75, 'Comp', ha='center', va='center', fontsize=14)

# Draw output label
y_out = 1.75
ax.text(x6+0.2, y_out, r'$g(i, j, \delta^2)$', va='center', ha='left', fontsize=16)

# Draw delta^2 input
ax.text(x5-0.2, 0.7, r'$\delta^2$', ha='right', va='center', fontsize=16)
ax.annotate('', xy=(x5-0.1, 1.25), xytext=(x5-0.1, 0.8), arrowprops=dict(arrowstyle='-|>', lw=2))
ax.annotate('', xy=(x5-0.1, 1.25), xytext=(x5-0.1, 0.8), arrowprops=dict(arrowstyle='-|>', lw=2))
ax.annotate('', xy=(x5-0.1, 1.75), xytext=(x5-0.1, 1.25), arrowprops=dict(arrowstyle='-|>', lw=2))

# Draw arrows for data flow
# xi, xj to dx
ax.annotate('', xy=(x1-0.5, y_x), xytext=(x0, y_x), arrowprops=dict(arrowstyle='-|>', lw=2))
# yi, yj to dy
ax.annotate('', xy=(x1-0.5, y_y), xytext=(x0, y_y), arrowprops=dict(arrowstyle='-|>', lw=2))
# dx to square
ax.annotate('', xy=(x2-0.7, y_x), xytext=(x1+0.5, y_x), arrowprops=dict(arrowstyle='-|>', lw=2))
# dy to square
ax.annotate('', xy=(x2-0.7, y_y), xytext=(x1+0.5, y_y), arrowprops=dict(arrowstyle='-|>', lw=2))
# square x to sum
ax.annotate('', xy=(x3-0.7, y_x), xytext=(x2+0.7, y_x), arrowprops=dict(arrowstyle='-|>', lw=2))
# square y to sum
ax.annotate('', xy=(x3-0.7, y_y), xytext=(x2+0.7, y_y), arrowprops=dict(arrowstyle='-|>', lw=2))
# sum to d^2
ax.annotate('', xy=(x4-0.6, 1.75), xytext=(x3+0.7, 1.75), arrowprops=dict(arrowstyle='-|>', lw=2))
# d^2 to Comp
ax.annotate('', xy=(x5-0.7, 1.75), xytext=(x4+0.6, 1.75), arrowprops=dict(arrowstyle='-|>', lw=2))
# Comp to output
y_out = 1.75
ax.annotate('', xy=(x6, y_out), xytext=(x5+0.7, y_out), arrowprops=dict(arrowstyle='-|>', lw=2))

plt.tight_layout()
plt.savefig('distance_oracle_2d.png', dpi=300)
plt.show() 