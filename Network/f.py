import badcrossbar

applied_voltages = [
    [1],
    [1],
    [1],
]

# Device resistances in ohms.
resistances = [
    [345, 903, 755, 257, 646],
    [652, 401, 508, 166, 454],
    [442, 874, 190, 244, 635],
]

# Interconnect resistance in ohms.
r_i = 0.5

# Computing the solution.
solution = badcrossbar.compute(applied_voltages, resistances, r_i)

# Printing the current through the word line segment to the left of device in
# the second row and fourth column (Python uses zero-based indexing).
current = solution.currents.word_line[1, 3]
print(f"\nThe current through the word line segment is ~{current:.3g} A.")