pixels = [0, 128, 255, 64, 32]   # example brightness values 0–255

with open("image1.bin", "wb") as f:
    f.write(bytes(pixels))