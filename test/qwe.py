def ten_to_two(n):
    string = ""
    while n > 0:
        string = str(n % 2) + string
        n = n // 2
    return string

print(bin(125))
print(int("101010",2))
def two_to_ten(r):
    sum = 0
    for idx, elem in enumerate(r):
        sum = sum + int(elem)*(2 ** (len(r) - idx))
    return sum


try:
    print(123)
except:
    print("qwe")


min = 1000000000
for idx in range(0, 1000):
    binary_number = ten_to_two(idx)
    summary_of_numbers = 0
    for let in binary_number:
        summary_of_numbers += int(let)
    if summary_of_numbers % 2 == 0:
        binary_number = binary_number + "0"
        binary_number = "10" + binary_number[3:]
    else:
        binary_number = binary_number + "1"
        binary_number = "11" + binary_number[3:]
    number_r = two_to_ten(binary_number)
    if number_r > 40 and idx < min:
        min = idx

print(min)