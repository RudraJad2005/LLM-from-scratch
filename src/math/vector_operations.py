

print("Vector addition result:")
vector1 = [1, 2, 3]

vector2 = [4, 5, 6]

result = [0 for _ in range(len(vector1))]

for i in range(len(vector1)):

    result[i] = vector1[i] + vector2[i]
    print(result)

print("Vector Scalar Multiplication")

vector1 = [1, 2, 3]

result = [0 for _ in range(len(vector1))]

for i in range(len(vector1)):

    result[i] = vector1[i] * 2
    print(result)


print("Vector Subtraction")



vector1 = [2, 3, 4]

vector2 = [5, 6, 7]

result = [0 for _ in range(len(vector1))]

for i in range(len(vector1)):

    result[i] = vector1[i] - vector2[i]
    print(result)