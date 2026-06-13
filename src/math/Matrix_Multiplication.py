matrix1 = [
    [1,2,3], 
    [4,5,6]
]
matrix2 = [
    [7,8], 
    [9,10], 
    [11, 12]
]

result = [[0 for _ in range(len(matrix1[0]))] for _ in range(len(matrix2))]

for i in range(len(matrix1)):
    for j in range(len(matrix2[0])):
        for k in range(len(matrix2)):

            result[i][j] += matrix1[i][k] * matrix2[k][j]

for row in result:
    print(row)


## Transpose of a matrix

matrix1 = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

result = [[0 for _ in range(len(matrix1))] for _ in range(len(matrix1[0]))]

for i in range(len(matrix1)):
    for j in range(len(matrix1[0])):
        result[j][i] = matrix1[i][j]

for row in result:
    print(row)