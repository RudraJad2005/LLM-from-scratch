
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def add(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def subtract(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    def scalar_multiply(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    def dot_product(self, other):
        return self.x * other.x + self.y * other.y
vector1 = Vector(1, 2)
vector2 = Vector(3, 4)
result_add = vector1.add(vector2)
print(f"Vector addition result: ({result_add.x}, {result_add.y})")


class Matrix:
    def __init__(self, data):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0

    def transpose(self):
        transposed_data = [[self.data[i][j] for i in range(self.rows)] for j in range(self.cols)]
        return Matrix(transposed_data)
    
    def multiply(self, other):
        if self.cols != other.rows:
            raise ValueError("Incompatible dimensions for matrix multiplication.")
        
        result_data = [[0 for _ in range(other.cols)] for _ in range(self.rows)]
        
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result_data[i][j] += self.data[i][k] * other.data[k][j]
        
        return Matrix(result_data)

matrix = Matrix([[1, 2, 3], [4, 5, 6]])
transposed_matrix = matrix.transpose()
print("Original Matrix:")
for row in matrix.data:
    print(row)
print("Transposed Matrix:")
for row in transposed_matrix.data:
    print(row)

