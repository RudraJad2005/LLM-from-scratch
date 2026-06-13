
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