import math as m

def norma2(xi, xj):
    suma = 0
    if len(xi) == len(xj):
        for i in range(len(xi)):
            suma += (xi[i] - xj[i])**2
        return m.sqrt(suma)
    raise ValueError("Las dimensiones de los vectores no coinciden")    

vector1 = [1, 2, 3]
vector2 = [4, 5, 6]
print(norma2(vector1, vector2))