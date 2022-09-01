import random

class Matrix():

    def __init__(self,rows,cols):
        self.rows = rows
        self.cols = cols
        self.matrix = [[0 for j in range(cols)] for i in range(rows)]
    
    #Creates matrix object from a 1D list
    @staticmethod
    def fromList(array):
        m = Matrix(len(array),1)
        for i in range(len(array)):
            m.matrix[i][0] = array[i]
        return m

    #Rerturns a list from a matrix object
    @staticmethod
    def toList(m):
        list = []
        for i in range(m.rows):
            for j in range(m.cols):
                list.append(m.matrix[i][j])
        return list

    #Gives each element a random int value between a and b
    def randint(self,a,b):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = random.randint(a,b)

    #Gives each element a random float value between a and b            
    def uniform(self,a,b):
        for i in range(self.rows):
            for j in range(self.cols):
                self.matrix[i][j] = random.uniform(a,b)

    #Returns a transposed matrix object
    @staticmethod
    def transpose(m):
        result = Matrix(m.cols,m.rows)
        for i in range(m.rows):
            for j in range(m.cols):
                result.matrix[j][i] = m.matrix[i][j]
        return result #returns matrix class object
    
    #Adds a number or a matrix to the self one
    def add(self,n):
        if type(n) == float or type(n) == int: #Scalar adition
            for i in range(self.rows):
                for j in range(self.cols):
                    self.matrix[i][j] += n
        elif isinstance(n,Matrix):
            if self.rows == n.rows and self.cols == n.cols: #matrix elementwise addition
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.matrix[i][j] += n.matrix[i][j]
            elif self.rows != n.rows or self.cols != n.cols:
                print("Number of rows or columns unmatched")
    
    #Substracts two matrices and returns result matrix object
    @staticmethod
    def substract(a,b):
        result = Matrix(a.rows,a.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                result.matrix[i][j] = a.matrix[i][j] - b.matrix[i][j]
        return result

    #Miltiplyes a number or a matrix to the self one
    def multiply(self,n):
        if type(n) == float or type(n) == int: #Scalar product
            for i in range(self.rows):
                for j in range(self.cols):
                    self.matrix[i][j] *= n
        elif isinstance(n,Matrix):
            if self.rows == n.rows and self.cols == n.cols: #matrix elementwise multiplication
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.matrix[i][j] *= n.matrix[i][j]
            elif self.rows != n.rows or self.cols != n.cols:
                print("Number of rows or columns unmatched")

    #Divides a number or a matrix to the self one
    def divide(self,n):
        if type(n) == float or type(n) == int: #Scalar division
            for i in range(self.rows):
                for j in range(self.cols):
                    self.matrix[i][j] /= n
        elif isinstance(n,Matrix):
            if self.rows == n.rows and self.cols == n.cols: #matrix elementwise division
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.matrix[i][j] /= n.matrix[i][j]
            elif self.rows != n.rows or self.cols != n.cols:
                print("Number of rows or columns unmatched")

    #Returns matrix object of the dot product of two matrices
    @staticmethod
    def dot_product(a,b): # AxB * BxC matrix dot product
        if a.cols == b.rows:
            result = Matrix(a.rows,b.cols) #result will be a AxC matrix
            for i in range(result.rows):        #Loops thru all elements 
                for j in range(result.cols):
                    result.matrix[i][j] = 0
                    for k in range(a.cols):  #extra loop to make sum              
                        result.matrix[i][j] += a.matrix[i][k]*b.matrix[k][j] 
            return result #does not afect matrix ab, return other result matrix
        elif a.cols != b.rows:
            print("Number of rows or columns unmatched")

    #Returns mapped matrix
    @staticmethod
    def smap(m,fn):
        result = Matrix(m.rows,m.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                element = m.matrix[i][j]
                result.matrix[i][j] = fn(element)
        return result

    #Applies a function to each matrix element
    def map(self,fn):
        for i in range(self.rows):
            for j in range(self.cols):
                element = self.matrix[i][j]
                self.matrix[i][j] = fn(element)

    #Copies matrix element 
    def copy(self):
        result = Matrix(self.rows,self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
               result.matrix[i][j] = self.matrix[i][j] 
        return result