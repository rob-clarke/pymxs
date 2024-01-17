import numpy.typing as npt
import unicodedata

def unicode_len(s):
    return sum([1 if unicodedata.combining(c) == 0 else 0 for c in s])

def dot(x):
    return f"{x}\u0307"

class Matrix:
    def __init__(self, content: npt.NDArray):
        self.content = content
    
    def get_height(self):
        return self.content.shape[0] + 2
    
    def get_width(self):
        return self.content.shape[1] * 9 + 3
        
    def get_line(self, linenum, total_lines):
        centre = total_lines // 2
        top = centre - self.get_height() // 2
        if linenum < top:
            return " " * self.get_width()
        linenum = linenum - top
        lines = self.get_height()
        if linenum >= lines:
            return " " * self.get_width()
        if linenum == 0:
            return "\u250c{}\u2510".format(
                " " * (self.get_width()-2)
            )
        if linenum == lines - 1:
            return "\u2514{}\u2518".format(
                " " * (self.get_width()-2)
            )
        format_string = "\u2502 {}\u2502".format(
            "{:8.2f} " * self.content.shape[1]
        )
        return format_string.format(
            *self.content[linenum-1,:]
        )

class SymbolVector:
    def __init__(self, content):
        self.content = content
        self.content_width = max(map(unicode_len, content))
    
    def get_height(self):
        return len(self.content) + 2
    
    def get_width(self):
        return self.content_width + 4
    
    def get_line(self, linenum, total_lines):
        centre = total_lines // 2
        top = centre - self.get_height() // 2
        if linenum < top:
            return " " * self.get_width()
        linenum = linenum - top
        lines = self.get_height()
        if linenum >= lines:
            return " " * self.get_width()
        if linenum == 0:
            return "\u250c{}\u2510".format(
                " " * (self.get_width()-2)
            )
        if linenum == lines - 1:
            return "\u2514{}\u2518".format(
                " " * (self.get_width()-2)
            )
        return f"\u2502 {{:^{self.content_width}}} \u2502".format(
            self.content[linenum-1]
        )

class Symbol:
    def __init__(self, symbol):
        self.symbol = symbol
    
    def get_height(self):
        return 1
    
    def get_width(self):
        return 1
    
    def get_line(self, linenum, total_lines):
        centre = total_lines // 2
        if linenum == centre:
            return self.symbol
        return " "


class Equation:
    def __init__(self, elements):
        self.elements = elements
    
    def __str__(self):
        line_strings = []
        total_lines = max(map(lambda e: e.get_height(), self.elements))
        for line_num in range(total_lines):
            line_string = ""
            for element in self.elements:
                line_string += "{} ".format(element.get_line(line_num,total_lines))
            line_strings.append(line_string)
        return "\n".join(line_strings)


def prettyeqn():
    import numpy
    from prettyeqn import Equation, SymbolVector, Matrix, Symbol
    result = SymbolVector([
        'u\u0307',
        'w\u0307',
        'q\u0307',
        '\u03b8\u0307',
        'z\u0307',
    ])
    a = Matrix(numpy.array([
        [-0.84,   0.71,   0.20,  -9.81, -0.00],
        [-0.93, -17.34,  25.00,   0.08, -0.00],
        [ 0.16,  -2.49, -19.79,  -0.00, -0.00],
        [ 0.00,  -0.00,   1.00,   0.00, -0.00],
        [ 0.01,   1.00,  -0.00, -25.00, -0.00],
    ]))
    state = SymbolVector([
        'u',
        'w',
        'q',
        '\u03b8',
        'z',
    ])
    b = Matrix(numpy.array([
        [  -0.00,  14.68],
        [  -0.00,   0.00],
        [ 385.54, -15.37],
        [  -0.00,   0.00],
        [  -0.00,   0.00],
    ]))
    input = SymbolVector([
        '\u03b7',
        '\u03c4',
    ])
    equation = Equation([result,Symbol('='),a,state,Symbol('+'),b,input])
    
    print(equation)

if __name__ == "__main__":
    print("prettyeqn demo!")
    prettyeqn()
