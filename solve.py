import sympy as sp

def solve_equation(symbols):

    equation_str = "".join(symbols)


    if '=' in equation_str:
        left_side, right_side = equation_str.split('=')
    else:

        left_side, right_side = equation_str, '0'


    left_expr = sp.sympify(left_side)
    right_expr = sp.sympify(right_side)


    equation = sp.Eq(left_expr, right_expr)


    x = sp.symbols('x')

    variables = equation.free_symbols
    if variables:
        solutions = sp.solve(equation, list(variables))
        return solutions
    else:
        return sp.simplify(left_expr - right_expr) == 0
