// Typeset labels for the example questions ($…$ is rendered with KaTeX); the plain text is what gets solved.
export const EXAMPLE_LABELS: Record<string, string> = {
  'Solve x^2 - 5x + 6 = 0': 'Solve $x^2 - 5x + 6 = 0$',
  'x + 2y = 7, 3x - y = 7': 'Solve $x + 2y = 7,\\ 3x - y = 7$',
  'factor x^3 - 8': 'Factor $x^3 - 8$',
  'Find the derivative of x^3 sin(x)': 'Derivative of $x^3 \\sin x$',
  'limit of sin(x)/x as x -> 0': '$\\lim_{x \\to 0} \\frac{\\sin x}{x}$',
  'second derivative of exp(x^2)': 'Second derivative of $e^{x^2}$',
  'Integrate x^2 from 0 to 3': '$\\int_0^3 x^2\\,dx$',
  'integral of x e^x dx': '$\\int x e^x\\,dx$',
  'integrate exp(-x^2) from -oo to oo': '$\\int_{-\\infty}^{\\infty} e^{-x^2}\\,dx$',
  'eigenvalues of [[2,1],[1,2]]': 'Eigenvalues of $\\left(\\begin{smallmatrix}2&1\\\\1&2\\end{smallmatrix}\\right)$',
  'inverse of [[4,7],[2,6]]': 'Inverse of $\\left(\\begin{smallmatrix}4&7\\\\2&6\\end{smallmatrix}\\right)$',
  'solve [[2,1],[1,3]] x = [3,5]':
    'Solve $\\left(\\begin{smallmatrix}2&1\\\\1&3\\end{smallmatrix}\\right)\\mathbf{x} = \\left(\\begin{smallmatrix}3\\\\5\\end{smallmatrix}\\right)$',
  'mean and standard deviation of 2, 4, 4, 4, 5, 5, 7, 9': 'Mean and standard deviation of $2, 4, 4, 4, 5, 5, 7, 9$',
  'Binomial n=10 p=0.5, P(X <= 3)': '$X \\sim \\mathrm{Bin}(10,\\ 0.5)$: find $P(X \\le 3)$',
  'normal mean 100 std 15, P(X > 130)': '$X \\sim N(100,\\ 15^2)$: find $P(X > 130)$',
  'sum of the first 100 positive integers': '$1 + 2 + \\dots + 100$',
  '10th term of the arithmetic sequence 3, 7, 11, ...': '10th term of $3, 7, 11, \\dots$',
  'critical points of f(x) = x^3 - 3x': 'Critical points of $f(x) = x^3 - 3x$',
}
