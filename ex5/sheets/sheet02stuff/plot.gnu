set terminal epslatex standalone
set output "nr1.tex"
m_s = 14
m_h = 1
f(x) = m_s / (x - 2*x**2/(x+m_s))
g(x) = m_h / (x - 2*x**2/(x+m_h))
plot f(x)-g(x) title "$x^2$"

