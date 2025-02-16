


a_v=15.67
a_s=17.23
a_c=0.714
a_a=93.15
E_B=28.3
f(x, y)=-4*a_v+8/3*a_s*x**(-1/3)+4*a_c*y*(1-y/3/x)*x**(-1/3)-a_a*(x-2*y)**2/x**2+E_B

#Z=47
#print f(107)

#Z=79
#print f(197)

#Z=92
#print f(238)


set term cairolatex standalone header '\usepackage{siunitx}'
set grid
set title '$\alpha$-Instabilität'
set output '1_2_q.tex'
set ylabel '$Q_\alpha (A, Z_{\text{min}}(A)) / \SI{}{\mega eV}$'
set xlabel '$A$'
plot '1_2_q.dat' using 1:2 w l t '',\
  '1_2_q.dat' using 1:2 pt 0 title ''

