set term cairolatex standalone

set output 'a1_2.tex'
set xlabel 'Zeit t/h'
set ylabel 'Zählrate $\dot{N}/\frac{1}{\SI{}{s}}$'

f(x)=-log(2)/T*N*exp(-log(2)*x/T)
set fit errorvariables
fit f(x) 'a1_2.dat' using ($0+1):1 via T,N

set title sprintf('Zählrate von einer Probe: $T=\SI{%.3g+-%.2g}{}, N=\SI{%.3g+-%.2g}{}$', T, T_err, N, N_err)

plot 'a1_2.dat' using ($0+1):1 title 'Probe',\
    f(x) title 'Anpassung'


