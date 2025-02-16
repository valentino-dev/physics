set term cairolatex standalone

set output '3_1.tex'
set key top center
set title 'Aufg. 3, Teilaufg. 1: Isobarenheihe mit $A=52$'
set ylabel '$M(A, Z)/$GeV'
set xlabel '$Z$'
set grid
plot '3_1.dat' using 2:1 pt 0 title 'Masse'

set output '3_3.tex'
set key top left
set title 'Aufg. 3, Teilaufg. 3: N-Z-Diagramm'
set xlabel '$N$'
set ylabel '$Z(E=E_{min})$'
set grid
plot '3_3.dat' using 2:1 pt 0 title 'Z für minimale Energie',\
      x title 'N-Z Gerade'





