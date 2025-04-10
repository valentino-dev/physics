from manim import *
from functools import partial

class harmonics(MovingCameraScene):
    def construct(self):

        axes = Axes(x_range=[0, 1], y_range=[-3, 3])
        self.add(axes)

        tracker = ValueTracker(0)

        def get_wave(m,k,A,c):
            t = tracker.get_value()/5
            return axes.plot(lambda x: 2*A*np.sin(PI*m*t)*np.sin(PI*k*x), color=c)

        c=[BLUE,RED,GREEN]
        for i in range(1, 4):
            wave = always_redraw(partial(get_wave, 1, i, 1, c[i-1]))
            self.add(wave)

        self.play(tracker.animate.set_value(4 * PI), run_time=5, rate_func=linear)
        self.wait()

class stehwelle(MovingCameraScene):
    def construct(self):

        axes = Axes(x_range=[0, 1], y_range=[-3, 3])
        self.add(axes)

        tracker = ValueTracker(0)

        def get_wave1(m,k,A,c):
            t = tracker.get_value()/5
            return axes.plot(lambda x: 2*A*np.sin(PI*m*t)*np.sin(PI*k*x), color=c)
        def get_wave2(m,k,A,c):
            t = tracker.get_value()/5
            return axes.plot(lambda x: A*np.cos(PI*m*t-PI*k*x), color=c)
        def get_wave3(m,k,A,c):
            t = tracker.get_value()/5
            return axes.plot(lambda x: A*np.cos(PI*m*t+PI*k*x+PI), color=c)

        k=2
        m=2
        f=[get_wave1, get_wave2, get_wave3]
        c=[BLUE,RED,GREEN]
        for i in range(3):
            wave = always_redraw(partial(f[i], m, k, 1, c[i]))
            self.add(wave)

        self.play(tracker.animate.set_value(4 * PI), run_time=10, rate_func=linear)
        self.wait()
