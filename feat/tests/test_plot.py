import matplotlib.pyplot as plt
import numpy as np
from feat.plotting import plot_face, draw_lineface, draw_vectorfield, predict
import matplotlib
matplotlib.use('TkAgg')

def assert_plot_shape(ax):
    assert ax.get_ylim()==(-240.0, -50.0)
    assert ax.get_xlim()==(25.0, 172.0)

au = np.ones(17)
au2 = np.ones(17)*3

def testpredict():
    landmarks = predict(au)
    assert landmarks.shape==(2,68)

def test_draw_lineface():
    landmarks = predict(au)
    draw_lineface(currx=landmarks[0,:], curry=landmarks[1,:])
    assert_plot_shape(plt.gca())
    plt.close()

def test_draw_vectorfield():
    draw_vectorfield(reference=predict(au), target=predict(au=au2))
    assert_plot_shape(plt.gca())
    plt.close()

def test_plot_face():
    plot_face()
    assert_plot_shape(plt.gca())
    plt.close()

    plot_face(au=au, vectorfield={'target':predict(au2)})
    assert_plot_shape(plt.gca())
    plt.close()
