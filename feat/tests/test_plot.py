import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from .utils import get_test_data_path
from feat.data import Facet, Openface, Affdex
from feat.utils import read_openface, read_affectiva
from feat.plotting import plot_face, draw_lineface, draw_vectorfield, predict
import matplotlib
import pytest
matplotlib.use('TkAgg')

def assert_plot_shape(ax):
    assert ax.get_ylim()==(-240.0, -50.0)
    assert ax.get_xlim()==(25.0, 172.0)

feature_length = 20
au = np.ones(feature_length)
au2 = np.ones(feature_length)*3

def testpredict():
    landmarks = predict(au)
    assert landmarks.shape==(2,68)
    with pytest.raises(ValueError):
        predict(au,model=[0])
    with pytest.raises(ValueError):
        predict(au[:-1])

def test_draw_lineface():
    landmarks = predict(au)
    draw_lineface(currx=landmarks[0,:], curry=landmarks[1,:])
    assert_plot_shape(plt.gca())
    plt.close()

def test_draw_vectorfield():
    draw_vectorfield(reference=predict(au), target=predict(au=au2))
    assert_plot_shape(plt.gca())
    plt.close()
    with pytest.raises(ValueError):
        draw_vectorfield(reference=predict(au).reshape(4,2*feature_length), target=predict(au=au2))
    with pytest.raises(ValueError):
        draw_vectorfield(reference=predict(au), target=predict(au=au2).reshape(4,2*feature_length))

def test_plot_face():
    # test plotting method
    fx = Facet(filename=join(get_test_data_path(), 'iMotions_Test.txt'),sampling_freq=30)
    fx.read_file()
    ax = fx.plot(row_n=0)
    assert_plot_shape(ax)
    plt.close()

    fx = Openface(filename=join(get_test_data_path(), 'OpenFace_Test.csv'),sampling_freq=30)
    fx.read_file()
    ax = fx.plot(row_n=0)
    assert_plot_shape(ax)
    plt.close()

    fx = Affdex(filename=join(get_test_data_path(), 'sample_affectiva-api-app_output.json'),sampling_freq=30)
    fx.read_file(orig_cols=True)
    ax = fx.plot(row_n=0)
    assert_plot_shape(ax)
    plt.close()

    # test plot in util
    plot_face()
    assert_plot_shape(plt.gca())
    plt.close()

    plot_face(au=au, vectorfield={'target':predict(au2)})
    assert_plot_shape(plt.gca())
    plt.close()

    with pytest.raises(ValueError):
        plot_face(model=au, au=au, vectorfield={'target':predict(au2)})
    with pytest.raises(ValueError):
        plot_face(model=au, au=au, vectorfield=[])
    with pytest.raises(ValueError):
        plot_face(model=au, au=au, vectorfield={'notarget':predict(au2)})
