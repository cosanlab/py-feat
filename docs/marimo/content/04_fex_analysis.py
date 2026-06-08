import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. Running a full analysis

    In this tutorial we'll perform a real analysis on part of the open dataset from ["A Data-Driven Characterisation Of Natural Facial Expressions When Giving Good And Bad News"](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008335) by Watson & Johnston 2020.  You can try it out interactively in Google Collab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cosanlab/py-feat/blob/master/notebooks/content/04_fex_analysis.ipynb)

    In the original paper the authors had 3 speakers deliver *good* or *bad* news while filming their facial expressions. They found that could accurately "decode" each condition based on participants' facial expressions extracted either using a custom multi-chanel-gradient model or action units (AUs) extracted using [Open Face](https://github.com/TadasBaltrusaitis/OpenFace).

    In this tutorial we'll show how easily it is to not only reproduce their decoding analysis with py-feat, but just as easily perform additional analyses. Specifically we'll:

    1. Download 20 of the first subject's videos (the full dataset is available on [OSF](https://osf.io/6tbwj/)
    2. Extract facial features using the `Detector`
    3. Aggregate and summarize detections per video using `Fex`
    2. Train and test a decoder to classify *good* vs *bad* news using extracted emotions, AUs, and poses
    3. Run a fMRI style "mass-univariate" comparison across all AUs between conditions
    4. Run a time-series analysis comparing videos based on the time-courses of extracted facial features
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.1 Download the data

    Here's we'll download and save the first 20 video files and their corresponding attributes from OSF. The next cell should run quickly on Google Collab, but will depend on your own internet connection if you're executing this notebook locally. You can rerun this cell in case the download fails for any reason, as it should skip downloading existing files:
    """)
    return


@app.cell
def _():
    import os
    import subprocess
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from glob import glob
    import seaborn as sns
    from tqdm import tqdm
    sns.set_context("talk")

    files_to_download = {
        "4c5mb": 'clip_attrs.csv',
        "n6rt3": '001.mp4',
        "3gh8v": '002.mp4',
        "twqxs": '003.mp4',
        "nc7d9": '004.mp4',
        "nrwcm": '005.mp4',
        "2rk9c": '006.mp4',
        "mxkzq": '007.mp4',
        "c2na7": '008.mp4',
        "wj7zy": '009.mp4',
        "mxywn": '010.mp4',
        "6bn3g": '011.mp4',
        "jkwsp": '012.mp4',
        "54gtv": '013.mp4',
        "c3hpm": '014.mp4',
        "utdqj": '015.mp4',
        "hpw4a": '016.mp4',
        "94swe": '017.mp4',
        "qte5y": '018.mp4',
        "aykvu": '019.mp4',
        "3d5ry": '020.mp4',
    }

    for fid, fname in files_to_download.items():
        if not os.path.exists(fname):
            print(f"Downloading: {fname}")
            subprocess.run(f"wget -O {fname} --content-disposition https://osf.io/{fid}/download".split())

    videos = np.sort(glob("*.mp4"))

    # Load in attributes
    clip_attrs = pd.read_csv("clip_attrs.csv")

    # Add in file names and rename conditions
    clip_attrs = clip_attrs.assign(
        input=clip_attrs.clipN.apply(lambda x: str(x).zfill(3) + ".mp4"),
        condition=clip_attrs["class"].replace({"gn": "goodNews", "ists": "badNews"}),
    )

    # We're only using a subset of videos for this tutorial so drop the rest
    clip_attrs = clip_attrs.query("input in @videos")

    print(f"Downloaded {len(videos)} videos")
    print(f"Downloaded attributes files with {clip_attrs.shape[0]} rows")
    return clip_attrs, np, os, pd, plt, sns, tqdm, videos


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.2 Extract facial features using `Detector`

    Now we'll initialize a new `Detector`, process each frame of each video using `.detect_video()`, and save the results to csv files named after the video.
    """)
    return


@app.cell
def _(os, tqdm, videos):
    from feat import Detector

    # Initialize the default detector
    detector = Detector()

    # Loop over and process each video and save results to csv
    for video in tqdm(videos):
        out_name = video.replace(".mp4", ".csv")
        if not os.path.exists(out_name):

            print(f"Processing: {video}")

            # This is the line that does detection and auto-saving!
            fex = detector.detect(video, data_type="video", save=out_name)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.3. Aggregate detections using a `Fex` dataframe
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then we can use `read_feat` to load each CSV file and concatenate them together:
    """)
    return


@app.cell
def _(pd, videos):
    from feat.utils.io import read_feat
    fex_1 = pd.concat(map(lambda video: read_feat(video.replace('.mp4', '.csv')), videos))
    print(f'Unique videos: {fex_1.inputs.nunique()}')
    print(f'Total processed frames: {fex_1.shape[0]}')
    print(f"Avg frames per video: {fex_1.groupby('input').size().mean()}")
    return (fex_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Our `Fex` dataframe now contains all detections for all frames of each video
    """)
    return


@app.cell
def _(fex_1):
    fex_1.shape
    fex_1.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summarize data with `Fex.sessions`

    `Fex` dataframes have a special attribute called `.sessions` that act as a grouping factor to make it easier to compute summary statistics with any of the `.extract_*` methods. By default `.sessions` is `None`, but you can use the `.update_sessions()` to return **a new Fex dataframe** with `.sessions` set.

    For example, if we update the sessions to be the name of each video, then `.extract_mean()` will group video-frames (rows) by video making it easy to compute a single summary statistic per file:
    """)
    return


@app.cell
def _(fex_1):
    by_video = fex_1.update_sessions(fex_1['input'])
    video_means = by_video.extract_mean()
    # Compute the mean per video
    video_means  # 20 rows for 20 videos
    return by_video, video_means


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then we can grab the AU detections and call standard pandas methods like `.loc` and `.plot`:
    """)
    return


@app.cell
def _(sns, video_means):
    # Grab the aus just for video 1
    video001_aus = video_means.aus.loc['001.mp4']
    # video001_aus = video_means.aus.loc['001.csv'] # if loading pre-computed csv
    _ax = video001_aus.plot(kind='bar', title='Video 001 AU detection')
    # Plot them
    _ax.set(ylabel='Average Probability')
    sns.despine()
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Chaining operations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `.update_sessions()` always returns a **copy** of the Fex object, so that you can **chain** operations together including existing pandas methods like `.plot()`. Here's an example passing a dictionary to `.update_sessions()`, which maps between old and new session names:
    """)
    return


@app.cell
def _(by_video, clip_attrs, plt, sns):
    # Which condition each video belonged to
    video2condition = dict(zip(clip_attrs['input'], clip_attrs['condition']))
    _ax = by_video.update_sessions(video2condition).extract_mean().aus.plot(kind='bar', legend=False, title='Mean AU detection by condition')
    _ax.set(ylabel='Average Probability', title='AU detection by condition', xticklabels=['Good News', 'Bad News'])  # if loading pre-computed csv
    plt.xticks(rotation=0)  # clip_attrs["input"].str.replace(".mp4", ".csv", regex=False),
    # Update sessions to group by condition, compute means (per condition), and make a
    # barplot of the mean AUs for each condition
    sns.despine()
    _ax
    return (video2condition,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also focus in on the AUs associated with happiness:
    """)
    return


@app.cell
def _(by_video, np, plt, sns, video2condition):
    aus = ['AU06', 'AU12', 'AU25']  # from https://py-feat.org/pages/au_reference.html
    summary = by_video.update_sessions(video2condition).extract_summary(mean=True, sem=True, std=False, min=False, max=False)
    # Update the sessions to condition compute summary stats
    bad_means = summary.loc['badNews', [f'mean_{au}' for au in aus]]
    bad_sems = summary.loc['badNews', [f'sem_{au}' for au in aus]]
    good_means = summary.loc['goodNews', [f'mean_{au}' for au in aus]]
    good_sems = summary.loc['goodNews', [f'sem_{au}' for au in aus]]
    # Organize them for plotting
    fig, _ax = plt.subplots(figsize=(3, 4))
    ind = np.arange(len(bad_means))
    width = 0.35
    rects1 = _ax.bar(ind - width / 2, bad_means, width, yerr=bad_sems, label='Bad News')
    rects2 = _ax.bar(ind + width / 2, good_means, width, yerr=good_sems, label='Good News')
    # Plot
    _ax.set(ylabel='Average Probability', title='', xticks=ind, xticklabels=aus, ylim=(0, 1))
    _ax.legend(loc='upper left', frameon=False, bbox_to_anchor=(0, 1.25))
    plt.axhline(0.5, ls='--', color='k')
    sns.despine()
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.4 Comparing the condition difference across AUs using regression

    One way we can compare what AUs in the plot show significant differences is by using the `.regress()` method along with numerical contrast codes. For example we can test the difference in activation of every AU when participants delivered *good* vs *bad* news.

    This is analogous to the "mass-univariate" GLM approach in fMRI research, and allows us to identify what AUs are significantly more active in one condition vs another:
    """)
    return


@app.cell
def _(pd, sns, video2condition, video_means):
    # Save the by_condition fex from above
    by_condition = video_means.update_sessions(video2condition)
    by_condition_codes = by_condition.update_sessions({'goodNews': 1.0, 'badNews': -1})
    # We set numerical contrasts to compare mean good news > mean bad news
    b, se, t, p, df, residuals = by_condition_codes.regress(X='sessions', y='aus', fit_intercept=True)
    p_bonf = p / p.shape[1]
    # Now we perform a regression (t-test) at every AU
    _results = pd.concat([b.round(3).loc[['sessions']].rename(index={'sessions': 'betas'}), se.round(3).loc[['sessions']].rename(index={'sessions': 'ses'}), t.round(3).loc[['sessions']].rename(index={'sessions': 't-stats'}), df.round(3).loc[['sessions']].rename(index={'sessions': 'dof'}), p_bonf.round(3).loc[['sessions']].rename(index={'sessions': 'p-values'})])
    _ax = _results.loc['betas'].plot(kind='bar', yerr=_results.loc['ses'], color=['steelblue' if elem else 'gray' for elem in _results.loc['p-values'] < 0.01], title='Good News > Bad News\n(blue: p < .01)')
    xticks = _ax.get_xticklabels()
    xticks = [elem.get_text().split('_')[-1] for elem in xticks]
    # We can perform bonferroni correction for multiple comparisons:
    _ax.set_xticklabels(xticks)
    _ax.set_ylabel('Beta +/- SE')
    sns.despine()
    _ax
    return (by_condition,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.5 Decoding condition from facial features

    We can easily perform an analysis just like Watson et al, by training a LinearDiscriminantAnalysis (LDA) decoder to classify which condition a video came from based on average **AU** and **headpose** detections.

    To do this we can use the `.predict()` which behaves just like `.regress()` but also requires a `sklearn` `Estimator`. We can use keyword arguments to perform 10-fold cross-validation to test the accuracy of each decoder:
    """)
    return


@app.cell
def _(by_condition, pd, plt, sns):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    feature_list = ['emotions', 'aus', 'poses', 'emotions,poses', 'aus,poses']  # always a good idea to normalize your features!
    _results = []
    models = {}
    # List of different models we'll train
    for features in feature_list:
        model, accuracy = by_condition.predict(X=features, y='sessions', model=make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()), cv_kwargs={'cv': 10})
        models[features] = model
        _results.append(pd.DataFrame({'Accuracy': accuracy * 100, 'Features': [features] * len(accuracy)}))
        print(f'{features} model accuracy: {accuracy.mean() * 100:.3g}% +/- {accuracy.std() * 100:.3g}%')
    _results = pd.concat(_results).assign(Features=lambda df: df.Features.map({'emotions': 'Emotions', 'poses': 'Pose', 'aus': 'AUs', 'emotions,poses': 'Emotions\n+ Pose', 'aus,poses': 'AUs+Pose'}))
    f, _ax = plt.subplots(1, 1, figsize=(3.75, 4))  # .predict is just like .regress, but this time session is our y.
    _ax = sns.barplot(x='Features', y='Accuracy', errorbar='sd', dodge=False, hue='Features', data=_results, ax=_ax, order=['Emotions', 'Emotions\n+ Pose', 'AUs+Pose', 'AUs', 'Pose'])
    _ax.get_legend().remove()
    _ax.set_title('Good News vs Bad News\nClassifier Performance')
    _ax.set(ylabel='Accuracy', xlabel='')
    sns.despine()
    plt.axhline(y=50, ls='--', color='k')
    plt.xticks(rotation=90)
    plt.tight_layout()  # Save the model
    # Concat results into a single dataframe and tweak column names
    # Plot it
    # with sns.plotting_context("talk", font_scale=1.8):
    return (models,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualizing decoder weights
    Using what we learned in the previous tutorial, we can visualize the coefficients for any models that used AU features. This allows us to "see" the underlying facial expression that the classifier learned!
    """)
    return


@app.cell
def _(models, plt, sns):
    from feat.plotting import plot_face

    _ax = plot_face(
        au=models['aus'][1].coef_.squeeze(), # the LDA coefs from the AUs pipeline model
        feature_range=(0, 1),
        muscles={"all": "heatmap"},
        title="Expression reconstructed from\nAU classifier weights",
        title_kwargs={'wrap':False}
    );
    sns.despine(left=True, bottom=True)
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Even cooler we can *animate* that face expression to **emphasize what's changing.** Here we start from a neutral face:
    """)
    return


@app.cell
def _(mo, models):
    from feat.plotting import animate_face

    animation = animate_face(
        end=models['aus'][1].coef_.squeeze(), # same as before
        feature_range=(0, 1),
        muscles={'all': 'heatmap'},
        title="Good vs Bad News Classifier Weights",
        save="weights.gif",
    )
    mo.image("weights.gif")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.6 Time-series analysis

    Finally we might be interested in looking the similarity of the detected features **over time**. We can do that using the `.isc()` method which takes a column and metric to use. Here we compare detected happiness between all pairs of videos.

    We use some helper functions to cluster, sort, and plot the correlation matrix. Warmer colors indicate a pair of videos elicited more *similar* detected Happiness over time. We see that some videos show high-correlation in-terms of their detected happiness over-time. This is likely why the classifier above was able to decode conditions so well.
    """)
    return


@app.cell
def _(fex_1):
    # ISC returns a video x video pearson correlation matrix
    isc = fex_1.isc(col='happiness', method='pearson')
    return (isc,)


@app.cell
def _(isc, np, plt, sns, video2condition):
    def cluster_corrs(df):
        """Helper to reorder rows and cols of correlation matrix based on clustering"""
        import scipy.cluster.hierarchy as sch
        pairwise_distances = sch.distance.pdist(df)
        linkage = sch.linkage(pairwise_distances, method='complete')
        cluster_distance_threshold = pairwise_distances.max() / 2
        idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion='distance')
        idx = np.argsort(idx_to_cluster_array)
        return df.iloc[idx, :].T.iloc[idx, :]

    def add_cond_to_ticks(ax):
        """Helper to add condition info to each tick label"""
        xlabels, ylabels = ([], [])
        for xlabel, ylabel in zip(_ax.get_xticklabels(), _ax.get_yticklabels()):
            x_condition = video2condition[xlabel.get_text()]
            y_condition = video2condition[ylabel.get_text()]
            x_new = f"{x_condition[:-4]}_{xlabel.get_text().split('.csv')[0][1:]}"
            y_new = f"{y_condition[:-4]}_{ylabel.get_text().split('.csv')[0][1:]}"
            xlabels.append(x_new)
            ylabels.append(y_new)
        _ax.set_xticklabels(xlabels)
        _ax.set_yticklabels(ylabels)
        return _ax
    _ax = sns.heatmap(cluster_corrs(isc), cmap='RdBu_r', vmin=-1, vmax=1, square=True)
    _ax = add_cond_to_ticks(_ax)
    _ax.set(xlabel='', ylabel='', title='Inter-video Happiness\ntimeseries correlation')
    # Plot it
    return


if __name__ == "__main__":
    app.run()
