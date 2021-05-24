"""Classes for report generation and add-ons."""

import os
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import FileSystemLoader, Environment
from json2html import json2html
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, precision_recall_curve, \
    average_precision_score, explained_variance_score, mean_absolute_error, \
    mean_squared_error, median_absolute_error, r2_score, f1_score, precision_score, recall_score, confusion_matrix

from ..utils.logging import get_logger

logger = get_logger(__name__)

base_dir = os.path.dirname(__file__)


def extract_params(input_struct):
    params = dict()
    iterator = input_struct if isinstance(input_struct, dict) else input_struct.__dict__
    for key in iterator:
        if key.startswith(('_', 'autonlp_params')):
            continue
        value = iterator[key]
        if type(value) in [bool, int, float, str]:
            params[key] = value
        elif value is None:
            params[key] = None
        elif hasattr(value, '__dict__') or isinstance(value, dict):
            params[key] = extract_params(value)
        else:
            params[key] = str(type(value))
    return params

def plot_roc_curve_image(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(10, 10));

    fpr, tpr, _ = roc_curve(data['y_true'], data['y_pred'])
    auc_score = roc_auc_score(data['y_true'], data['y_pred'])

    lw = 2
    plt.plot(fpr, tpr, color='blue', lw=lw, label='Trained model');
    plt.plot([0, 1], [0, 1], color='red', lw=lw, linestyle='--', label='Random model');
    plt.xlim([-0.05, 1.05]);
    plt.ylim([-0.05, 1.05]);
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    lgd = plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2);
    plt.xticks(np.arange(0, 1.01, 0.05), rotation=45);
    plt.yticks(np.arange(0, 1.01, 0.05));
    plt.grid(color='gray', linestyle='-', linewidth=1);
    plt.title('ROC curve (GINI = {:.3f})'.format(2 * auc_score - 1));
    plt.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight');
    plt.close()
    return auc_score


def plot_pr_curve_image(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(10, 10));

    precision, recall, _ = precision_recall_curve(data['y_true'], data['y_pred'])
    ap_score = average_precision_score(data['y_true'], data['y_pred'])

    lw = 2
    plt.plot(recall, precision, color='blue', lw=lw, label='Trained model');
    positive_rate = np.sum(data['y_true'] == 1) / data.shape[0]
    plt.plot([0, 1], [positive_rate, positive_rate], \
             color='red', lw=lw, linestyle='--', label='Random model');
    plt.xlim([-0.05, 1.05]);
    plt.ylim([0.45, 1.05]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    lgd = plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2);
    plt.xticks(np.arange(0, 1.01, 0.05), rotation=45);
    plt.yticks(np.arange(0, 1.01, 0.05));
    plt.grid(color='gray', linestyle='-', linewidth=1);
    plt.title('PR curve (AP = {:.3f})'.format(ap_score));
    plt.savefig(path, bbox_extra_artists=(lgd,), bbox_inches='tight');
    plt.close()


def plot_preds_distribution_by_bins(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(figsize=(16, 10))

    box_plot_data = []
    labels = []
    for name, group in data.groupby('bin'):
        labels.append(name)
        box_plot_data.append(group['y_pred'].values)

    box = axs.boxplot(box_plot_data, patch_artist=True, labels=labels)
    for patch in box['boxes']:
        patch.set_facecolor('green')
    axs.set_yscale('log')
    axs.set_xlabel('Bin number')
    axs.set_ylabel('Prediction')
    axs.set_title('Distribution of object predictions by bin')

    fig.savefig(path, bbox_inches='tight');
    plt.close()


def plot_distribution_of_logits(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(figsize=(16, 10))

    data['proba_logit'] = np.log(data['y_pred'].values / (1 - data['y_pred'].values))
    sns.kdeplot(data[data['y_true'] == 0]['proba_logit'], shade=True, color="r", label='Class 0 logits', ax=axs)
    sns.kdeplot(data[data['y_true'] == 1]['proba_logit'], shade=True, color="g", label='Class 1 logits', ax=axs)
    axs.set_xlabel('Logits')
    axs.set_ylabel('Density')
    axs.set_title('Logits distribution of object predictions (by classes)');
    fig.savefig(path, bbox_inches='tight');
    plt.close()


def plot_pie_f1_metric(data, F1_thresh, path):
    tn, fp, fn, tp = confusion_matrix(data['y_true'], (data['y_pred'] > F1_thresh).astype(int)).ravel()
    (_, prec), (_, rec), (_, F1), (_, _) = precision_recall_fscore_support(data['y_true'],
                                                                           (data['y_pred'] > F1_thresh).astype(int))

    sns.set(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(20, 10), subplot_kw=dict(aspect="equal"))

    recipe = ["{} True Positives".format(tp),
              "{} False Positives".format(fp),
              "{} False Negatives".format(fn),
              "{} True Negatives".format(tn)]

    wedges, texts = ax.pie([tp, fp, fn, tn], wedgeprops=dict(width=0.5), startangle=-40)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-", color='k'),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(recipe[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)

    ax.set_title(
        "Trained model: Precision = {:.2f}%, Recall = {:.2f}%, F1-Score = {:.2f}%".format(prec * 100, rec * 100, F1 * 100))
    plt.savefig(path, bbox_inches='tight');
    plt.close()
    return prec, rec, F1


def f1_score_w_co(data, min_co=.01, max_co=.99, step=0.01):
    data['y_pred'] = np.clip(np.ceil(data['y_pred'].values / step) * step, min_co, max_co)

    pos = data['y_true'].sum()
    neg = data['y_true'].shape[0] - pos

    grp = pd.DataFrame(data).groupby('y_pred')['y_true'].agg(['sum', 'count'])
    grp.sort_index(inplace=True)

    grp['fp'] = grp['sum'].cumsum()
    grp['tp'] = pos - grp['fp']
    grp['tn'] = (grp['count'] - grp['sum']).cumsum()
    grp['fn'] = neg - grp['tn']

    grp['pr'] = grp['tp'] / (grp['tp'] + grp['fp'])
    grp['rec'] = grp['tp'] / (grp['tp'] + grp['fn'])

    grp['f1_score'] = 2 * (grp['pr'] * grp['rec']) / (grp['pr'] + grp['rec'])

    best_score = grp['f1_score'].max()
    best_co = grp.index.values[grp['f1_score'] == best_score].mean()

    # print((y_pred < best_co).mean())

    return best_score, best_co


def get_bins_table(data):
    bins_table = data.groupby('bin').agg({'y_true': [len, np.mean], \
                                          'y_pred': [np.min, np.mean, np.max]}).reset_index()
    bins_table.columns = ['Bin number', 'Amount of objects', 'Mean target', \
                          'Min probability', 'Average probability', 'Max probability']
    return bins_table.to_html(index=False)


# Regression plots:

def plot_target_distribution_1(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(2, 1, figsize=(16, 20))

    sns.kdeplot(data['y_true'], shade=True, color="g", ax=axs[0])
    axs[0].set_xlabel('Target value')
    axs[0].set_ylabel('Density')
    axs[0].set_title('Target distribution (y_true)');

    sns.kdeplot(data['y_pred'], shade=True, color="r", ax=axs[1])
    axs[1].set_xlabel('Target value')
    axs[1].set_ylabel('Density')
    axs[1].set_title('Target distribution (y_pred)');

    fig.savefig(path, bbox_inches='tight');
    plt.close()


def plot_target_distribution_2(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(figsize=(16, 10))

    sns.kdeplot(data['y_true'], shade=True, color="g", label="y_true", ax=axs)
    sns.kdeplot(data['y_pred'], shade=True, color="r", label="y_pred", ax=axs)
    axs.set_xlabel('Target value')
    axs.set_ylabel('Density')
    axs.set_title('Target distribution');

    fig.savefig(path, bbox_inches='tight');
    plt.close()


def plot_target_distribution(data, path):
    data_pred = pd.DataFrame({'Target value': data['y_pred']})
    data_pred['source'] = 'y_pred'
    data_true = pd.DataFrame({'Target value': data['y_true']})
    data_true['source'] = 'y_true'
    data = pd.concat([data_pred, data_true], ignore_index=True)

    sns.set(style="whitegrid", font_scale=1.5)
    g = sns.displot(data, x="Target value", row="source", height=9, aspect=1.5, kde=True, color="m",
                    facet_kws=dict(margin_titles=True))
    g.fig.suptitle("Target distribution")
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.95)

    g.fig.savefig(path, bbox_inches='tight');
    plt.close()


def plot_error_hist(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(16, 10))

    g = sns.kdeplot(data['y_pred'] - data['y_true'], shade=True, color="m", ax=ax)
    ax.set_xlabel('Error = y_pred - y_true')
    ax.set_ylabel('Density')
    ax.set_title('Error histogram');

    fig.savefig(path, bbox_inches='tight');
    plt.close()


def plot_reg_scatter(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    g = sns.jointplot(x="y_pred", y="y_true", data=data, \
                      kind="reg", truncate=False, color="m", \
                      height=14)
    g.fig.suptitle("Scatter plot")
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.95)

    g.fig.savefig(path, bbox_inches='tight');
    plt.close()


# Multiclass plots:

def plot_confusion_matrix(data, path):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, ax = plt.subplots(figsize=(16, 12))

    cmat = confusion_matrix(data['y_true'], data['y_pred'], normalize='true')
    g = sns.heatmap(cmat, annot=True, linewidths=.5, cmap='Purples', ax=ax)
    ax.set_xlabel('y_pred')
    ax.set_ylabel('y_true')
    ax.set_title('Confusion matrix');

    fig.savefig(path, bbox_inches='tight');
    plt.close()


class ReportDeco:
    """
    Decorator to wrap :class:`~lightautoml.automl.base.AutoML` class to generate html report on ``fit_predict`` and ``predict``.

    Example:

        >>> report_automl = ReportDeco(output_path='output_path', report_file_name='report_file_name')(automl).
        >>> report_automl.fit_predict(train_data)
        >>> report_automl.predict(test_data)

    Report will be generated at output_path/report_file_name automatically.

    Warning:
         Do not use it just to inference (if you don't need report), because:

            - It needs target variable to calc performance metrics.
            - It takes additional time to generate report.
            - Dump of decorated automl takes more memory to store.

    To get unwrapped fitted instance to pickle
    and inferecne access ``report_automl.model`` attribute.

    """

    @property
    def model(self):
        """Get unwrapped model.

        Returns:
            model.

        """
        return self._model

    @property
    def mapping(self):
        return self._model.reader.class_mapping

    def __init__(self, *args, **kwargs):
        """

        Note:
            Valid kwargs are:

                - output_path: Folder with report files.
                - report_file_name: Name of main report file.

        Args:
            *args: Arguments.
            **kwargs: Additional parameters.

        """
        if not kwargs:
            kwargs = {}

        # self.task = kwargs.get('task', 'binary')
        self.n_bins = kwargs.get('n_bins', 20)
        self.template_path = kwargs.get('template_path', os.path.join(base_dir, 'lama_report_templates/'))
        self.output_path = kwargs.get('output_path', 'lama_report/')
        self.report_file_name = kwargs.get('report_file_name', 'lama_interactive_report.html')

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

        self._base_template_path = 'lama_base_template.html'
        self._model_section_path = 'model_section.html'
        self._train_set_section_path = 'train_set_section.html'
        self._results_section_path = 'results_section.html'

        self._inference_section_path = {'binary': 'binary_inference_section.html', \
                                        'reg': 'reg_inference_section.html', \
                                        'multiclass': 'multiclass_inference_section.html'}

        self.title = 'LAMA report'
        self.sections_order = ['intro', 'model', 'train_set', 'results']
        self._sections = {}
        self._sections['intro'] = '<p>This report was generated automatically.</p>'
        self._model_results = []

        self.generate_report()

    def __call__(self, model):
        self._model = model

        # AutoML only
        self.task = self._model.task._name  # valid_task_names = ['binary', 'reg', 'multiclass']

        # add informataion to report
        self._model_name = model.__class__.__name__
        self._model_parameters = json2html.convert(extract_params(model))
        self._model_summary = None

        self._sections = {}
        self._sections['intro'] = '<p>This report was generated automatically.</p>'
        self._model_results = []
        self._n_test_sample = 0

        self._generate_model_section()
        self.generate_report()
        return self

    def _binary_classification_details(self, data):
        self._inference_content['sample_bins_table'] = get_bins_table(data)
        prec, rec, F1 = plot_pie_f1_metric(data, self._F1_thresh, \
                                           path=os.path.join(self.output_path, self._inference_content['pie_f1_metric']))
        auc_score = plot_roc_curve_image(data, path=os.path.join(self.output_path, self._inference_content['roc_curve']))
        plot_pr_curve_image(data, path=os.path.join(self.output_path, self._inference_content['pr_curve']))
        plot_preds_distribution_by_bins(data, path=os.path.join(self.output_path, \
                                                                self._inference_content['preds_distribution_by_bins']))
        plot_distribution_of_logits(data, path=os.path.join(self.output_path, \
                                                            self._inference_content['distribution_of_logits']))
        return auc_score, prec, rec, F1

    def _regression_details(self, data):
        # graphics
        plot_target_distribution(data, path=os.path.join(self.output_path, self._inference_content['target_distribution']))
        plot_error_hist(data, path=os.path.join(self.output_path, self._inference_content['error_hist']))
        plot_reg_scatter(data, path=os.path.join(self.output_path, self._inference_content['scatter_plot']))
        # metrics
        mean_ae = mean_absolute_error(data['y_true'], data['y_pred'])
        median_ae = median_absolute_error(data['y_true'], data['y_pred'])
        mse = mean_squared_error(data['y_true'], data['y_pred'])
        r2 = r2_score(data['y_true'], data['y_pred'])
        evs = explained_variance_score(data['y_true'], data['y_pred'])
        return mean_ae, median_ae, mse, r2, evs

    def _multiclass_details(self, data):
        y_true = data['y_true']
        y_pred = data['y_pred']
        # precision
        p_micro = precision_score(y_true, y_pred, average='micro')
        p_macro = precision_score(y_true, y_pred, average='macro')
        p_weighted = precision_score(y_true, y_pred, average='weighted')
        # recall
        r_micro = recall_score(y_true, y_pred, average='micro')
        r_macro = recall_score(y_true, y_pred, average='macro')
        r_weighted = recall_score(y_true, y_pred, average='weighted')
        # f1-score
        f_micro = f1_score(y_true, y_pred, average='micro')
        f_macro = f1_score(y_true, y_pred, average='macro')
        f_weighted = f1_score(y_true, y_pred, average='weighted')

        # classification report for features
        if self.mapping:
            classes = sorted(self.mapping, key=self.mapping.get)
        else:
            classes = np.arange(self._N_classes)
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
        cls_report = pd.DataFrame({'Class name': classes, 'Precision': p, 'Recall': r, 'F1-score': f, 'Support': s})
        self._inference_content['classification_report'] = cls_report.to_html(index=False, float_format='{:.4f}'.format,
                                                                              justify='left')

        plot_confusion_matrix(data, path=os.path.join(self.output_path, self._inference_content['confusion_matrix']))

        return [p_micro, p_macro, p_weighted, r_micro, r_macro, r_weighted, f_micro, f_macro, f_weighted]

    def _collect_data(self, preds, sample):
        data = pd.DataFrame({'y_true': sample[self._target].values})
        if self.task in 'multiclass':
            if self.mapping is not None:
                data['y_true'] = np.array([self.mapping[y] for y in data['y_true'].values])
            data['y_pred'] = preds._data.argmax(axis=1)
            data = data[~np.isnan(preds._data).any(axis=1)]
        else:
            data['y_pred'] = preds._data[:, 0]
            data.sort_values('y_pred', ascending=False, inplace=True)
            data['bin'] = (np.arange(data.shape[0]) / data.shape[0] * self.n_bins).astype(int)
            data = data[~data['y_pred'].isnull()]
        return data

    def fit_predict(self, *args, **kwargs):
        """Wrapped ``automl.fit_predict`` method.

        Valid args, kwargs are the same as wrapped automl.

        Args:
            *args: Arguments.
            **kwargs: Additional parameters.

        Returns:
            OOF predictions.

        """
        # TODO: parameters parsing in general case

        preds = self._model.fit_predict(*args, **kwargs)

        train_data = kwargs["train_data"] if "train_data" in kwargs else args[0]
        input_roles = kwargs["roles"] if "roles" in kwargs else args[1]
        self._target = input_roles['target']
        valid_data = kwargs.get("valid_data", None)
        if valid_data is None:
            data = self._collect_data(preds, train_data)
        else:
            data = self._collect_data(preds, valid_data)

        self._inference_content = {}
        if self.task == 'binary':
            # filling for html
            self._inference_content = {}
            self._inference_content['roc_curve'] = 'valid_roc_curve.png'
            self._inference_content['pr_curve'] = 'valid_pr_curve.png'
            self._inference_content['pie_f1_metric'] = 'valid_pie_f1_metric.png'
            self._inference_content['preds_distribution_by_bins'] = 'valid_preds_distribution_by_bins.png'
            self._inference_content['distribution_of_logits'] = 'valid_distribution_of_logits.png'
            # graphics and metrics
            _, self._F1_thresh = f1_score_w_co(data)
            auc_score, prec, rec, F1 = self._binary_classification_details(data)
            # update model section
            evaluation_parameters = ['AUC-score', \
                                     'Precision', \
                                     'Recall', \
                                     'F1-score']
            self._model_summary = pd.DataFrame({'Evaluation parameter': evaluation_parameters, \
                                                'Validation sample': [auc_score, prec, rec, F1]})
        elif self.task == 'reg':
            # filling for html
            self._inference_content['target_distribution'] = 'valid_target_distribution.png'
            self._inference_content['error_hist'] = 'valid_error_hist.png'
            self._inference_content['scatter_plot'] = 'valid_scatter_plot.png'
            # graphics and metrics
            mean_ae, median_ae, mse, r2, evs = self._regression_details(data)
            # model section
            evaluation_parameters = ['Mean absolute error', \
                                     'Median absolute error', \
                                     'Mean squared error', \
                                     'R^2 (coefficient of determination)', \
                                     'Explained variance']
            self._model_summary = pd.DataFrame({'Evaluation parameter': evaluation_parameters, \
                                                'Validation sample': [mean_ae, median_ae, mse, r2, evs]})
        elif self.task == 'multiclass':
            self._N_classes = len(train_data[self._target].drop_duplicates())
            self._inference_content['confusion_matrix'] = 'valid_confusion_matrix.png'

            index_names = np.array([['Precision', 'Recall', 'F1-score'], \
                                    ['micro', 'macro', 'weighted']])
            index = pd.MultiIndex.from_product(index_names, names=['Evaluation metric', 'Average'])

            summary = self._multiclass_details(data)
            self._model_summary = pd.DataFrame({'Validation sample': summary}, index=index)

        self._inference_content['title'] = 'Results on validation sample'

        self._generate_model_section()

        # generate train data section
        self._train_data_overview = self._data_genenal_info(train_data)
        self._describe_roles(train_data)
        self._describe_dropped_features(train_data)
        self._generate_train_set_section()

        # generate fit_predict section
        self._generate_inference_section(data)
        self.generate_report()
        return preds

    def predict(self, *args, **kwargs):
        """Wrapped automl.predict method.

        Valid args, kwargs are the same as wrapped automl.

        Args:
            *args: arguments.
            **kwargs: additional parameters.

        Returns:
            predictions.

        """
        self._n_test_sample += 1
        # get predictions
        test_preds = self._model.predict(*args, **kwargs)

        test_data = kwargs["test"] if "test" in kwargs else args[0]
        data = self._collect_data(test_preds, test_data)

        if self.task == 'binary':
            # filling for html
            self._inference_content = {}
            self._inference_content['roc_curve'] = 'test_roc_curve_{}.png'.format(self._n_test_sample)
            self._inference_content['pr_curve'] = 'test_pr_curve_{}.png'.format(self._n_test_sample)
            self._inference_content['pie_f1_metric'] = 'test_pie_f1_metric_{}.png'.format(self._n_test_sample)
            self._inference_content['bins_preds'] = 'test_bins_preds_{}.png'.format(self._n_test_sample)
            self._inference_content['preds_distribution_by_bins'] = 'test_preds_distribution_by_bins_{}.png'.format(
                self._n_test_sample)
            self._inference_content['distribution_of_logits'] = 'test_distribution_of_logits_{}.png'.format(self._n_test_sample)
            # graphics and metrics
            auc_score, prec, rec, F1 = self._binary_classification_details(data)

            if self._n_test_sample >= 2:
                self._model_summary['Test sample {}'.format(self._n_test_sample)] = [auc_score, prec, rec, F1]
            else:
                self._model_summary['Test sample'] = [auc_score, prec, rec, F1]

        elif self.task == 'reg':
            # filling for html
            self._inference_content = {}
            self._inference_content['target_distribution'] = 'test_target_distribution_{}.png'.format(self._n_test_sample)
            self._inference_content['error_hist'] = 'test_error_hist_{}.png'.format(self._n_test_sample)
            self._inference_content['scatter_plot'] = 'test_scatter_plot_{}.png'.format(self._n_test_sample)
            # graphics
            mean_ae, median_ae, mse, r2, evs = self._regression_details(data)
            # update model section
            if self._n_test_sample >= 2:
                self._model_summary['Test sample {}'.format(self._n_test_sample)] = [mean_ae, median_ae, mse, r2, evs]
            else:
                self._model_summary['Test sample'] = [mean_ae, median_ae, mse, r2, evs]

        elif self.task == 'multiclass':
            self._inference_content['confusion_matrix'] = 'test_confusion_matrix_{}.png'.format(self._n_test_sample)
            test_summary = self._multiclass_details(data)
            if self._n_test_sample >= 2:
                self._model_summary['Test sample {}'.format(self._n_test_sample)] = test_summary
            else:
                self._model_summary['Test sample'] = test_summary

        # layout depends on number of test samples
        if self._n_test_sample >= 2:
            self._inference_content['title'] = 'Results on test sample {}'.format(self._n_test_sample)

        else:
            self._inference_content['title'] = 'Results on test sample'

        # update model section
        self._generate_model_section()

        # generate predict section    
        self._generate_inference_section(data)
        self.generate_report()
        return test_preds

    def _data_genenal_info(self, data):
        general_info = pd.DataFrame(columns=['Parameter', 'Value'])
        general_info.loc[0] = ('Number of records', data.shape[0])
        general_info.loc[1] = ('Total number of features', data.shape[1])
        general_info.loc[2] = ('Used features', len(self._model.reader._used_features))
        general_info.loc[3] = ('Dropped features', len(self._model.reader._dropped_features))
        # general_info.loc[4] = ('Number of positive cases', np.sum(data[self._target] == 1))
        # general_info.loc[5] = ('Number of negative cases', np.sum(data[self._target] == 0))
        return general_info.to_html(index=False, justify='left')

    def _describe_roles(self, train_data):

        # detect feature roles
        roles = self._model.reader._roles
        numerical_features = [feat_name for feat_name in roles if roles[feat_name].name == 'Numeric']
        categorical_features = [feat_name for feat_name in roles if roles[feat_name].name == 'Category']
        datetime_features = [feat_name for feat_name in roles if roles[feat_name].name == 'Datetime']

        # numerical roles
        numerical_features_df = []
        for feature_name in numerical_features:
            item = {'Feature name': feature_name}
            item['NaN ratio'] = "{:.4f}".format(train_data[feature_name].isna().sum() / train_data.shape[0])
            values = train_data[feature_name].dropna().values
            item['min'] = np.min(values)
            item['quantile_25'] = np.quantile(values, 0.25)
            item['average'] = np.mean(values)
            item['median'] = np.median(values)
            item['quantile_75'] = np.quantile(values, 0.75)
            item['max'] = np.max(values)
            numerical_features_df.append(item)
        if numerical_features_df == []:
            self._numerical_features_table = None
        else:
            self._numerical_features_table = pd.DataFrame(numerical_features_df).to_html(index=False,
                                                                                         float_format='{:.2f}'.format,
                                                                                         justify='left')
        # categorical roles
        categorical_features_df = []
        for feature_name in categorical_features:
            item = {'Feature name': feature_name}
            item['NaN ratio'] = "{:.4f}".format(train_data[feature_name].isna().sum() / train_data.shape[0])
            value_counts = train_data[feature_name].value_counts(normalize=True)
            values = value_counts.index.values
            counts = value_counts.values
            item['Number of unique values'] = len(counts)
            item['Most frequent value'] = values[0]
            item['Occurance of most frequent'] = "{:.1f}%".format(100 * counts[0])
            item['Least frequent value'] = values[-1]
            item['Occurance of least frequent'] = "{:.1f}%".format(100 * counts[-1])
            categorical_features_df.append(item)
        if categorical_features_df == []:
            self._categorical_features_table = None
        else:
            self._categorical_features_table = pd.DataFrame(categorical_features_df).to_html(index=False, justify='left')
        # datetime roles
        datetime_features_df = []
        for feature_name in datetime_features:
            item = {'Feature name': feature_name}
            item['NaN ratio'] = "{:.4f}".format(train_data[feature_name].isna().sum() / train_data.shape[0])
            values = train_data[feature_name].dropna().values
            item['min'] = np.min(values)
            item['max'] = np.max(values)
            item['base_date'] = self._model.reader._roles[feature_name].base_date
            datetime_features_df.append(item)
        if datetime_features_df == []:
            self._datetime_features_table = None
        else:
            self._datetime_features_table = pd.DataFrame(datetime_features_df).to_html(index=False, justify='left')

    def _describe_dropped_features(self, train_data):
        self._max_nan_rate = self._model.reader.max_nan_rate
        self._max_constant_rate = self._model.reader.max_constant_rate
        self._features_dropped_list = self._model.reader._dropped_features
        # dropped features table
        dropped_list = [col for col in self._features_dropped_list if col != self._target]
        if dropped_list == []:
            self._dropped_features_table = None
        else:
            dropped_nan_ratio = train_data[dropped_list].isna().sum() / train_data.shape[0]
            dropped_most_occured = pd.Series(np.nan, index=dropped_list)
            for col in dropped_list:
                col_most_occured = train_data[col].value_counts(normalize=True).values
                if len(col_most_occured) > 0:
                    dropped_most_occured[col] = col_most_occured[0]
            dropped_features_table = pd.DataFrame({'nan_rate': dropped_nan_ratio, 'constant_rate': dropped_most_occured})
            self._dropped_features_table = dropped_features_table.reset_index().rename(
                columns={'index': 'Название переменной'}).to_html(index=False, justify='left')

    def _generate_model_section(self):
        model_summary = None
        if self._model_summary is not None:
            model_summary = self._model_summary.to_html(index=self.task == 'multiclass', justify='left',
                                                        float_format='{:.4f}'.format)

        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        model_section = env.get_template(self._model_section_path).render(
            model_name=self._model_name,
            model_parameters=self._model_parameters,
            model_summary=model_summary
        )
        self._sections['model'] = model_section

    def _generate_train_set_section(self):
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        train_set_section = env.get_template(self._train_set_section_path).render(
            train_data_overview=self._train_data_overview,
            numerical_features_table=self._numerical_features_table,
            categorical_features_table=self._categorical_features_table,
            datetime_features_table=self._datetime_features_table,
            target=self._target,
            max_nan_rate=self._max_nan_rate,
            max_constant_rate=self._max_constant_rate,
            dropped_features_table=self._dropped_features_table
        )
        self._sections['train_set'] = train_set_section

    def _generate_inference_section(self, data):
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        inference_section = env.get_template(self._inference_section_path[self.task]).render(self._inference_content)
        self._model_results.append(inference_section)

    def _generate_results_section(self):
        if self._model_results:
            env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
            results_section = env.get_template(self._results_section_path).render(
                model_results=self._model_results
            )
            self._sections['results'] = results_section

    def generate_report(self):
        # collection sections
        self._generate_results_section()
        sections_list = []
        for sec_name in self.sections_order:
            if sec_name in self._sections:
                sections_list.append(self._sections[sec_name])
        # put sections inside
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        report = env.get_template(self._base_template_path).render(
            title=self.title,
            sections=sections_list
        )
        with open(os.path.join(self.output_path, self.report_file_name), "w", encoding='utf-8') as f:
            f.write(report)


_default_wb_report_params = {"automl_date_column": "",
                             "report_name": 'autowoe_report.html',
                             "report_version_id": 1,
                             "city": "",
                             "model_aim": "",
                             "model_name": "",
                             "zakazchik": "",
                             "high_level_department": "",
                             "ds_name": "",
                             "target_descr": "",
                             "non_target_descr": ""}


class ReportDecoWhitebox(ReportDeco):
    """
    Special report wrapper for :class:`~lightautoml.automl.presets.whitebox_presets.WhiteBoxPreset`.
    Usage case is the same as main
    :class:`~lightautoml.report.report_deco.ReportDeco` class.
    It generates same report as :class:`~lightautoml.report.report_deco.ReportDeco` ,
    but with additional whitebox report part.

    Difference:

        - report_automl.predict gets additional report argument.
          It stands for updating whitebox report part.
          Calling ``report_automl.predict(test_data, report=True)``
          will update test part of whitebox report.
          Calling ``report_automl.predict(test_data, report=False)``
          will extend general report with.
          New data and keeps whitebox part as is (much more faster).
        - :class:`~lightautoml.automl.presets.whitebox_presets.WhiteBoxPreset`
          should be created with parameter ``general_params={'report': True}``
          to get white box report part.
          If ``general_params`` set to ``{'report': False}``,
          only standard ReportDeco part will be created (much faster).

    """

    @property
    def model(self):
        """Get unwrapped WhiteBox.

        Returns:
            model.

        """
        # this is made to remove heavy whitebox inner report deco
        model = copy(self._model)
        try:
            model_wo_report = model.whitebox.model
        except AttributeError:
            return self._model

        pipe = copy(self._model.levels[0][0])
        ml_algo = copy(pipe.ml_algos[0])

        ml_algo.models = [model_wo_report]
        pipe.ml_algos = [ml_algo]

        model.levels = [[pipe]]

        return model

    @property
    def content(self):
        return self._model.whitebox._ReportDeco__stat

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.wb_report_params = copy(_default_wb_report_params)

        # self.wb_report_params = wb_report_params
        self.wb_report_params['output_path'] = self.output_path
        self._whitebox_section_path = 'whitebox_section.html'
        self.sections_order.append('whitebox')

    def fit_predict(self, *args, **kwargs):
        """Wrapped :meth:`AutoML.fit_predict` method.

        Valid args, kwargs are the same as wrapped automl.

        Args:
            *args: Arguments.
            **kwargs: Additional parameters.

        Returns:
            OOF predictions.

        """
        predict_proba = super().fit_predict(*args, **kwargs)

        if self._model.general_params['report']:
            self._generate_whitebox_section()
        else:
            logger.warning("Whitebox part is not created. Fit WhiteBox with general_params['report'] = True")

        self.generate_report()
        return predict_proba

    def predict(self, *args, **kwargs):
        """Wrapped :meth:`AutoML.predict` method.

        Valid args, kwargs are the same as wrapped automl.

        Args:
            *args: Arguments.
            **kwargs: Additional parameters.

        Returns:
            Predictions.

        """
        if len(args) >= 2:
            args = (args[0],)

        kwargs['report'] = self._model.general_params['report']

        predict_proba = super().predict(*args, **kwargs)

        if self._model.general_params['report']:
            self._generate_whitebox_section()
        else:
            logger.warning("Whitebox part is not created. Fit WhiteBox with general_params['report'] = True")

        self.generate_report()
        return predict_proba

    def _generate_whitebox_section(self):
        self._model.whitebox.generate_report(self.wb_report_params)
        content = self.content.copy()

        if self._n_test_sample >= 2:
            content['n_test_sample'] = self._n_test_sample
        content['model_coef'] = pd.DataFrame(content['model_coef'], \
                                             columns=['Feature name', 'Coefficient']).to_html(index=False)
        content['p_vals'] = pd.DataFrame(content['p_vals'], \
                                         columns=['Feature name', 'P-value']).to_html(index=False)
        content['p_vals_test'] = pd.DataFrame(content['p_vals_test'], \
                                              columns=['Feature name', 'P-value']).to_html(index=False)
        content['train_vif'] = pd.DataFrame(content['train_vif'], \
                                            columns=['Feature name', 'VIF value']).to_html(index=False)
        content['psi_total'] = pd.DataFrame(content['psi_total'], \
                                            columns=['Feature name', 'PSI value']).to_html(index=False)
        content['psi_zeros'] = pd.DataFrame(content['psi_zeros'], \
                                            columns=['Feature name', 'PSI value']).to_html(index=False)
        content['psi_ones'] = pd.DataFrame(content['psi_ones'], \
                                           columns=['Feature name', 'PSI value']).to_html(index=False)

        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        self._sections['whitebox'] = env.get_template(self._whitebox_section_path).render(content)

        
        
def plot_data_hist(data, title='title', bins=100, path=None):
    sns.set(style="whitegrid", font_scale=1.5)
    fig, axs = plt.subplots(figsize=(16, 10))
    sns.distplot(data, bins=bins, color="m", ax=axs)
    axs.set_title(title);
    fig.savefig(path, bbox_inches='tight');
    plt.close()


class ReportDecoNLP(ReportDeco):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._nlp_section_path = 'nlp_section.html'
        self._nlp_subsection_path = 'nlp_subsection.html'
        self._nlp_subsections = []
        self.sections_order.append('nlp')
        
        

    def __call__(self, model):
        self._model = model

        # AutoML only
        self.task = self._model.task._name  # valid_task_names = ['binary', 'reg', 'multiclass']

        # add informataion to report
        self._model_name = model.__class__.__name__
        self._model_parameters = json2html.convert(extract_params(model))
        self._model_summary = None

        self._sections = {}
        self._sections['intro'] = '<p>This report was generated automatically.</p>'
        self._model_results = []
        self._n_test_sample = 0

        self._generate_model_section()
        self.generate_report()
        return self
    
    
    def fit_predict(self, *args, **kwargs):
        preds = super().fit_predict(*args, **kwargs)
        
        train_data = kwargs["train_data"] if "train_data" in kwargs else args[0]
        roles = kwargs["roles"] if "roles" in kwargs else args[1]
        
        self._text_fields = roles['text']
        for text_field in self._text_fields:
            content = {}
            content['title'] = 'Text field: ' + text_field
            content['char_len_hist'] = text_field + '_char_len_hist.png'
            plot_data_hist(data=train_data[text_field].apply(len).values,
                           path = os.path.join(self.output_path, content['char_len_hist']),
                           title='Length in char')
            content['tokens_len_hist'] = text_field + '_tokens_len_hist.png'
            plot_data_hist(data=train_data[text_field].str.split(' ').apply(len).values,
                           path = os.path.join(self.output_path, content['tokens_len_hist']),
                           title='Length in tokens')
            self._generate_nlp_subsection(content)
        # Concatenated text fields
        if len(self._text_fields) >= 2:
            all_fields = train_data[self._text_fields].agg(' '.join, axis=1)
            content = {}
            content['title'] = 'Concatenated text fields'
            content['char_len_hist'] = 'concat_char_len_hist.png'
            plot_data_hist(data=all_fields.apply(len).values,
                           path = os.path.join(self.output_path, content['char_len_hist']),
                           title='Length in char')
            content['tokens_len_hist'] = 'concat_tokens_len_hist.png'
            plot_data_hist(data=all_fields.str.split(' ').apply(len).values,
                           path = os.path.join(self.output_path, content['tokens_len_hist']),
                           title='Length in tokens')
            self._generate_nlp_subsection(content)
            
        
        self._generate_nlp_section()
        self.generate_report()
        return preds
    
    
    def _generate_nlp_subsection(self, content):
        # content has the following fields:
        # title:            subsection title
        # char_len_hist:    path to histogram of text length (number of chars)
        # tokens_len_hist:  path to histogram of text length (number of tokens)
        env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
        nlp_subsection = env.get_template(self._nlp_subsection_path).render(content)
        self._nlp_subsections.append(nlp_subsection)

    
    def _generate_nlp_section(self):
        if self._model_results:
            env = Environment(loader=FileSystemLoader(searchpath=self.template_path))
            nlp_section = env.get_template(self._nlp_section_path).render(
                nlp_subsections=self._nlp_subsections
            )
            self._sections['nlp'] = nlp_section
