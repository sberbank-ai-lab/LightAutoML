

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
