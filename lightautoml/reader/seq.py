import numpy as np

def rolling_window(a, window, step=1, from_last=True):
    'from_last == True - will cut first step-1 elements'
    return np.lib.stride_tricks.sliding_window_view(a, window)[(len(a) - window) % step if from_last else 0:][::step]

class CreateInd:
    def __init__(self, top_k=5):
        self.top_k = top_k
        pass

    def create_data(self, data):
        ind = np.arange(len(data))
        pass

    def create_target(self):
        pass

    def _groupby_ind(self, data):
        pass

    def _timeborder_ind(self, data):
        pass

    def _kclosest_ind(self, data):
        pass


class TopInd:
    def __init__(self, n_target=7, history=100, step=3, from_last=True, test_last=True, scheme=None):
        self.n_target = n_target
        self.history = history
        self.step = step
        self.from_last = from_last
        self.test_last = test_last

    def read(self, data, plain_data=None):
        self.len_data = len(data)

        ## TO DO:
        # add asserts

    def create_test(self, data=None, plain_data=None):
        # for predicting future
        return rolling_window(np.arange(self.len_data if data is None else len(data)), self.history, self.step, self.from_last)[
               -1 if self.test_last else 0:, :]

    def create_data(self, data=None, plain_data=None):
        return rolling_window(np.arange(self.len_data if data is None else len(data))[:-self.n_target], self.history, self.step,
                              self.from_last)

    def create_target(self, data=None, plain_data=None):
        return rolling_window(np.arange(self.len_data if data is None else len(data))[self.history:], self.n_target, self.step,
                              self.from_last)


class IDSInd:
    def __init__(self, scheme, **kwargs):
        self.scheme = scheme

    def read(self, data, plain_data=None):
        self.len_data = len(data)

        ## TO DO:
        # add asserts

    def create_test(self, data, plain_data):
        # for predicting future
        return self.create_data(data, plain_data)

    def create_data(self, data, plain_data):
        ids = data.reset_index().groupby(self.scheme['from_id'])['index'].apply(self._to_list).to_dict()
        s = plain_data[self.scheme['to_id']].map(ids)
        s.loc[s.isna()] = [[] for i in range(len(s.loc[s.isna()]))]
        return s.values

    def create_target(self, data=None, plain_data=None):
        return None

    @staticmethod
    def _to_list(x):
        return list(x)
