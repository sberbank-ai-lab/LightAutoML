"""Profiler."""

import inspect
import types
from typing import Optional

import networkx as nx
import pandas as pd

from .logging import get_logger

logger = get_logger(__name__)


def get_last_element(x): return x[-1]


def get_string(x): return x.__module__ + '.' + x.__qualname__


class Profiler:
    """AutoML algorithm statistics profiler."""

    _DROP_FUNCS = ['get_record_history_wrapper',
                   '__repr__', '__hash__', '__eq__']

    _CSS_ELEM = '''
        <style type="text/css">
        /* Remove default bullets */
        ul, #myUL {
          list-style-type: none;
        }

        /* Remove margins and padding from the parent ul */
        #myUL {
          margin: 0;
          padding: 0;
        }

        /* Style the caret/arrow */
        .caret {
          cursor: pointer; 
          user-select: none; /* Prevent text selection */
        }

        /* Create the caret/arrow with a unicode, and style it */
        .caret::before {
          content: "\\27A4";
          color: black;
          display: inline-block;
          margin-right: 6px;
        }

        /* Rotate the caret/arrow icon when clicked on (using JavaScript) */
        .caret-down::before {
          content: "\\25BC";
          color: black;
          display: inline-block;
          margin-right: 6px; 
        }

        /* Hide the nested list */
        .nested {
          display: none;
        }

        /* Show the nested list when the user clicks on the caret/arrow (with JavaScript) */
        .active {
          display: block;
        }

        .sub-items li {
          position: relative;
          margin-left: -32px;
          padding-left: 10px;
          padding-top: 2px;
          border-left: 1px dotted #0000BB;
        }
        </style>
    '''

    _JS_ELEM = '''
        <script>
        var toggler = document.getElementsByClassName("caret");
        var i;

        for (i = 0; i < toggler.length; i++) {
          toggler[i].addEventListener("click", function() {
            this.parentElement.querySelector(".nested").classList.toggle("active");
            this.classList.toggle("caret-down");
          });
        }
        </script>
    '''

    _GRADIENT_COLORS = ['669900', '6A9900', '6E9900', '729900', '779900', '7B9900', '7F9900', '839900',
                        '889900', '8C9900', '909900', '949900', '999900', '9D9900', 'A19900', 'A59900',
                        'AA9900', 'AE9900', 'B29900', 'B69900', 'BB9900', 'BF9900', 'C39900', 'C79900',
                        'CC9900', 'CC9900', 'CC9400', 'CC9000', 'CC8C00', 'CC8800', 'CC8300', 'CC7F00',
                        'CC7B00', 'CC7700', 'CC7200', 'CC6E00', 'CC6A00', 'CC6600', 'CC6100', 'CC5D00',
                        'CC5900', 'CC5500', 'CC5000', 'CC4C00', 'CC4800', 'CC4400', 'CC3F00', 'CC3B00',
                        'CC3700', 'CC3300']

    def __init__(self, drop_funcs: Optional[list] = None, ):
        """Profiler init function

        Args:
            drop_funcs: Function names that will not be inspected.

        """

        self.queue = []
        if drop_funcs is None:
            drop_funcs = Profiler._DROP_FUNCS
        self.drop_funcs = drop_funcs
        # ====================================
        self.all_funcs = None
        self.full_stats_df = None
        self.prof_graph = None

        self._get_all_funcs()

    def _get_all_funcs(self):
        """Get all funcs of lightautoml module to gather its statistics."""
        queue = [__import__('lightautoml')]

        modules = set()
        while len(queue) > 0:
            queue_elem = queue.pop(0)
            added_cnt = 0
            for el in dir(queue_elem):
                new_el = getattr(queue_elem, el)
                if type(new_el) == types.ModuleType and new_el.__name__.startswith('lightautoml'):
                    queue.append(new_el)
                    added_cnt += 1
            if added_cnt == 0:
                modules.add(queue_elem)

        all_classes = set()
        self.all_funcs = set()
        for modul in modules:
            cls_from_module = [x[1] for x in inspect.getmembers(modul, inspect.isclass) if
                               x[1].__module__.startswith(modul.__name__)]
            all_classes.update(cls_from_module)
            funcs_from_module = [x[1] for x in inspect.getmembers(modul, inspect.isfunction) if
                                 x[1].__module__.startswith(modul.__name__)]
            self.all_funcs.update(funcs_from_module)
            meth_from_module = [x[1] for x in inspect.getmembers(modul, inspect.ismethod) if
                                x[0] not in self.drop_funcs and x[1].__module__.startswith(modul.__name__)]
            self.all_funcs.update(meth_from_module)

        for cls in all_classes:
            cls_name = cls.__name__.split('.')[-1]
            funcs_from_class = [x[1] for x in inspect.getmembers(cls, inspect.isfunction) if
                                x[1].__qualname__.startswith(cls_name)]
            self.all_funcs.update(funcs_from_class)
            meth_from_module = [x[1] for x in inspect.getmembers(cls, inspect.ismethod) if
                                x[0] not in self.drop_funcs and x[1].__qualname__.startswith(cls_name)]
            self.all_funcs.update(meth_from_module)

        self.all_funcs = sorted(
            list(self.all_funcs), key=get_string)
        self.all_funcs = [
            f for f in self.all_funcs if f.__name__ not in self.drop_funcs]
        logger.debug('ALL_FUNCS len = {}'.format(len(self.all_funcs)))

    def _aggregate_stats_from_functions(self):
        """Gather stats from all found functions into one dataframe."""
        cols_df = ['call_num', 'elapsed_secs', 'timestamp',
                   'prefixed_func_name', 'caller_chain']
        dfs_arr = []
        for f in self.all_funcs:
            if not hasattr(f, 'stats'):
                logger.debug('\t Func with no stats - {}'.format(f))
                continue
            cur_df = pd.DataFrame([[getattr(el, col) for col in cols_df]
                                   for el in f.stats.history], columns=cols_df)
            if len(cur_df) > 0:
                if cur_df['call_num'].value_counts().values[0] > 1:
                    cur_df = cur_df.sort_values(
                        'timestamp', kind='mergesort').reset_index(drop=True)
                    cur_df['call_num'] = list(range(1, len(cur_df) + 1))
                cur_df.insert(4, 'run_fname',
                              cur_df['prefixed_func_name'].astype(str) + ' [' + cur_df['call_num'].astype(str) + ']')
                cur_df['caller_chain'] = cur_df['caller_chain'].map(get_last_element)
                dfs_arr.append(cur_df)
            f.stats.clear_history()

        if len(dfs_arr) == 0:
            logger.debug('There is no info from functions to profile... Abort')
            return

        self.full_stats_df = pd.concat(dfs_arr)
        self.full_stats_df = self.full_stats_df.sort_values(
            ['timestamp', 'call_num']).reset_index(drop=True)

        logger.debug('FULL_STATS_DF shape = {}'.format(self.full_stats_df.shape))
        logger.debug('RUN_FNAME vc head: \n {}'.format(
            self.full_stats_df['run_fname'].value_counts().head()))

    def _generate_and_check_calls_graph(self):
        """Build graph from functions calls and check its correctness."""
        self.prof_graph = nx.Graph()
        self.prof_graph.add_edges_from(list(zip(self.full_stats_df['caller_chain'].values,
                                                self.full_stats_df['run_fname'].values)))

        cc = list(nx.connected_components(self.prof_graph))
        logger.debug('CONNECTED COMPONENTS cnt = {}'.format(len(cc)))
        assert len(
            cc) == 1, 'Profiler calls graph has more than 1 connected component but it must be a tree...'

        path_lens = {x: len(y) - 1
                     for x, y in nx.shortest_path(self.prof_graph,
                                                  source=self.full_stats_df.caller_chain.values[0]).items()}
        logger.debug('PATH LENS describe: \n {}'.format(pd.Series(
            [x[1] for x in path_lens.items()]).describe()))

        self.full_stats_df['level'] = self.full_stats_df['run_fname'].map(
            path_lens)
        self.full_stats_df = self.full_stats_df.sort_values(['timestamp', 'call_num', 'level'],
                                                            kind='mergesort').reset_index(drop=True)

    def _create_html_report(self, report_path: str):
        """Create HTML report for LightAutoML profiling."""
        df = self.full_stats_df[['run_fname', 'level', 'elapsed_secs']]
        df = pd.concat([pd.DataFrame({'run_fname': ['ROOT'], 'level': [
            0], 'elapsed_secs': [0.0]}), df]).reset_index(drop=True)
        nodes = list(df['run_fname'].values)
        levels = df['level'].values
        times = df['elapsed_secs'].values
        times[0] = sum(times[levels == 1])
        nodes[0] = '<span>' + \
                   '<b>{:.3f} secs</b>'.format(times[0]) + ', ' + nodes[0] + '</span>'
        for i in range(1, len(times)):
            times[i] = times[i] * 100 / times[0]
            nodes[i] = '<span style="color:#000000; background: #{0}AA;";>'.format(Profiler._GRADIENT_COLORS[int(times[i] // 2)]) \
                       + '<b>{:.2f}%</b>'.format(
                times[i]) + '</span>, <span style="color:#000000;";>' + nodes[i] + '</span>'

        with open(report_path, 'w') as fout:
            fout.write(Profiler._CSS_ELEM + '\n')
            fout.write('<body>\n')

            fout.write('<ul id="myUL">' + '\n')
            for i in range(len(nodes) - 1):
                if levels[i] < levels[i + 1]:
                    fout.write('\t' * levels[i] + '<li><span class="caret">{}</span><ul class="nested sub-items">'.format(
                        nodes[i]) + '\n')
                elif levels[i] == levels[i + 1]:
                    fout.write(
                        '\t' * levels[i] + '<li>&emsp;&nbsp;{}</li>'.format(nodes[i]) + '\n')
                else:
                    fout.write(
                        '\t' * levels[i] + '<li>&emsp;&nbsp;{}</li>'.format(nodes[i]) + '\n')
                    for t in range(levels[i], levels[i + 1], -1):
                        fout.write('\t' * (t - 1) + '</ul></li>' + '\n')
            for t in range(levels[-1], 0, -1):
                fout.write('\t' * (t - 1) + '</ul></li>' + '\n')
            fout.write('</ul>')
            fout.write(Profiler._JS_ELEM + '\n')
            fout.write('<br>\n')

            fout.write(
                '<span style="color:#669900;";><b>LOAD_OK</b></span><br>')
            fout.write('</body>\n')

    def profile(self, report_path: str = './profile_report.html'):
        """Create profile of algorithm.

        Args:
            report_path: path to save profile.

        """
        self._aggregate_stats_from_functions()
        if self.full_stats_df is None:
            return
        self._generate_and_check_calls_graph()
        self._create_html_report(report_path)

    def change_deco_settings(self, new_settings: dict):
        """Update profiling deco settings.

        Args:
            new_settings: dict with new key-values for decorator.

        """
        for f in self.all_funcs:
            if not hasattr(f, 'record_history_settings'):
                logger.debug('\t Func with no decorator - {}'.format(f))
                continue
            for k in new_settings:
                f.record_history_settings[k] = new_settings[k]
