LATEX_TABLE_FOOTER = r"""
\end{tabular}
\end{document}
"""
LATEX_TABLE_HEADER = r"""
\documentclass{standalone}
%%\usepackage[landscape, paperwidth=15cm, paperheight=30cm, margin=0mm]{geometry}
\usepackage{multirow}
\usepackage{color, colortbl}
\usepackage{xcolor, soul}
\usepackage{amssymb}
\definecolor{Gray}{gray}{0.9}
\definecolor{StrongGray}{gray}{0.7}
\begin{document}
\begin{tabular}{%s}
\hline
\rowcolor{StrongGray}
Dataset & %s \\"""
LATEX_TABLE_METRICS_INTERACTIONS_HEADER = r"""
\hline
\hline
\rowcolor{Gray}
Measure & %s \\
\hline
\rowcolor{Gray}
T & %s \\
\hline
\hline
"""


def generate_table_spec(nums_interactions_to_show, num_datasets_preprocessors):
    res = '|'
    for i in range(1 +
                   len(nums_interactions_to_show) * num_datasets_preprocessors):
        res += 'c'
        if i % (len(nums_interactions_to_show)) == 0:
            res += '|'
    return res


def generate_datasets_line(nums_interactions_to_show, preprocessors_names):
    return ' & '.join([
        r"\multicolumn{%d}{c|}{%s}" % (len(nums_interactions_to_show), i)
        for i in preprocessors_names
    ])


def generate_metrics_header_line(nums_interactions_to_show, num_preprocessors,
                                 metric_name):
    return ' & '.join(
        map(
            lambda x: r"\multicolumn{%d}{c|}{%s}" %
            (len(nums_interactions_to_show), x),
            [metric_name] * num_preprocessors))


def generate_interactions_header_line(nums_interactions_to_show,
                                      num_preprocessors):
    return ' & '.join([' & '.join(map(str, nums_interactions_to_show))] *
                      num_preprocessors)


def generate_metric_interactions_header(nums_interactions_to_show,
                                        num_preprocessors, metric_name):
    btex = LATEX_TABLE_METRICS_INTERACTIONS_HEADER % (
        generate_metrics_header_line(nums_interactions_to_show,
                                     num_preprocessors, metric_name),
        generate_interactions_header_line(nums_interactions_to_show,
                                          num_preprocessors))
    return btex
