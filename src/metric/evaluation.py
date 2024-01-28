import numpy as np
import pandas as pd


def probability_table(m=None, n=1000000, do={}, dat=None):
    assert m is not None or dat is not None

    if dat is None:
        dat = m(n, do=do, evaluating=True)

    cols = dict()
    for v in sorted(dat):
        result = dat[v].detach().numpy()
        for i in range(result.shape[1]):
            cols["{}{}".format(v, i)] = np.squeeze(result[:, i])

    df = pd.DataFrame(cols)
    return (df.groupby(list(df.columns))
            .apply(lambda x: len(x) / len(df))
            .rename('P(V)').reset_index()
            [[*df.columns, 'P(V)']])


def kl(real_data, fake_data):
    m_table = fake_data
    t_table = real_data
    cols = list(t_table.columns[:-1])
    joined_table = t_table.merge(m_table, how='left', on=cols, suffixes=['_t', '_m']).fillna(0.0000001)
    p_t = joined_table['P(V)_t']
    p_m = joined_table['P(V)_m']
    return (p_t * (np.log(p_t) - np.log(p_m))).sum()
