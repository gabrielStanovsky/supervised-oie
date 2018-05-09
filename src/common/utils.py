import pandas as pd
import pdb

def joinstr(delim, ls, nl = True):
    """
    Join and map str
    nl - controls whether a new line is added at the end of the output
    """
    return delim.join(map(str, ls)) + ("\n" if nl else "")

def concat_dfs(df1, df2, running_keys):
    """
    Returns a concatenation of df1 and df2
    which fixes running keys (list of strings)
    as continuation.
    """
    df2 = df2.copy()
    for run_key in running_keys:
        # Find the last value for this key in df1
        base = df1[run_key].values[-1] + 1

        # Modify df2 to continue df1 on this key
        df2[run_key] = [orig_key + base
                        for orig_key
                        in df2[run_key].values]

    return pd.concat([df1, df2])


def df_to_conll(df, out_fn):
    """
    Write a conll representation of this df to file
    with column header and newlines between sentences.
    """
    header = list(df.columns)
    sents = [df[df.run_id == run_id]
             for run_id
             in sorted(set(df.run_id.values))]

    with open(out_fn, 'w') as fout:
        fout.write('\t'.join(header) + '\n')
        fout.write('\n'.join([sent.to_csv(header = False,
                                          index = False,
                                          sep = '\t')
                              for sent in sents]))
