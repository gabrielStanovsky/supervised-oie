def joinstr(delim, ls, nl = True):
    """
    Join and map str
    nl - controls whether a new line is added at the end of the output
    """
    return delim.join(map(str, ls)) + ("\n" if nl else "")
