"""utility libraries"""
from cytoolz.functoolz import memoize


@memoize
def load_context(path=None):
    """Load the context object. Unless otherwise specified, we will search
    in two directory, first we will try to look for a context.ini in the
    current working directory and the parent directory. Then, we will try
    to load it in a default path at ~/.cosmosrc. This is decorated with
    memoize so that the object is cached and can be easily loaded on
    subsequent calls.

    """
    import os, os.path as op
    from configparser import ConfigParser, ExtendedInterpolation as EI
    # is path is provided, load context from there
    if path:
        config_path = path
    else:
        # search for the context.ini files in os
        if op.exists(op.join(os.getcwd(),'context.ini')):
            config_path = op.join(os.getcwd(),'context.ini')
        elif op.exists(op.join(os.getcwd(),'..','context.ini')):
            config_path = op.join(os.getcwd(),'..','context.ini')
        elif os.path.exists("~/.cosmosrc"):
            config_path = "~/.cosmosrc"
        else:
            return None
    # if we are here we will have found one context
    print(f"Loading context: {config_path}")
    context = ConfigParser(config_path, EI)
    return context
