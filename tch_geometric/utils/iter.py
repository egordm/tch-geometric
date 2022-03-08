def zip_dict(d1, d2):
    """
    Zip two dictionaries together.
    """
    yield from ((k, (d1[k], d2[k])) for k in d1.keys() & d2.keys())
