import random, string

def rand_chars(n):
    """Generate n random characters"""
    return ''.join(random.choice(string.ascii_letters) for x in range(n))
