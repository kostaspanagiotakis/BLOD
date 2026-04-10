def moving_average(values, window=50):
    if len(values) < window:
        return values
    out, running = [], 0.0
    for i, v in enumerate(values):
        running += v
        if i >= window:
            running -= values[i - window]
        if i >= window - 1:
            out.append(running / window)
    return out