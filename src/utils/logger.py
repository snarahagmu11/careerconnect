import logging

def setup_logger(name="careerconnect", level=logging.INFO):
    lg = logging.getLogger(name)
    lg.setLevel(level)
    if not lg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s"))
        lg.addHandler(h)
    return lg

