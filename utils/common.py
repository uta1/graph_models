def repeat_generator(generator, args):
    while True:
        for result in generator(*args):
            yield result
