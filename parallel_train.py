from concurrent.futures import ProcessPoolExecutor

def train(settings):
    pass

settings = None

pool = ProcessPoolExecutor(4)
future = pool.submit(train, settings)
done = future.done()
print(done)
result = future.result()
print(result)

# stuff disappeared from here.