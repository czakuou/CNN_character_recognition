from tensorflow import keras

class OneCycleScheduler(keras.callbacks.Callback):
    '''
    1 Cycle scheduler implementation
    '''
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = last_rate or self.start_rate / 1000
        self.iteration = 0
        
    def interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)) / ((iter2 - iter1) + rate1)
    
    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0,
                                     self.half_iteration,
                                     self.start_rate,
                                     self.max_rate,)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration,
                                     2 * self.half_iteration,
                                     self.max_rate,
                                     self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration,
                                     self.iterations,
                                     self.start_rate,
                                     self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)
        