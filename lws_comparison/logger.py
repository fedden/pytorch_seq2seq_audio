import time

class TrainLogger():
    
    def __init__(self, size_of_dataset, epoch, number_epochs):
        self.size_of_dataset = size_of_dataset
        self.epoch = epoch
        self.number_epochs = number_epochs
        self.get_millis = lambda: int(round(time.time() * 1000))
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.current_batch = 0
        self.time_elapsed = 0
        
    def start_iteration(self):
        self.start = self.get_millis()
    
    def end_iteration(self, loss):
        self.time_elapsed += (self.get_millis() - self.start) 
        average_iteration_time = self.time_elapsed / (self.current_batch + 1)
        iterations_left = self.size_of_dataset - self.current_batch
        time_left_secs = int((average_iteration_time * iterations_left) / 1000) 
        self.minutes, self.seconds = divmod(time_left_secs, 60)
        self.current_batch += 1
        self.loss = loss
        self.total_loss += loss
        self.print()
        
    def print(self):
        if self.current_batch >= self.size_of_dataset:
            epoch_str = "epoch {:d}/{:d}, average loss {:.2f}" + (" " * 50)
            epoch_str = epoch_str.format(self.epoch, self.number_epochs, self.total_loss / self.current_batch)
            print(epoch_str)
            self.epoch += 1
            self.reset()
        else:
            epoch_str = "epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.2f}, approx time left {:d} mins {:d} secs"
            epoch_str += (' ' * 20)
            epoch_str = epoch_str.format(self.epoch, 
                                         self.number_epochs, 
                                         self.current_batch, 
                                         self.size_of_dataset, 
                                         self.loss, 
                                         self.minutes, 
                                         self.seconds)
            print(epoch_str, end='\r')