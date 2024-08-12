import pickle
from configparser import ConfigParser
from os.path import expanduser, join

from neuralop import LpLoss, H1Loss


class Parameters:
    #  Initializer выполняется перед! основной программой.
    #  Private Instance or static Class attribute. Переменные должны начинаться с двух подчеркиваний.
    train_loss: LpLoss | H1Loss = LpLoss(d=2, p=2)
    n_epochs: int = 10
    device: str = "cpu"
    n_layers: int = 3
    file_name: str = '/home/dimitri/Documents/fno_darcy.py.ini'

    def __init__(self, file_name='/home/dimitri/Documents/fno_darcy.py.ini') -> None:
        # TODO: Load .ini file
        self.file_name = file_name
        config_parser: ConfigParser = ConfigParser()
        config_parser.read(self.file_name)
        self.n_epochs = config_parser.getint(section='Darcy', option='n_epochs')
        self.device = config_parser.get(section='Darcy', option='device')
        self.n_layers = config_parser.getint(section='Darcy', option='n_layers')
        train_loss_str: str = config_parser.get(section='Darcy', option='train_loss')
        self.train_loss = self.__private_method(train_loss_str)

    # Methods
    def __private_method(self, train_loss_str: str) -> LpLoss | H1Loss:
        if train_loss_str == 'LpLoss':
            return LpLoss(d=2, p=2)
        elif train_loss_str == 'H1Loss':
            return H1Loss(d=2)
        else:
            raise ValueError(f"Unknown loss function: {train_loss_str}")

    # Accessor( = getter) methods
    def get_parameters(self) -> tuple[LpLoss | H1Loss, int, str, int]:
        return self.train_loss, self.n_epochs, self.device, self.n_layers


if __name__ == '__main__':
    parameters: Parameters = Parameters(file_name='/home/dimitri/Documents/fno_darcy.py.ini')
    shape: tuple[LpLoss | H1Loss, int, str, int] = parameters.get_parameters()
    # Save the shape of the array
    with open(join(expanduser("~"), "Documents", 'file_name.pickle'), 'wb') as f:
        pickle.dump(shape, f)
    print(shape)
pass  # Press Ctrl+8 to toggle the breakpoint.
