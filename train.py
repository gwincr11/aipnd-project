
import argparse
from network import trainNetwork

parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument('--save_dir', action="store", default=".", help='Directory to save checkpoints')
parser.add_argument('--arch', action="store", default="vgg19", help='architecture')
parser.add_argument('--learning_rate', action="store", default=0.001, type=float, help='learning rate')
parser.add_argument('--hidden_units', action="store", default=2058, type=int, help='hidden units')
parser.add_argument('--epochs', action="store", type=int, default=10, help='epochs')
parser.add_argument('--gpu', action="store_true", default=False, help='use GPU for training')


def main():
    args = parser.parse_args()
    trainNetwork(save_dir=args.save_dir, arch=args.arch, learning_rate=args.learning_rate, hidden_units=args.hidden_units, epochs=args.epochs, gpu=args.gpu)

if __name__ == '__main__':
    main()
