import argparse
from network import predict

parser = argparse.ArgumentParser(conflict_handler='resolve')
# The first argument is the path to the image
parser.add_argument('image_path', action="store", help='Path to image')
parser.add_argument('--model_path', action="store", default="checkpoint.pth", help='Path to model checkpoints')
parser.add_argument('--top-k', action="store", default=5, type=int, help='number of top classes')
parser.add_argument('--category-names', action="store", default='cat_to_name.json', help='category names')
parser.add_argument('--gpu', action="store_true", default=False, help='use GPU for prediction')

def main():
    args = parser.parse_args()
    predict(image_path=args.image_path, model_path=args.model_path, topk=args.top_k, category_names=args.category_names, gpu=args.gpu)

if __name__ == '__main__':
    main()
