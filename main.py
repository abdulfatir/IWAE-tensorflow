import argparse
from iwae import IWAE


def set_args(parser):
    parser.add_argument('--batch_size', type=int,
                        default=100, help='input batch size')
    parser.add_argument('--z_dim', type=int, default=50,
                        help='latent vector dim')
    parser.add_argument('--n_steps', type=int, default=200000,
                        help='numbers of training steps')
    parser.add_argument('--k', type=int, default=5,
                        help='numbers of training steps')
    parser.add_argument('--test_k', type=int, default=5,
                        help='numbers of training steps')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = set_args(parser)
    args = parser.parse_args()
    print(args)
    model = IWAE(batch_size=args.batch_size, z_dim=args.z_dim, k=args.k, test_k=args.test_k, n_steps=args.n_steps)
    model.train()
    print('IWAE Bound:', model.compute_test_loss())
