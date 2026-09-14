"""Run MEGSA pretraining, downstream training, or evaluation."""

import sys
from pathlib import Path

# Support both module execution and direct execution from any directory.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import Trainer
from src.args import get_args


def main(argv=None):
    """Dispatch the selected workflow without changing model computations.

    Args:
        argv: Optional command-line argument list.

    Returns:
        None.
    """
    args = get_args(argv)
    # Check required weights before loading or downloading datasets.
    if args.mode in ('train', 'eval'):
        required = [args.pretrain_dir / f'pretrain_model_{args.dataset}_{args.motif_num}.pth']
        if args.mode == 'eval':
            required.append(args.model_dir / f'model_{args.dataset}_{args.motif_num}.pth')
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(
                    f'Missing checkpoint: {path}. Run --mode pretrain before train, '
                    'or --mode all to train both stages before evaluation.'
                )
    trainer = Trainer(args)
    if args.mode in ('pretrain', 'all'):
        trainer.pretrain()
        trainer.save_pretrain()
    if args.mode in ('train', 'all'):
        trainer.train()
        trainer.save()
    if args.mode == 'eval':
        trainer.load()
    if args.mode in ('eval', 'all'):
        trainer.score()


if __name__ == '__main__':
    main()
