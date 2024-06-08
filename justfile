

set export
export OSTYPE := "linux-gnu"
export TQDM_MININTERVAL := "30"

craftax_smaller:
  . ./.venv/bin/activate
  python dreamer.py --configs craftax_smaller --logdir ./logdir/craftax_smaller

craftax_small:
  . ./.venv/bin/activate
  python dreamer.py --configs craftax_small --logdir ./logdir/craftax_small2
  
logs:
  tensorboard --logdir logdir/craftax 
