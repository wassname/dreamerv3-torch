

set export
export OSTYPE := "linux-gnu"
export TQDM_MININTERVAL := "30"

main:
  . ./.venv/bin/activate
  python dreamer.py --configs craftax_smaller --logdir ./logdir/crafterer
  
logs:
  tensorboard --logdir logdir/craftax 
