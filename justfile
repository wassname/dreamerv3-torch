

set export
export OSTYPE := "linux-gnu"
export TQDM_MININTERVAL := "30"

main:
  . ./.venv/bin/activate
  python dreamer.py --configs craftax_smaller --logdir ./logdir/craftax_smaller
  
logs:
  tensorboard --logdir logdir/craftax 
