

set export
export OSTYPE := "linux-gnu"
export TQDM_MININTERVAL := "30"

main:
  . ./.venv/bin/activate
  python dreamer.py --configs crafter --task crafter_reward --logdir ./logdir/crafter
  
logs:
  tensorboard --logdir logdir/craftax 
