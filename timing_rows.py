import pandas as pd, subprocess, os
src  = pd.read_csv(os.path.expanduser('~/env_QGA/data/pleiades.csv'))
base = os.path.expanduser('~/env_QGA/scripts/baseline.py')
gro  = os.path.expanduser('~/env_QGA/scripts/grover_emulation.py')

sizes = [2000, 5000, 10000]
for n in sizes:
    tmp = f'/tmp/sample_{n}.csv'
    src.sample(n, random_state=42).to_csv(tmp, index=False)

    subprocess.run(['python', base, tmp, '10'], check=True)
    subprocess.run(['python', gro , tmp, '10'], check=True)

    (pd.read_csv(os.path.expanduser('~/env_QGA/results/baseline.csv'))
       .assign(N=n)
       .to_csv(os.path.expanduser('~/env_QGA/results/baseline_sub.csv'),
               mode='a',
               header=not os.path.exists(os.path.expanduser('~/env_QGA/results/baseline_sub.csv')),
               index=False))

    (pd.read_csv(os.path.expanduser('~/env_QGA/results/grover.csv'))
       .assign(N=n)
       .to_csv(os.path.expanduser('~/env_QGA/results/grover_sub.csv'),
               mode='a',
               header=not os.path.exists(os.path.expanduser('~/env_QGA/results/grover_sub.csv')),
               index=False))
