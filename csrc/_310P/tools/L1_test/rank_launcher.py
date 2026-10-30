# SPDX-License-Identifier: Apache-2.0
"""torchrun rank shim: one msprof process per rank, same guarded process group."""
import argparse
import json
import os
import sys
from pathlib import Path

from .profiler_tool import command


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--job',required=True)
    args=parser.parse_args()
    run=json.loads(args.run.read_text())
    job=next(j for j in run['plan']['jobs'] if j['id']==args.job)
    if job['stage']!='msprof' or os.environ.get('WORLD_SIZE')!='2' or os.environ.get('RANK') not in ('0','1'):
        raise ValueError('msprof rank launcher only accepts its explicit diagnostic TP2 job')
    rank=int(os.environ['RANK']);directory=args.run.parent/job['id']
    application=[sys.executable,'-m','L1_test.worker','--run',str(args.run.resolve()),'--job',job['id']]
    argv=command(run['manifest']['msprof'],directory/f'msprof-rank{rank}',application)
    with (directory/f'msprof-launch-rank{rank}.json').open('x') as stream:
        json.dump(dict(rank=rank,argv=argv,application=application,scope='whole process; MSTX-selected analysis'),stream,indent=2)
    os.execv(argv[0],argv)  # No new process group/session or shell.


if __name__=='__main__':
    main()
