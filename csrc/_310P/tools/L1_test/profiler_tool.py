# SPDX-License-Identifier: Apache-2.0
"""CPU-only msprof capability probe and frozen, shell-free rank command builder."""
import hashlib
import os
import re
import shlex
import subprocess
from pathlib import Path

SCHEMA = 'MF310P_MSPROF_TOOL_V1'
ALLOWED_OPTIONS = {'task-time','runtime-api','ascendcl','aicpu','hccl','aic-metrics','ai-core','msproftx','mstx'}
DEFAULT_OPTIONS = ('--task-time=on','--runtime-api=on','--ascendcl=on','--aicpu=on','--hccl=on')
RESERVED = {'application', 'output', 'devices', 'pid', 'model-id', 'iteration-id', 'delay', 'duration',
            'system-wide', 'sys-period', 'export', 'parse', 'query', 'type', 'show', 'help', 'version'}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def validate_options(options, help_text):
    names = set()
    for option in options:
        match = re.fullmatch(r'--([a-z][a-z0-9-]*)=([A-Za-z0-9_,.:-]+)', option)
        if not match or match[1] not in ALLOWED_OPTIONS or match[1] in RESERVED or match[1] in names:
            raise ValueError('msprof options must be distinct supported --name=value; launch/device/scope overrides forbidden')
        name, value = match.groups()
        if not re.search(r'(?<![\w-])--'+re.escape(name)+r'(?![\w-])', help_text):
            raise ValueError(f'msprof option absent from frozen --help: {name}')
        names.add(name)
    if len(names & {'msproftx','mstx'})!=1 or not any(x in options for x in ('--msproftx=on','--mstx=on')):
        raise ValueError('msprof requires MSTX enabled for exact measured-window attribution')
    return list(options)


def probe(binary, options=(), dependencies=()):
    path = Path(binary).resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError('msprof must be an executable file')
    help_run = subprocess.run([str(path),'--help'],capture_output=True,text=True,timeout=30,check=True)
    help_text = help_run.stdout + help_run.stderr
    for required in ('application','output'):
        if '--'+required not in help_text:
            raise ValueError('msprof application/output interface missing')
    options = list(options)
    automatic=[]
    explicit={option.split('=',1)[0] for option in options}
    for option in DEFAULT_OPTIONS:
        name=option.split('=',1)[0]
        if name not in explicit and re.search(r'(?<![\w-])'+re.escape(name)+r'(?![\w-])',help_text):
            options.append(option);automatic.append(option)
    if not any(x.startswith(('--msproftx=','--mstx=')) for x in options):
        flag = next((x for x in ('msproftx','mstx') if '--'+x in help_text),None)
        if flag is None:
            raise ValueError('installed msprof lacks an advertised MSTX option; do not guess scope')
        options.append('--'+flag+'=on')
    validate_options(options,help_text)
    version = subprocess.run([str(path),'--version'],capture_output=True,text=True,timeout=30)
    files = [path] + [Path(p).resolve(strict=True) for p in dependencies]
    return dict(schema=SCHEMA,binary=str(path),files={str(p):digest(p) for p in files},
        help_text=help_text,help_sha256=hashlib.sha256(help_text.encode()).hexdigest(),
        version=dict(returncode=version.returncode,text=version.stdout+version.stderr),options=options,
        auto_enabled=automatic,unadvertised_defaults=[x for x in DEFAULT_OPTIONS if x.split('=',1)[0] not in help_text],
        scope='application process capture; analysis restricted to validated L1 MSTX windows',
        identity_scope='declared tool/dependency files only',device_validation='NOT_RUN')


def validate_tool(spec):
    if spec.get('schema') != SCHEMA or not spec.get('files'):
        raise ValueError('frozen msprof probe required')
    path = Path(spec.get('binary',''))
    if not path.is_absolute() or str(path) not in spec['files'] or not os.access(path,os.X_OK):
        raise ValueError('frozen executable msprof path required')
    for name, expected in spec['files'].items():
        if not Path(name).is_absolute() or digest(name) != expected:
            raise ValueError(f'msprof tool/dependency identity changed: {name}')
    help_text = spec.get('help_text','')
    if hashlib.sha256(help_text.encode()).hexdigest() != spec.get('help_sha256'):
        raise ValueError('frozen msprof capability evidence changed')
    validate_options(spec.get('options',[]),help_text)


def command(spec, output, application):
    validate_tool(spec)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('msprof output must be fresh; no mixing or overwrite')
    if not application or any('\x00' in str(x) or '\n' in str(x) for x in application):
        raise ValueError('invalid msprof application argv')
    # msprof takes its application as one string; shell quoting preserves each
    # argument, while the outer process is always invoked with shell=False.
    return [spec['binary'],'--output='+str(output),'--application='+shlex.join(map(str,application)),*spec['options']]
