#!/bin/bash
# Run reconstruction eval with Cohere adapter.
# Imports the custom model API in the same process before invoking inspect.
source "$(dirname "$0")/env.cohere.sh"
cd "$(dirname "$0")"
exec uv run python -c "
import evals.providers.cohere  # register @modelapi before inspect resolves the model
from inspect_ai._cli.main import main
import sys, os
sys.argv = [
    'inspect', 'eval', 'evals/reconstruction.py',
    '--model', 'cohere/openai/' + os.environ['OPENAI_MODEL'],
] + sys.argv[1:]
main()
" "$@"
